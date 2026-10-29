"""Tests du chatbot minimal avec recherches et modèles simulés, sans réseau."""

import json
from copy import deepcopy

import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.RAG import chatbot


def make_reference(uid):
    """Créer une fiche structurée avec horaires et types variés à préserver."""
    return Document(page_content=f"Événement {uid}", metadata={
        "uid": uid,
        "event": {"uid": uid, "title": f"Festival {uid}", "age_min": 6,
                  "timings": [{"begin": "2026-11-10T20:00:00+01:00"}],
                  "extra": {"available": True}, "online_access_link": None},
    })


@pytest.fixture(autouse=True)
def block_real_models(monkeypatch, tmp_path):
    """Empêcher les tests d'initialiser les clients réels, y compris leur tokenizer."""
    monkeypatch.setattr(chatbot, "MistralAIEmbeddings", Mock(side_effect=AssertionError("Modèle réel interdit")))
    monkeypatch.setattr(chatbot, "ChatMistralAI", Mock(side_effect=AssertionError("Modèle réel interdit")))
    monkeypatch.setattr(chatbot, "load_dotenv", Mock())
    monkeypatch.setattr(chatbot, "RAG_DIR", tmp_path)


def test_search_events_deduplicates_uids_and_limits_distinct_events():
    """Conserver l'ordre des premiers UID et cinq fiches au maximum."""
    store = Mock()
    store.similarity_search_with_score.return_value = [
        (Document(page_content="Passage", metadata={"uid": uid}), np.float32(distance))
        for uid, distance in zip([9, 9, "9", 3, 7, 7, 5, 1, 2],
                                 [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125])
    ]
    references = {uid: make_reference(uid) for uid in [9, 3, 7, 5, 1, 2]}
    before = {uid: deepcopy(reference.metadata) for uid, reference in references.items()}
    store.docstore.search.side_effect = lambda key: references[int(key.removeprefix("event_"))]

    events = chatbot.search_events("  Un festival ?  ", store)

    assert [reference.metadata["uid"] for reference, _, _ in events] == [9, 3, 7, 5, 1]
    assert [rank for _, rank, _ in events] == [1, 2, 3, 4, 5]
    assert [distance for _, _, distance in events] == [0.125, 0.5, 0.625, 0.875, 1.0]
    assert all(reference is references[reference.metadata["uid"]] for reference, _, _ in events)
    chatbot.build_context(events)
    chatbot.build_markdown_context(events)
    assert {uid: reference.metadata for uid, reference in references.items()} == before
    store.similarity_search.assert_not_called()
    store.similarity_search_with_score.assert_called_once_with("Un festival ?", k=20)
    assert [call.args[0] for call in store.docstore.search.call_args_list] == ["event_9", "event_3", "event_7", "event_5", "event_1"]


def test_build_context_preserves_all_reference_metadata():
    """Garder l'intégralité des données, notamment timings, sans changer leurs types JSON."""
    reference = make_reference(12345)
    context = json.loads(chatbot.build_context([(reference, 1, 0.42)]))

    assert context == [{"metadata": reference.metadata, "faiss_rank": 1, "faiss_distance": 0.42}]
    assert context[0]["metadata"]["event"]["timings"] == reference.metadata["event"]["timings"]
    assert context[0]["metadata"]["event"]["extra"]["available"] is True
    assert context[0]["metadata"]["event"]["online_access_link"] is None


def test_search_events_reports_missing_reference():
    """Signaler un ancien index sans fiche complète au lieu d'inventer l'événement."""
    store = Mock()
    store.similarity_search_with_score.return_value = [(Document(page_content="Jazz", metadata={"uid": 12345}), 0.42)]
    store.docstore.search.return_value = "ID event_12345 not found."

    with pytest.raises(ValueError, match="event_12345"):
        chatbot.search_events("Du jazz ?", store)


def test_generate_answer_without_results_does_not_call_llm():
    """Ne pas payer une génération lorsqu'aucun chunk n'est retrouvé."""
    store, llm = Mock(), Mock()
    store.similarity_search_with_score.return_value = []

    answer = chatbot.generate_answer("Du jazz ?", store, llm, "Consignes")

    assert "Aucun événement" in answer
    store.docstore.search.assert_not_called()
    llm.invoke.assert_not_called()


def test_generate_answer_sends_literal_prompt_and_full_context(monkeypatch):
    """Séparer instructions et données, en conservant les accolades du prompt libre."""
    monkeypatch.setattr(chatbot, "USE_MARKDOWN_CONTEXT", False)
    store, llm = Mock(), Mock()
    reference = make_reference(12345)
    store.similarity_search_with_score.return_value = [(Document(page_content="Jazz", metadata={"uid": 12345}), 0.42)]
    store.docstore.search.return_value = reference
    llm.invoke.return_value = AIMessage(content="Voici le festival.")
    prompt = 'Instructions libres avec {"exemple": true}'

    assert chatbot.generate_answer("Du jazz ?", store, llm, prompt) == "Voici le festival."
    messages = llm.invoke.call_args.args[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage) and messages[0].content == prompt
    assert isinstance(messages[1], HumanMessage)
    assert "Du jazz ?" in messages[1].content
    payload = messages[1].content.split("Fiches d'événements (JSON) :\n", 1)[1]
    assert json.loads(payload) == [{"metadata": reference.metadata, "faiss_rank": 1, "faiss_distance": 0.42}]


def test_markdown_preserves_business_data_without_mutating_source():
    """Conserver descriptions, horaires, zéro, false et URL ; retirer les embeds."""
    reference = make_reference(12345)
    reference.metadata["event"].update({
        "description": "Concert\npour tous", "age_min": 0,
        "accessibility": {"mi": False},
        "event_links": [{"link": "https://example.org/a/b", "data": {
            "title": "Programme", "description": "Tout le programme",
            "html": "<iframe>TECHNIQUE</iframe>", "script": "SCRIPT_TECHNIQUE",
        }}],
    })
    before = deepcopy(reference.metadata)
    context = chatbot.build_markdown_context([(reference, 1, 0.42)])
    for text in ("Festival 12345", "UID 12345", "Concert\npour tous", "Âge minimum\n", "2026-11-10T20:00:00+01:00", "false", "Non renseigné", "https://example.org/a/b", "Programme", "Tout le programme"):
        assert text in context
    assert "\n0" in context
    assert "TECHNIQUE" not in context
    assert reference.metadata == before
    assert "Classement FAISS : 1\nDistance FAISS du meilleur chunk : 0.42" in context


def test_generate_answer_uses_optional_markdown_context(monkeypatch):
    """Changer uniquement le contexte utilisateur, sans toucher au prompt système."""
    monkeypatch.setattr(chatbot, "USE_MARKDOWN_CONTEXT", True)
    reference = make_reference(12345)
    monkeypatch.setattr(chatbot, "search_events", Mock(return_value=[(reference, 1, 0.42)]))
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="Réponse")
    chatbot.generate_answer("Du jazz ?", Mock(), llm, "Consignes inchangées")
    messages = llm.invoke.call_args.args[0]
    assert messages[0].content == "Consignes inchangées"
    assert messages[1].content.endswith(chatbot.build_markdown_context([(reference, 1, 0.42)]))
    assert "Fiches d'événements (Markdown)" in messages[1].content


@pytest.mark.parametrize("markdown", [False, True])
def test_generation_sends_full_selected_events_and_scores_without_export(monkeypatch, tmp_path, markdown):
    """Vérifier les messages transmis au LLM simulé sans export sur disque."""
    monkeypatch.setattr(chatbot, "USE_MARKDOWN_CONTEXT", markdown)
    store, llm = Mock(), Mock()
    references = {uid: make_reference(uid) for uid in range(1, 7)}
    before = {uid: deepcopy(reference.metadata) for uid, reference in references.items()}
    store.similarity_search_with_score.return_value = [
        (Document(page_content="Chunk fictif", metadata={"uid": uid}), distance)
        for uid, distance in [(1, 0.125), (1, 0.25), (2, 0.375), (3, 0.5), (4, 0.625), (5, 0.75), (6, 0.875)]
    ]
    store.docstore.search.side_effect = lambda key: references[int(key.removeprefix("event_"))]
    llm.invoke.return_value = AIMessage(content="Réponse fictive")

    chatbot.generate_answer("Du jazz ?", store, llm, "Prompt système fictif")

    assert not (tmp_path / "debug_requests").exists()
    messages = llm.invoke.call_args.args[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Question :\nDu jazz ?" in messages[1].content
    context = messages[1].content.split(" :\n", 2)[2]
    distances = [0.125, 0.375, 0.5, 0.625, 0.75]
    if markdown:
        assert context.count("## Festival ") == 5
        assert "Festival 6" not in context
        for uid, distance in enumerate(distances, 1):
            assert f"## Festival {uid} — UID {uid}" in context
            assert f"Classement FAISS : {uid}\nDistance FAISS du meilleur chunk : {distance}" in context
        assert "2026-11-10T20:00:00+01:00" in context
    else:
        assert json.loads(context) == [
            {"metadata": references[uid].metadata, "faiss_rank": uid, "faiss_distance": distance}
            for uid, distance in enumerate(distances, 1)
        ]
    assert {uid: reference.metadata for uid, reference in references.items()} == before


@pytest.mark.parametrize("url, data", [
    ("https://youtu.be/abc", {}),
    ("https://www.youtube.com/watch?v=abc", {}),
    ("https://player.vimeo.com/video/123", {}),
    ("https://example.org/video", {"type": "video"}),
    ("https://example.org/page", {"url": "https://www.youtube.com/watch?v=abc"}),
])
def test_markdown_removes_video_descriptions_only(url, data):
    """Retirer le bruit des clips sans toucher aux textes métier ni aux sources."""
    reference = make_reference(12345)
    reference.metadata["event"].update({
        "long_description": "Présentation complète\navec informations pratiques.",
        "event_links": [
            {"link": url, "description": "PAROLES ET CREDITS",
             "data": {**data, "title": "Vidéo de présentation", "description": "PAROLES ET CREDITS"}},
            {"link": "https://example.org/programme", "data": {
                "title": "Programme", "description": "Réservation obligatoire avant vendredi"}},
            {"link": "mailto:contact@example.org"},
        ],
        "custom_field": {"important": False},
    })
    before = deepcopy(reference.metadata)

    context = chatbot.build_markdown_context([(reference, 1, 0.42)])

    assert "PAROLES ET CREDITS" not in context
    for text in (url, "Vidéo de présentation", "Programme", "Réservation obligatoire avant vendredi",
                 "mailto:contact@example.org", "Présentation complète\navec informations pratiques.",
                 "### Présentation", "### Dates et horaires", "### Public et accessibilité",
                 "### Conditions et liens utiles", "### Autres informations", "custom field", "false"):
        assert text in context
    assert reference.metadata == before
    assert json.loads(chatbot.build_context([(reference, 1, 0.42)]))[0]["metadata"] == before


def test_load_resources_reads_prompt_file_and_uses_local_index(monkeypatch, tmp_path):
    """Lire le prompt depuis son fichier et charger notre paire locale sans clients réels."""
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text('Consignes libres : {"test": true}', encoding="utf-8")
    index_dir = tmp_path / "vector_store"
    index_dir.mkdir()
    for name in ("Vector_Store.faiss", "DocStore_LangChain.pkl"):
        (index_dir / name).write_bytes(b"fichier fictif, jamais deserialise")
    monkeypatch.setattr(chatbot, "PROMPT_PATH", prompt_path)
    monkeypatch.setattr(chatbot, "INDEX_DIR", index_dir)
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-fictive")
    embeddings, llm, store = Mock(), Mock(), Mock()
    store.index.ntotal, store.index.d = 3, 1024
    constructor = Mock(return_value=embeddings)
    monkeypatch.setattr(chatbot, "MistralAIEmbeddings", constructor)
    monkeypatch.setattr(chatbot, "ChatMistralAI", Mock(return_value=llm))
    index_reader = Mock(return_value=store.index)
    docstore, mapping = Mock(), {0: "12345_1"}
    monkeypatch.setattr(chatbot.faiss, "read_index", index_reader)
    monkeypatch.setattr(chatbot.pickle, "load", Mock(return_value=(docstore, mapping)))
    loader = Mock(return_value=store)
    monkeypatch.setattr(chatbot, "FAISS", loader)

    loaded_store, loaded_llm, prompt = chatbot.load_resources()

    assert loaded_store is store and loaded_llm is llm
    assert prompt == prompt_path.read_text(encoding="utf-8")
    index_reader.assert_called_once_with(str(index_dir / "Vector_Store.faiss"))
    loader.assert_called_once_with(embeddings, store.index, docstore, mapping)
    constructor.assert_called_once_with(model="mistral-embed", api_key="cle-fictive", max_retries=0)


def test_empty_question_does_not_trigger_search():
    """Refuser une question vide avant toute vectorisation payante."""
    store = Mock()
    with pytest.raises(ValueError, match="vide"):
        chatbot.search_events("   ", store)
    store.similarity_search_with_score.assert_not_called()


def test_terminal_handles_empty_question_and_context_error(monkeypatch, capsys):
    """Afficher un refus API compréhensible sans tronquer les données ni quitter brutalement."""
    store, llm = Mock(), Mock()
    monkeypatch.setattr(chatbot, "load_resources", Mock(return_value=(store, llm, "Consignes")))
    questions = iter(["", "Du jazz ?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/chat"))
    generator = Mock(side_effect=httpx.HTTPStatusError("contexte trop long", request=response.request, response=response))
    monkeypatch.setattr(chatbot, "generate_answer", generator)

    chatbot.main()

    output = capsys.readouterr().out
    assert "question non vide" in output
    assert "taille du contexte complet" in output
    assert "Aucune donnée n'a été tronquée" in output
    generator.assert_called_once_with("Du jazz ?", store, llm, "Consignes")


@pytest.mark.parametrize("problem", ["empty-prompt", "missing-directory", "missing-file", "missing-key", "empty-index", "wrong-dimension"])
def test_load_resources_rejects_invalid_local_configuration(monkeypatch, tmp_path, problem):
    """Vérifier les garde-fous avant de chercher ou générer une réponse."""
    prompt = tmp_path / "prompt.md"
    prompt.write_text(" " if problem == "empty-prompt" else "Consignes fictives", encoding="utf-8")
    directory = tmp_path / "vector_store"
    if problem != "missing-directory":
        directory.mkdir()
        (directory / "Vector_Store.faiss").write_bytes(b"index fictif")
        if problem != "missing-file":
            (directory / "DocStore_LangChain.pkl").write_bytes(b"pickle fictif jamais deserialise")
    monkeypatch.setattr(chatbot, "PROMPT_PATH", prompt)
    monkeypatch.setattr(chatbot, "INDEX_DIR", directory)
    if problem == "missing-key":
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_API_KEY", "cle-fictive")
    reader = Mock()
    monkeypatch.setattr(chatbot.faiss, "read_index", reader)
    monkeypatch.setattr(chatbot.pickle, "load", Mock(return_value=(Mock(), {})))
    if problem in ("empty-index", "wrong-dimension"):
        store = Mock()
        store.index.ntotal = 0 if problem == "empty-index" else 1
        store.index.d = 512 if problem == "wrong-dimension" else 1024
        monkeypatch.setattr(chatbot, "MistralAIEmbeddings", Mock(return_value=Mock()))
        monkeypatch.setattr(chatbot, "FAISS", Mock(return_value=store))
    with pytest.raises(ValueError):
        chatbot.load_resources()
    chatbot.ChatMistralAI.assert_not_called()
    if problem not in ("empty-index", "wrong-dimension"):
        chatbot.MistralAIEmbeddings.assert_not_called()
        reader.assert_not_called()


@pytest.mark.parametrize("error, expected", [
    (httpx.ConnectError("Connexion simulée"), "Connexion à Mistral impossible"),
    (ValueError("Réponse vide simulée"), "Réponse impossible"),
])
def test_terminal_reports_generation_errors_and_accepts_next_input(monkeypatch, capsys, error, expected):
    monkeypatch.setattr(chatbot, "load_resources", Mock(return_value=(Mock(), Mock(), "Consignes")))
    monkeypatch.setattr(chatbot, "generate_answer", Mock(side_effect=error))
    answers = iter(["Du jazz ?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    chatbot.main()
    output = capsys.readouterr().out
    assert expected in output
    assert "Au revoir" in output


@pytest.mark.parametrize("error", [FileNotFoundError("Index absent"), httpx.ConnectError("Erreur simulée")])
def test_terminal_reports_initialization_error_without_asking_question(monkeypatch, capsys, error):
    monkeypatch.setattr(chatbot, "load_resources", Mock(side_effect=error))
    ask = Mock(side_effect=AssertionError("Le terminal ne doit pas demander de question"))
    monkeypatch.setattr("builtins.input", ask)
    chatbot.main()
    assert "Initialisation impossible" in capsys.readouterr().out
    ask.assert_not_called()


def test_terminal_reports_mistral_quota_error(monkeypatch, capsys):
    request = httpx.Request("POST", "https://example.invalid/chat")
    error = httpx.HTTPStatusError("Quota simulé", request=request, response=httpx.Response(429, request=request))
    monkeypatch.setattr(chatbot, "load_resources", Mock(return_value=(Mock(), Mock(), "Consignes")))
    monkeypatch.setattr(chatbot, "generate_answer", Mock(side_effect=error))
    questions = iter(["Du jazz ?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    chatbot.main()
    assert "HTTP 429" in capsys.readouterr().out

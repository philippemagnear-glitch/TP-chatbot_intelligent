"""Tests de collecte avec chatbot simulé, sans modèles réels ni appels réseau."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.RAG.evaluation import evaluate as evaluation


@pytest.fixture(autouse=True)
def isolated_evaluation(monkeypatch, tmp_path):
    """Rediriger les résultats et remplacer les fonctions pouvant appeler Mistral."""
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(evaluation.chatbot, "load_resources", Mock(return_value=("store", "llm", "Prompt fictif {modifiable}")))
    monkeypatch.setattr(evaluation.chatbot, "generate_answer", Mock(return_value="Réponse fictive"))


def test_load_questions_reads_the_ten_annotated_questions():
    """Contrôler le jeu livré, son ordre et ses annotations, sans dataset OpenAgenda."""
    questions = evaluation.load_questions()

    assert [question["id"] for question in questions] == [f"Q{n:02}" for n in range(1, 11)]
    assert questions[0]["question"] == "Où se déroule le ciné-concert C.A.V.E.R.N.E.S. du 28 octobre 2026 ?"
    assert questions[6]["expected_uids"] == [33128226, 65028914]
    assert questions[9]["category"] == "Question piège"
    assert all(set(q) == {"id", "category", "question", "expected_answer", "expected_uids"} for q in questions)


@pytest.mark.parametrize("problem", ["missing-question", "duplicate-id", "missing-answer"])
def test_load_questions_rejects_incomplete_or_inconsistent_game(monkeypatch, tmp_path, problem):
    """Refuser des annotations incomplètes avant de lancer une collecte."""
    questions = evaluation.load_questions()
    if problem == "missing-question":
        questions.pop()
    elif problem == "duplicate-id":
        questions[-1]["id"] = questions[0]["id"]
    else:
        questions[0].pop("expected_answer")
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    monkeypatch.setattr(evaluation, "QUESTIONS_PATH", path)

    with pytest.raises(ValueError):
        evaluation.load_questions()


@pytest.mark.parametrize("markdown", [False, True])
def test_evaluate_records_answers_prompt_parameters_and_progress(capsys, monkeypatch, markdown):
    """Vérifier la lisibilité du rapport et l'appel unique au chargement des ressources."""
    monkeypatch.setattr(evaluation.chatbot, "USE_MARKDOWN_CONTEXT", markdown)
    questions = evaluation.load_questions()
    output = evaluation.evaluate("v1")
    content = output.read_text(encoding="utf-8")

    assert output.name == "v1.txt"
    assert f"USE_MARKDOWN_CONTEXT : {markdown}" in content
    for text in ("VERSION : v1", "DATE :", evaluation.chatbot.CHAT_MODEL, "SEARCH_K : 20", "MAX_EVENTS : 5", "Prompt fictif {modifiable}"):
        assert text in content
    assert content.count("STATUT : RÉPONSE OBTENUE") == 10
    for question in questions:
        assert f"QUESTION {question['id']} — {question['category']}" in content
        assert question["question"] in content and question["expected_answer"] in content
    evaluation.chatbot.load_resources.assert_called_once_with()
    assert [call.args for call in evaluation.chatbot.generate_answer.call_args_list] == [
        (question["question"], "store", "llm", "Prompt fictif {modifiable}") for question in questions
    ]
    progress = capsys.readouterr().out
    assert "Question 1/10" in progress and "Question 10/10" in progress


def test_evaluate_refuses_overwrite_before_initializing_chatbot():
    """Conserver un résultat existant et éviter tout appel supplémentaire."""
    evaluation.RESULTS_DIR.mkdir()
    output = evaluation.RESULTS_DIR / "v1.txt"
    output.write_text("Résultat à conserver", encoding="utf-8")

    with pytest.raises(FileExistsError, match="autre nom"):
        evaluation.evaluate("v1")
    assert output.read_text(encoding="utf-8") == "Résultat à conserver"
    evaluation.chatbot.load_resources.assert_not_called()
    evaluation.chatbot.generate_answer.assert_not_called()


def test_evaluate_records_api_error_and_continues():
    """Distinguer une erreur technique d'une réponse et poursuivre les autres questions."""
    request = httpx.Request("POST", "https://example.invalid/chat")
    error = httpx.HTTPStatusError("Rate limit exceeded", request=request, response=httpx.Response(429, request=request))
    evaluation.chatbot.generate_answer.side_effect = [error] + ["Réponse suivante"] * 9

    content = evaluation.evaluate("v_erreur").read_text(encoding="utf-8")

    assert content.count("STATUT : ERREUR TECHNIQUE") == 1
    assert "HTTPStatusError" in content
    assert content.count("STATUT : RÉPONSE OBTENUE") == 9
    assert evaluation.chatbot.generate_answer.call_count == 10


def test_evaluate_preserves_completed_answer_on_keyboard_interrupt():
    """Conserver les réponses terminées quand l'utilisateur interrompt la collecte."""
    evaluation.chatbot.generate_answer.side_effect = ["Première réponse", KeyboardInterrupt()]

    content = evaluation.evaluate("v_interrompue").read_text(encoding="utf-8")

    assert "Première réponse" in content
    assert "STATUT : INTERROMPU PAR L'UTILISATEUR" in content
    assert "QUESTION Q03" not in content
    assert evaluation.chatbot.generate_answer.call_count == 2


def test_evaluate_rejects_version_with_path_traversal():
    """Empêcher le nom de version de déplacer le résultat hors du dossier prévu."""
    with pytest.raises(ValueError, match="Version incorrecte"):
        evaluation.evaluate("../v1")
    evaluation.chatbot.load_resources.assert_not_called()

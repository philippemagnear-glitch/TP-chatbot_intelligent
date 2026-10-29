"""Tests fonctionnels HTTP avec RAG simulé, sans serveur ni appel Mistral.

Depuis la racine : uv run pytest tests/api_test.py -v
FastAPI doit être disponible dans l'environnement du projet.
Les seules écritures des tests de reconstruction se font dans tmp_path.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from tenacity import Future, RetryError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.API import rag_api as api


# ---------- Isolation des données et des services ----------
@pytest.fixture
def client(monkeypatch, tmp_path):
    """Exercer les vraies routes, en remplaçant leurs dépendances externes."""
    monkeypatch.setattr(api, "load_dotenv", Mock())
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "jeton-admin-fictif")
    monkeypatch.setattr(api.chatbot, "INDEX_DIR", tmp_path / "Data" / "vector_store")
    monkeypatch.setattr(api.chatbot, "load_resources", Mock(
        return_value=("index simulé", "client simulé", "prompt simulé"),
    ))
    monkeypatch.setattr(api.chatbot, "generate_answer", Mock(return_value="Un concert de jazz est proposé."))
    # Un test ne doit jamais construire un vrai index ou lire un vrai pickle.
    monkeypatch.setattr(api.build_index, "main", Mock(side_effect=AssertionError("Construction réelle interdite")))
    monkeypatch.setattr(api.chatbot.faiss, "read_index", Mock(side_effect=AssertionError("Lecture FAISS réelle interdite")))
    monkeypatch.setattr(api.chatbot.pickle, "load", Mock(side_effect=AssertionError("Lecture pickle réelle interdite")))
    monkeypatch.setattr(api.chatbot, "MistralAIEmbeddings", Mock(side_effect=AssertionError("Appel Mistral interdit")))
    monkeypatch.setattr(api.chatbot, "ChatMistralAI", Mock(side_effect=AssertionError("Appel Mistral interdit")))
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer jeton-admin-fictif"}


# ---------- Disponibilité et questions ----------
def test_health_does_not_reload_rag(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "API disponible."}
    api.chatbot.load_resources.assert_called_once_with()
    api.chatbot.generate_answer.assert_not_called()


def test_question_returns_answer_and_passes_question_to_rag(client):
    response = client.post("/ask", json={"question": "  Du jazz à Toulouse ?  "})

    assert response.status_code == 200
    assert response.json() == {"answer": "Un concert de jazz est proposé."}
    api.chatbot.load_resources.assert_called_once_with()
    api.chatbot.generate_answer.assert_called_once_with(
        "Du jazz à Toulouse ?", "index simulé", "client simulé", "prompt simulé",
    )


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": " \n "}, {"question": 123},
    {"question": None}, {"question": ["jazz"]}, {"question": "a" * 4001},
    {"question": "Du jazz ?", "model": "autre-modèle"},
])
def test_invalid_question_is_rejected_without_calling_rag(client, payload):
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()
    api.chatbot.load_resources.assert_called_once_with()
    api.chatbot.generate_answer.assert_not_called()


def test_malformed_json_is_rejected(client):
    response = client.post("/ask", content='{"question":',
                           headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    api.chatbot.load_resources.assert_called_once_with()


@pytest.mark.parametrize("error", [
    FileNotFoundError("/chemin/interne/index.faiss"),
    ValueError("Index illisible : secret-fictif"),
])
def test_unavailable_index_at_startup_returns_safe_error(client, error, caplog):
    api.chatbot.load_resources.side_effect = error

    # Une nouvelle durée de vie simule un démarrage avec un index absent/illisible.
    with caplog.at_level(logging.ERROR, logger=api.__name__):
        with TestClient(api.app) as unavailable_client:
            response = unavailable_client.post("/ask", json={"question": "Du jazz ?"})
            assert unavailable_client.get("/health").status_code == 200

    assert response.status_code == 503
    assert response.json() == {"detail": "Les ressources RAG sont indisponibles."}
    assert (api.__name__, logging.ERROR,
            f"Chargement initial des ressources RAG impossible ({type(error).__name__}).") in caplog.record_tuples
    assert str(error) not in caplog.text
    assert str(error) not in response.text
    api.chatbot.generate_answer.assert_not_called()


@pytest.mark.parametrize("upstream_status, expected_status", [(429, 429), (401, 502), (500, 502)])
def test_mistral_http_error_does_not_expose_technical_details(client, upstream_status, expected_status):
    request = httpx.Request("POST", "https://example.invalid/chat")
    response = httpx.Response(upstream_status, request=request)
    api.chatbot.generate_answer.side_effect = httpx.HTTPStatusError(
        "secret-fictif : détails internes", request=request, response=response,
    )

    result = client.post("/ask", json={"question": "Du jazz ?"})

    assert result.status_code == expected_status
    assert set(result.json()) == {"detail"}
    assert "secret-fictif" not in result.text
    assert "example.invalid" not in result.text


def test_mistral_timeout_returns_gateway_timeout(client):
    api.chatbot.generate_answer.side_effect = httpx.ReadTimeout("secret-fictif")

    response = client.post("/ask", json={"question": "Du jazz ?"})

    assert response.status_code == 504
    assert "secret-fictif" not in response.text


# ---------- Protection et reconstruction ----------
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer mauvais-jeton"}])
def test_rebuild_requires_valid_admin_token(client, headers):
    response = client.post("/rebuild", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    api.build_index.main.assert_not_called()


def test_rebuild_is_disabled_without_configured_token(client, monkeypatch, admin_headers):
    monkeypatch.delenv("RAG_ADMIN_TOKEN")

    response = client.post("/rebuild", headers=admin_headers)

    assert response.status_code == 503
    api.build_index.main.assert_not_called()


def test_successive_questions_reuse_startup_resources(client):
    for question in ("Du jazz ?", "Un spectacle pour enfants ?"):
        assert client.post("/ask", json={"question": question}).status_code == 200

    api.chatbot.load_resources.assert_called_once_with()
    assert api.chatbot.generate_answer.call_count == 2
    assert all(call.args[1:] == ("index simulé", "client simulé", "prompt simulé")
               for call in api.chatbot.generate_answer.call_args_list)


def test_old_question_route_is_removed(client):
    assert client.post("/question", json={"question": "Du jazz ?"}).status_code == 404
    api.chatbot.generate_answer.assert_not_called()


def test_authorized_rebuild_publishes_simulated_index(client, admin_headers):
    """Tester le remplacement de fichiers fictifs, sans calculer de vecteurs."""
    target = api.chatbot.INDEX_DIR
    target.mkdir(parents=True)
    (target / "ancien.txt").write_text("Ancien index fictif", encoding="utf-8")

    def fake_build(destination):
        destination.mkdir()
        (destination / "Vector_Store.faiss").write_bytes(b"vecteurs fictifs")
        (destination / "DocStore_LangChain.pkl").write_bytes(b"documents fictifs")

    api.build_index.main.side_effect = fake_build
    api.chatbot.load_resources.return_value = ("nouvel index", "nouveau client", "nouveau prompt")

    response = client.post("/rebuild", headers=admin_headers)

    assert response.status_code == 200
    assert response.json() == {"status": "Index reconstruit."}
    assert {path.name for path in target.iterdir()} == {
        "Vector_Store.faiss", "DocStore_LangChain.pkl",
    }
    api.build_index.main.assert_called_once()
    assert api.chatbot.load_resources.call_count == 2  # démarrage puis reconstruction
    api.chatbot.generate_answer.assert_not_called()
    assert client.post("/ask", json={"question": "Du jazz ?"}).status_code == 200
    api.chatbot.generate_answer.assert_called_once_with(
        "Du jazz ?", "nouvel index", "nouveau client", "nouveau prompt",
    )
    assert api.chatbot.load_resources.call_count == 2


def test_failed_rebuild_preserves_existing_index(client, admin_headers):
    target = api.chatbot.INDEX_DIR
    target.mkdir(parents=True)
    sentinel = target / "ancien.txt"
    sentinel.write_text("À conserver", encoding="utf-8")
    api.build_index.main.side_effect = ValueError("secret-fictif : embeddings incorrects")

    response = client.post("/rebuild", headers=admin_headers)

    assert response.status_code == 503
    assert "secret-fictif" not in response.text
    assert sentinel.read_text(encoding="utf-8") == "À conserver"
    assert client.post("/ask", json={"question": "Du jazz ?"}).status_code == 200
    api.chatbot.generate_answer.assert_called_once_with(
        "Du jazz ?", "index simulé", "client simulé", "prompt simulé",
    )


def test_reload_failure_restores_files_and_keeps_previous_resources(client, admin_headers):
    """Une construction réussie ne suffit pas : le nouvel index doit être rechargeable."""
    target = api.chatbot.INDEX_DIR
    target.mkdir(parents=True)
    sentinel = target / "ancien.txt"
    sentinel.write_text("À conserver", encoding="utf-8")

    def fake_build(destination):
        destination.mkdir()
        (destination / "nouveau.txt").write_text("Index fictif non rechargeable", encoding="utf-8")

    api.build_index.main.side_effect = fake_build
    api.chatbot.load_resources.side_effect = ValueError("secret-fictif : index incompatible")

    response = client.post("/rebuild", headers=admin_headers)

    assert response.status_code == 503
    assert "secret-fictif" not in response.text
    assert sentinel.read_text(encoding="utf-8") == "À conserver"
    assert not (target / "nouveau.txt").exists()
    assert client.post("/ask", json={"question": "Du jazz ?"}).status_code == 200
    api.chatbot.generate_answer.assert_called_once_with(
        "Du jazz ?", "index simulé", "client simulé", "prompt simulé",
    )


def test_publish_failure_restores_previous_directory(client, admin_headers, monkeypatch):
    """Simuler un refus de déplacement après la mise à l'abri de l'ancien index."""
    target = api.chatbot.INDEX_DIR
    target.mkdir(parents=True)
    sentinel = target / "ancien.txt"
    sentinel.write_text("À conserver", encoding="utf-8")
    api.build_index.main.side_effect = lambda destination: destination.mkdir()
    original_rename = Path.rename

    def refuse_publication(path, destination):
        if path.name == "ready" and Path(destination) == target:
            raise OSError("Publication refusée : détail interne")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", refuse_publication)

    response = client.post("/rebuild", headers=admin_headers)

    assert response.status_code == 503
    assert "détail interne" not in response.text
    assert sentinel.read_text(encoding="utf-8") == "À conserver"
    api.chatbot.load_resources.assert_called_once_with()


@pytest.mark.parametrize("problem, expected_status", [
    ("quota", 429), ("timeout", 504), ("connection", 502), ("other", 502),
])
def test_retry_error_preserves_safe_http_status(client, problem, expected_status):
    """Simuler l'enveloppe Tenacity réellement utilisée par les embeddings Mistral."""
    request = httpx.Request("POST", "https://example.invalid/embeddings")
    if problem == "quota":
        cause = httpx.HTTPStatusError("secret-fictif", request=request,
                                     response=httpx.Response(429, request=request))
    elif problem == "timeout":
        cause = httpx.ReadTimeout("secret-fictif", request=request)
    elif problem == "connection":
        cause = httpx.ConnectError("secret-fictif", request=request)
    else:
        cause = ValueError("secret-fictif")
    attempt = Future(1)
    attempt.set_exception(cause)
    api.chatbot.generate_answer.side_effect = RetryError(attempt)

    response = client.post("/ask", json={"question": "Du jazz ?"})

    assert response.status_code == expected_status
    assert "secret-fictif" not in response.text
    assert "example.invalid" not in response.text

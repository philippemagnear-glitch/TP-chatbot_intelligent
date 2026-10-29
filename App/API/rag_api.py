"""Interface HTTP du RAG : lancer depuis la racine avec un seul processus Uvicorn.

Commande : uv run uvicorn App.API.rag_api:app --host 127.0.0.1 --port 8000
Documentation interactive : http://127.0.0.1:8000/docs
FastAPI et Uvicorn doivent être installés au préalable.
Usage local uniquement : un worker, un utilisateur et des appels successifs.
Il n'y a volontairement aucun verrou ni gestion des accès concurrents.
"""

import logging
import os
import pickle
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import RetryError

from App.RAG import build_index, chatbot


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charger une fois ; exiger un RAG prêt si le démarrage Docker le demande."""
    try:
        app.state.resources = chatbot.load_resources()
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError,
            httpx.HTTPError, RetryError) as error:
        # Ne pas journaliser le message brut : une erreur distante peut contenir
        # des informations sensibles. Le type aide à identifier l'étape en défaut.
        logger.error("Chargement initial des ressources RAG impossible (%s).",
                     type(error).__name__)
        if os.getenv("RAG_REQUIRE_READY") == "1":
            raise RuntimeError("Démarrage refusé : ressources RAG indisponibles.") from None
        app.state.resources = None
    try:
        yield
    finally:
        app.state.resources = None


app = FastAPI(title="RAG OpenAgenda Toulouse", version="1.0.0", lifespan=lifespan)
security = HTTPBearer(auto_error=False)


# ---------- Contrats HTTP ----------
class QuestionRequest(BaseModel):
    """Une question textuelle ; aucun paramètre de modèle ni chemin fourni par le client."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(strict=True, min_length=1, max_length=4000,
                          description="Question sur les événements de Toulouse.")

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La question ne doit pas être vide.")
        return value


class AnswerResponse(BaseModel):
    answer: str


class StatusResponse(BaseModel):
    status: str


def require_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> None:
    """Vérifier un secret administrateur distinct de la clé Mistral, sans l'exposer."""
    load_dotenv(chatbot.PROJECT_ROOT / ".env", override=False)
    expected = os.getenv("RAG_ADMIN_TOKEN", "")
    if not expected.strip():
        raise HTTPException(503, "La reconstruction n'est pas configurée.")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"), expected.encode("utf-8"),
    ):
        raise HTTPException(401, "Authentification administrateur requise.",
                            headers={"WWW-Authenticate": "Bearer"})


def generation_error(error: Exception) -> HTTPException:
    """Traduire les erreurs HTTPX, y compris celles enveloppées par Tenacity.

    L'intégration LangChain installée utilise HTTPX. Pour les embeddings, une
    tentative échouée peut remonter dans RetryError, même avec max_retries=0.
    Le corps d'erreur distant n'est jamais renvoyé au client HTTP.
    """
    if isinstance(error, RetryError):
        error = error.last_attempt.exception()
    if isinstance(error, httpx.TimeoutException):
        return HTTPException(504, "Mistral n'a pas répondu dans le délai prévu.")
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 429:
            return HTTPException(429, "Quota Mistral atteint ; réessayez plus tard.")
        return HTTPException(502, "Le service Mistral a refusé la génération.")
    if isinstance(error, httpx.RequestError):
        return HTTPException(502, "Communication avec Mistral impossible.")
    return HTTPException(502, "La réponse RAG n'a pas pu être produite.")


# ---------- Routes ----------
@app.post("/ask", response_model=AnswerResponse, summary="Poser une question au RAG",
          responses={422: {"description": "Question absente, vide ou incorrecte."},
                     429: {"description": "Quota Mistral atteint."},
                     502: {"description": "Échec de génération."},
                     503: {"description": "Ressources RAG indisponibles."},
                     504: {"description": "Délai Mistral dépassé."}})
def ask_question(payload: QuestionRequest, request: Request) -> AnswerResponse:
    """Recevoir {"question": "…"} et renvoyer {"answer": "…"}, sans historique.

    Réutiliser l'index, le prompt et les clients chargés au démarrage. Seules la
    vectorisation de la question et la génération consomment de l'usage Mistral.
    """
    resources = request.app.state.resources
    if resources is None:
        raise HTTPException(503, "Les ressources RAG sont indisponibles.")
    store, llm, prompt = resources
    try:
        return AnswerResponse(answer=chatbot.generate_answer(payload.question, store, llm, prompt))
    except (httpx.HTTPError, RetryError, ValueError, RuntimeError) as error:
        raise generation_error(error) from None


@app.post("/rebuild", response_model=StatusResponse, dependencies=[Depends(require_admin)],
          summary="Reconstruire FAISS depuis les embeddings existants",
          responses={401: {"description": "Jeton administrateur absent ou incorrect."},
                     503: {"description": "Reconstruction indisponible."}})
def rebuild_index(request: Request) -> StatusResponse:
    """Aucun corps attendu. Fournir Authorization: Bearer <RAG_ADMIN_TOKEN>.

    Reconstruire uniquement FAISS, sans extraction ni appel Mistral. Préparer le
    nouvel index séparément avant remplacement, puis recharger les ressources.
    En cas d'échec, conserver les ressources précédentes et restaurer les fichiers.
    Ne pas appeler /ask ou /rebuild en parallèle pendant cette opération.
    """
    try:
        with build_index.rebuild_index(destination=chatbot.INDEX_DIR):
            resources = chatbot.load_resources()
        request.app.state.resources = resources
        return StatusResponse(status="Index reconstruit.")
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError,
            httpx.HTTPError, RetryError):
        raise HTTPException(503, "La reconstruction de l'index a échoué.") from None


@app.get("/health", response_model=StatusResponse, summary="Vérifier que l'API répond")
def health() -> StatusResponse:
    """Vérifier uniquement le serveur HTTP, sans charger FAISS ni appeler Mistral."""
    return StatusResponse(status="API disponible.")

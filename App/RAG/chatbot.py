"""Première version du RAG OpenAgenda en terminal, sans historique conversationnel.

Les événements et leurs vecteurs restent dans l'index local déjà construit.
Seules les questions sont vectorisées ; les fiches retrouvées sont envoyées
intégralement au modèle de génération. L'exécution utilise l'API payante Mistral.
"""

import json
import os
import pickle
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


# ---------- Configuration ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = Path(__file__).resolve().parent
INDEX_DIR = RAG_DIR / "vector_store"
PROMPT_PATH = RAG_DIR / "prompt.txt"
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "ministral-8b-2512"
SEARCH_K = 20
MAX_EVENTS = 5


# ---------- Initialisation ----------
def load_resources() -> tuple[FAISS, ChatMistralAI, str]:
    """Charger le prompt, les modèles Mistral et notre index local de confiance."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Le fichier prompt.txt est vide.")
    if INDEX_DIR.is_symlink() or not INDEX_DIR.is_dir():
        raise ValueError(f"Dossier d'index local introuvable ou incorrect : {INDEX_DIR}")
    for name in ("index.faiss", "index.pkl"):
        path = INDEX_DIR / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Fichier d'index absent ou incorrect : {name}")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("MISTRAL_API_KEY absente : renseigner le fichier .env.")
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key, max_retries=0)
    # Le pickle peut exécuter du code lors de sa lecture : cette autorisation
    # concerne exclusivement l'index créé par notre application, jamais un fichier reçu.
    store = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True,
    )
    if store.index.ntotal <= 0 or store.index.d != 1024:
        raise ValueError("Index vide ou incompatible avec mistral-embed.")
    llm = ChatMistralAI(model=CHAT_MODEL, api_key=api_key, temperature=0, max_retries=0)
    return store, llm, prompt


# ---------- Recherche et récupération des fiches ----------
def search_events(question: str, store: FAISS) -> list[Document]:
    """Retenir les cinq premiers événements distincts parmi les vingt chunks retrouvés."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question ne doit pas être vide.")
    # LangChain vectorise la question avec le modèle associé à l'index.
    # La similarité concerne les chunks ; les fiches complètes ne sont pas vectorisées.
    chunks = store.similarity_search(question.strip(), k=SEARCH_K)
    events = []
    seen = set()
    for chunk in chunks:
        uid = chunk.metadata.get("uid")
        if isinstance(uid, bool) or not isinstance(uid, (str, int)) or not str(uid).strip():
            raise ValueError("Un chunk retrouvé ne possède pas d'UID valide.")
        event_id = f"event_{uid}"
        if event_id in seen:
            continue
        reference = store.docstore.search(event_id)
        if (not isinstance(reference, Document)
                or str(reference.metadata.get("uid")) != str(uid)
                or not isinstance(reference.metadata.get("event"), dict)
                or str(reference.metadata["event"].get("uid")) != str(uid)):
            raise ValueError(f"Fiche complète absente ou incohérente : {event_id}. Reconstruire l'index.")
        seen.add(event_id)
        events.append(reference)
        if len(events) == MAX_EVENTS:
            break
    return events


# ---------- Contexte brut et génération ----------
def build_context(events: list[Document]) -> str:
    """Sérialiser toutes les métadonnées des fiches, sans filtrer ni tronquer."""
    return json.dumps([event.metadata for event in events], ensure_ascii=False, indent=2, allow_nan=False)


def generate_answer(question: str, store: FAISS, llm: ChatMistralAI, prompt: str) -> str:
    """Envoyer une question indépendante et les fiches complètes au LLM."""
    events = search_events(question, store)
    if not events:
        return "Aucun événement n'a été retrouvé pour cette question."
    context = build_context(events)
    # Le prompt est un texte libre, pas un template : les accolades sont acceptées.
    # Un nouveau couple de messages est créé à chaque question, sans historique.
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Question :\n{question.strip()}\n\nFiches d'événements (JSON) :\n{context}"),
    ]
    response = llm.invoke(messages)
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("Le modèle n'a pas renvoyé de réponse textuelle exploitable.")
    return response.content


# ---------- Boucle du terminal ----------
def main() -> None:
    """Poser plusieurs questions indépendantes ; quitter avec exit, quitter ou Ctrl+C."""
    try:
        store, llm, prompt = load_resources()
    except httpx.HTTPError:
        print("Initialisation impossible : erreur de communication avec Mistral.")
        return
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        print(f"Initialisation impossible : {error}")
        return
    print("Bienvenue ! Posez une question sur les événements de Toulouse. Pour sortir : quitter ou exit.")
    while True:
        question = input("\nVotre question : ").strip()
        if question.lower() in ("quitter", "exit"):
            break
        if not question:
            print("Saisissez une question non vide.")
            continue
        try:
            print("\n" + generate_answer(question, store, llm, prompt))
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status in (400, 413, 422):
                print("Mistral refuse la requête : vérifiez notamment la taille du contexte complet. Aucune donnée n'a été tronquée.")
            else:
                print(f"Erreur API Mistral (HTTP {status}). Vérifiez la clé, les quotas ou la disponibilité du service.")
        except httpx.RequestError:
            print("Connexion à Mistral impossible ou délai dépassé. Vous pouvez réessayer.")
        except ValueError as error:
            print(f"Réponse impossible : {error}")
    print("Au revoir !")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nAu revoir !")

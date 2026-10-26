"""Première version du RAG OpenAgenda en terminal, sans historique conversationnel.

Les événements et leurs vecteurs restent dans l'index local déjà construit.
Seules les questions sont vectorisées ; les fiches retrouvées sont transmises
en JSON brut ou en Markdown selon la configuration. L'exécution utilise l'API Mistral.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import faiss
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
PROMPT_PATH = RAG_DIR / "prompt.md"
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "ministral-8b-2512"
SEARCH_K = 20
MAX_EVENTS = 5
USE_MARKDOWN_CONTEXT = True


# ---------- Initialisation ----------
def load_resources() -> tuple[FAISS, ChatMistralAI, str]:
    """Charger le prompt Markdown, les modèles Mistral et notre index local de confiance."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Le fichier prompt.md est vide.")
    if INDEX_DIR.is_symlink() or not INDEX_DIR.is_dir():
        raise ValueError(f"Dossier d'index local introuvable ou incorrect : {INDEX_DIR}")
    for name in ("Vector_Store.faiss", "DocStore_LangChain.pkl"):
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
    # load_local() attend deux fichiers avec le même préfixe. Nos noms distincts
    # nécessitent de lire séparément les vecteurs et le couple DocStore / IDs.
    index = faiss.read_index(str(INDEX_DIR / "Vector_Store.faiss"))
    with (INDEX_DIR / "DocStore_LangChain.pkl").open("rb") as file:
        docstore, index_to_docstore_id = pickle.load(file)
    store = FAISS(embeddings, index, docstore, index_to_docstore_id)
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


# ---------- Présentation du contexte et génération ----------
def build_context(events: list[Document]) -> str:
    """Sérialiser toutes les métadonnées des fiches, sans filtrer ni tronquer."""
    return json.dumps([event.metadata for event in events], ensure_ascii=False, indent=2, allow_nan=False)


def build_markdown_context(events: list[Document]) -> str:
    """Présenter les fiches en Markdown, sans modifier les événements du DocStore.

    Les données métier restent disponibles, sans la syntaxe JSON. Seules les
    informations techniques des liens intégrés (iframe, script, miniature, etc.)
    sont écartées ; leurs adresses, titres et descriptions sont conservés.
    """
    labels = {
        "description": "Description", "long_description": "Description longue",
        "timings": "Horaires", "begin": "Début", "end": "Fin",
        "date_range": "Période", "location_name": "Lieu", "address": "Adresse",
        "public": "Public", "age_min": "Âge minimum", "age_max": "Âge maximum",
        "accessibility": "Accessibilité (valeurs déclarées, false = non confirmé)",
        "conditions": "Conditions", "access": "Accès", "event_links": "Liens",
    }

    def render(value, depth=0):
        """Déplier listes et dictionnaires sans perdre les valeurs zéro ou false."""
        indent = "  " * depth
        if isinstance(value, dict):
            return "\n".join(
                f"{indent}- {labels.get(key, key.replace('_', ' '))} :\n{render(item, depth + 1)}"
                for key, item in value.items()
            ) or f"{indent}Non renseigné"
        if isinstance(value, list):
            return "\n".join(f"{indent}-\n{render(item, depth + 1)}" for item in value) or f"{indent}Non renseigné"
        if value is None or value == "":
            return f"{indent}Non renseigné"
        text = str(value).lower() if isinstance(value, bool) else str(value)
        return "\n".join(indent + line for line in text.splitlines())

    sheets = []
    for reference in events:
        event = reference.metadata["event"]
        lines = [f"## {event.get('title', 'Sans titre')} — UID {event['uid']}"]
        for key, value in event.items():
            if key in ("uid", "title"):
                continue
            if key == "event_links" and isinstance(value, list):
                # Ne filtrer que les détails d'intégration, jamais les autres champs.
                links = []
                for link in value:
                    if not isinstance(link, dict):
                        links.append(link)
                        continue
                    useful = {name: link[name] for name in ("link", "title", "description", "url") if name in link}
                    if isinstance(link.get("data"), dict):
                        useful["data"] = {name: link["data"][name] for name in ("title", "description", "url") if name in link["data"]}
                    links.append(useful)
                value = links
            lines.append(f"### {labels.get(key, key.replace('_', ' '))}\n{render(value)}")
        sheets.append("\n\n".join(lines))
    return "\n\n---\n\n".join(sheets)


# ---------- Export temporaire pour observer les requêtes ----------
def export_mistral_request(messages: list[SystemMessage | HumanMessage]) -> None:
    """Exporter les messages de génération, sans clé API ni en-têtes réseau.

    Cet export montre les entrées fournies à LangChain, pas la requête HTTP
    complète produite par la bibliothèque. Les retours à la ligne sont échappés
    en JSON : ils restent de vrais sauts de ligne dans le contenu des messages.
    Pour retirer ce diagnostic, supprimer cette fonction et son appel avant invoke().
    """
    payload = {
        "model": CHAT_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system" if isinstance(message, SystemMessage) else "user",
             "content": message.content}
            for message in messages
        ],
    }
    try:
        directory = RAG_DIR / "debug_requests"
        directory.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = directory / f"mistral_request_{timestamp}.json"
        # Le mode x protège les exports précédents contre tout écrasement.
        with path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        print(f"Requête Mistral exportée : {path}")
    except OSError as error:
        print(f"Export de diagnostic impossible : {error}. La génération continue.")


def generate_answer(question: str, store: FAISS, llm: ChatMistralAI, prompt: str) -> str:
    """Envoyer une question indépendante et les fiches dans le format choisi."""
    events = search_events(question, store)
    if not events:
        return "Aucun événement n'a été retrouvé pour cette question."
    context = build_markdown_context(events) if USE_MARKDOWN_CONTEXT else build_context(events)
    context_format = "Markdown" if USE_MARKDOWN_CONTEXT else "JSON"
    # Le prompt est un texte libre, pas un template : les accolades sont acceptées.
    # Un nouveau couple de messages est créé à chaque question, sans historique.
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Question :\n{question.strip()}\n\nFiches d'événements ({context_format}) :\n{context}"),
    ]
    export_mistral_request(messages)
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

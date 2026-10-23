"""Construire FAISS à partir des textes, métadonnées et vecteurs déjà sauvegardés.

Ce script ne lit ni le JSON OpenAgenda ni .env et n'appelle aucune API.
Exécuter d'abord build_embeddings.py pour préparer un fichier complet.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ---------- Configuration ----------
RAG_DIR = Path(__file__).resolve().parent
CACHE_PATH = RAG_DIR / "embeddings.jsonl"
INDEX_DIR = RAG_DIR / "vector_store"
MODEL = "mistral-embed"
VECTOR_DIMENSION = 1024


# FAISS attend un objet Embeddings même avec des vecteurs déjà calculés.
# Cet objet satisfait cette interface sans service externe. Les méthodes lèvent
# une erreur si une vectorisation est demandée par accident. Pour la recherche,
# l'index sera rechargé avec le véritable modèle qui vectorisera les questions.
class ExistingEmbeddings(Embeddings):
    """Empêcher toute vectorisation pendant la construction hors ligne."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Utiliser les vecteurs existants, sans nouvelle vectorisation.")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Ce script construit l'index ; il ne vectorise pas de question.")


# ---------- Lecture du fichier autonome ----------
# Chaque lot fournit les trois éléments : page_content pour restituer le texte,
# metadata pour identifier ou filtrer l'événement, vector pour la similarité.
# Le nombre total attendu et les IDs uniques empêchent de publier un index
# incomplet ou contenant des doublons. Seul le format actuel sans signature est lu.
def load_embeddings() -> tuple[list[Document], list[list[float]]]:
    """Lire les lots complets et vérifier qu'aucun chunk ne manque ou se répète."""
    documents, vectors = [], []
    ids = set()
    total = None
    with CACHE_PATH.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Ligne {number} incomplète : relancer build_embeddings.py.") from error
            if (not isinstance(record, dict)
                    or set(record) != {"format_version", "model", "total_chunks", "chunks"}
                    or record.get("format_version") != 2
                    or record.get("model") != MODEL):
                raise ValueError("Format de cache ou modèle incorrect : aucun ancien format n'est accepté.")
            batch = record.get("chunks")
            batch_total = record.get("total_chunks")
            if (not isinstance(batch, list) or not 1 <= len(batch) <= 32
                    or type(batch_total) is not int or batch_total <= 0):
                raise ValueError(f"Lot incorrect à la ligne {number}.")
            if total is None:
                total = batch_total
            if batch_total != total or len(batch) != min(32, total - len(documents)):
                raise ValueError(f"Nombre de chunks incohérent à la ligne {number}.")
            for item in batch:
                if (not isinstance(item, dict) or not isinstance(item.get("page_content"), str)
                        or not item["page_content"] or not isinstance(item.get("metadata"), dict)):
                    raise ValueError(f"Texte ou métadonnées incorrects à la ligne {number}.")
                chunk_id = item["metadata"].get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id or chunk_id in ids:
                    raise ValueError(f"Identifiant absent ou dupliqué à la ligne {number}.")
                vector = item.get("vector")
                try:
                    valid = isinstance(vector, list) and len(vector) == VECTOR_DIMENSION and all(
                        type(value) in (int, float) and math.isfinite(value) for value in vector
                    )
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError(f"Vecteur incorrect à la ligne {number}.")
                ids.add(chunk_id)
                documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
                vectors.append(vector)
    if not documents or len(documents) != total:
        raise ValueError(f"Embeddings incomplets : {len(documents)}/{total}. Relancer build_embeddings.py.")
    return documents, vectors


# ---------- Construction et sauvegarde ----------
# from_embeddings() utilise les vecteurs reçus sans appeler embed_documents().
# LangChain associe les vecteurs FAISS aux documents et aux métadonnées pour
# restituer les événements lors des futures recherches. Tout l'index reste en RAM
# jusqu'à la sauvegarde finale, effectuée une seule fois.
def main() -> None:
    """Construire un index complet sans remplacer une destination existante."""
    if INDEX_DIR.exists() or INDEX_DIR.is_symlink():
        raise FileExistsError(f"Index déjà présent : {INDEX_DIR}")
    documents, vectors = load_embeddings()
    ids = [document.metadata["chunk_id"] for document in documents]
    store = FAISS.from_embeddings(
        text_embeddings=list(zip([document.page_content for document in documents], vectors)),
        embedding=ExistingEmbeddings(),
        metadatas=[document.metadata for document in documents], ids=ids,
    )
    if store.index.ntotal != len(documents):
        raise ValueError("Le nombre de vecteurs indexés ne correspond pas aux chunks.")
    # index.faiss contient l'index vectoriel ; index.pkl conserve les documents
    # et la correspondance des IDs. Les deux fichiers sont écrits dans un dossier
    # temporaire sur le même système de fichiers, puis déplacés ensemble.
    # Une erreur normale nettoie le temporaire et laisse l'ancien index intact.
    # Le futur chargement du pickle doit se limiter à un index local de confiance.
    with tempfile.TemporaryDirectory(prefix=".faiss_", dir=RAG_DIR) as temporary:
        ready = Path(temporary) / "vector_store"
        store.save_local(str(ready))
        if not all((ready / name).is_file() for name in ("index.faiss", "index.pkl")):
            raise OSError("Sauvegarde FAISS incomplète.")
        if INDEX_DIR.exists() or INDEX_DIR.is_symlink():
            raise FileExistsError(f"Destination apparue pendant la construction : {INDEX_DIR}")
        ready.rename(INDEX_DIR)
    print(f"Index terminé : {len(documents)} chunks | {store.index.ntotal} vecteurs | {INDEX_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Construction interrompue : {error}", file=sys.stderr)
        sys.exit(1)

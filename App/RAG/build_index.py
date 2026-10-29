"""Construire FAISS à partir des textes, métadonnées et vecteurs déjà sauvegardés.

Le dernier JSON prétraité fournit les fiches complètes ajoutées au DocStore.
Relancer le pipeline complet après une extraction pour utiliser le même dataset.
Ce script ne lit pas .env et n'appelle aucune API.
"""

import json
import math
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ---------- Configuration ----------
RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR.parent / "Data"
CACHE_PATH = DATA_DIR / "embeddings" / "embeddings.jsonl"
INDEX_DIR = DATA_DIR / "vector_store"
PROCESSED_DIR = RAG_DIR.parent / "Data" / "processed"
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


# ---------- Documents de référence ----------
def load_event_documents() -> dict[str, Document]:
    """Charger une fiche complète par UID depuis le dernier JSON prétraité."""
    files = list(PROCESSED_DIR.glob("openagenda_processed_*.json"))
    if not files:
        raise FileNotFoundError(f"Aucun JSON prétraité dans {PROCESSED_DIR}")
    source = max(files, key=lambda path: (path.stat().st_mtime, path.name))
    with source.open(encoding="utf-8") as file:
        data = json.load(file)
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list) or not events:
        raise ValueError("La clé events doit contenir une liste non vide.")

    references = {}
    for event in events:
        uid = event.get("uid") if isinstance(event, dict) else None
        if isinstance(uid, bool) or not isinstance(uid, (str, int)) or not str(uid).strip():
            raise ValueError("Document de référence sans UID valide.")
        event_id = f"event_{uid}"
        if event_id in references:
            raise ValueError(f"Événement dupliqué : {uid}")
        # Garder la fiche comme dictionnaire préserve listes, nombres et valeurs
        # nulles, notamment dans timings. Ce texte d'identification n'est pas vectorisé.
        references[event_id] = Document(
            page_content=f"Événement {uid}", metadata={"uid": uid, "event": event},
        )
    print(f"Fiches complètes chargées : {len(references)} | Source : {source.name}")
    return references


# ---------- Construction et sauvegarde ----------
# from_embeddings() utilise les vecteurs reçus sans appeler embed_documents().
# LangChain associe les vecteurs FAISS aux documents et aux métadonnées pour
# restituer les événements lors des futures recherches. Tout l'index reste en RAM
# jusqu'à la sauvegarde finale, effectuée une seule fois.
def main(destination: Path | None = None) -> None:
    """Construire un index complet sans remplacer une destination existante."""
    # L'API peut préparer un index séparé ; le lancement terminal reste inchangé.
    index_dir = INDEX_DIR if destination is None else Path(destination)
    if index_dir.exists() or index_dir.is_symlink():
        raise FileExistsError(f"Index déjà présent : {index_dir}")
    documents, vectors = load_embeddings()
    ids = [document.metadata["chunk_id"] for document in documents]
    references = load_event_documents()
    if set(ids).intersection(references):
        raise ValueError("Collision entre identifiants de chunks et d'événements.")
    for document in documents:
        uid = document.metadata.get("uid")
        if (isinstance(uid, bool) or not isinstance(uid, (str, int))
                or f"event_{uid}" not in references):
            raise ValueError(f"Événement absent pour le chunk {document.metadata['chunk_id']}")
    store = FAISS.from_embeddings(
        text_embeddings=list(zip([document.page_content for document in documents], vectors)),
        embedding=ExistingEmbeddings(),
        metadatas=[document.metadata for document in documents], ids=ids,
    )
    if store.index.ntotal != len(documents):
        raise ValueError("Le nombre de vecteurs indexés ne correspond pas aux chunks.")
    # Ajouter directement au DocStore ne crée aucun vecteur : ne pas utiliser
    # add_documents(), qui déclencherait une vectorisation des fiches complètes.
    # Un chunk retrouvé donne son uid, puis docstore.search(f"event_{uid}") donne
    # la fiche complète, même si plusieurs chunks appartiennent au même événement.
    store.docstore.add(references)
    # Vector_Store.faiss contient les vecteurs ; DocStore_LangChain.pkl conserve les documents
    # et la correspondance des IDs. Les deux fichiers sont écrits dans un dossier
    # temporaire sur le même système de fichiers, puis déplacés ensemble.
    # Une erreur normale nettoie le temporaire et laisse l'ancien index intact.
    # Le futur chargement du pickle doit se limiter à un index local de confiance.
    # Garder aussi les fichiers temporaires dans Data, sur le même disque que l'index.
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".faiss_", dir=index_dir.parent) as temporary:
        ready = Path(temporary) / "vector_store"
        store.save_local(str(ready))
        # LangChain utilise un même préfixe pour ses deux fichiers. Les renommer
        # avant publication permet de distinguer clairement leurs rôles.
        (ready / "index.faiss").rename(ready / "Vector_Store.faiss")
        (ready / "index.pkl").rename(ready / "DocStore_LangChain.pkl")
        if not all((ready / name).is_file() for name in ("Vector_Store.faiss", "DocStore_LangChain.pkl")):
            raise OSError("Sauvegarde FAISS incomplète.")
        if index_dir.exists() or index_dir.is_symlink():
            raise FileExistsError(f"Destination apparue pendant la construction : {index_dir}")
        ready.rename(index_dir)
    print(f"Index terminé : {len(documents)} chunks | {len(references)} événements "
          f"| {store.index.ntotal} vecteurs | {index_dir}")


@contextmanager
def rebuild_index(destination: Path | None = None):
    """Préparer puis remplacer l'index ; restaurer l'ancien si le rechargement échoue.

    Utiliser ``with rebuild_index():`` et recharger les ressources dans ce bloc.
    La sortie normale confirme le nouvel index. Une exception dans le bloc
    restaure l'ancien dossier. Réservé aux appels successifs du TP local.
    """
    target = INDEX_DIR if destination is None else Path(destination)
    backup = target.with_name(".vector_store_previous")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("Destination d'index incorrecte.")
    if backup.exists() or backup.is_symlink():
        raise ValueError("Une sauvegarde précédente doit être vérifiée manuellement.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rebuild_", dir=target.parent) as temporary:
        ready = Path(temporary) / "ready"
        main(destination=ready)
        if target.exists():
            target.rename(backup)
        try:
            ready.rename(target)
            yield
        except BaseException:
            # Protéger aussi l'ancien index lors d'une interruption. La sauvegarde
            # reste hors du temporaire si sa restauration devait elle-même échouer.
            if target.exists():
                target.rename(Path(temporary) / "rejected")
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                # L'index est déjà validé : un échec de nettoyage ne l'invalide pas.
                warnings.warn("Index reconstruit ; sauvegarde précédente conservée.", stacklevel=2)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Construction interrompue : {error}", file=sys.stderr)
        sys.exit(1)

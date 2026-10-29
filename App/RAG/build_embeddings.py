"""Créer les embeddings OpenAgenda et sauvegarder texte, métadonnées et vecteur.

Le fichier embeddings.jsonl contient un lot complet par ligne. Une relance
compare les lots enregistrés aux chunks actuels avant de les réutiliser.
Seul le format actuel, sans signature, est accepté. Ce script appelle
l'API payante Mistral uniquement pour les chunks encore manquants.
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- Configuration ----------
# Les chemins partent de ce fichier pour rester indépendants du dossier depuis
# lequel le script est lancé. Le cache et l'index ont des rôles différents :
# le premier permet la reprise, le second servira à la recherche sémantique.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "App" / "Data" / "processed"
RAG_DIR = PROJECT_ROOT / "App" / "RAG"
CACHE_PATH = PROJECT_ROOT / "App" / "Data" / "embeddings" / "embeddings.jsonl"

MODEL = "mistral-embed"
BATCH_SIZE = 32
VECTOR_DIMENSION = 1024
MAX_CHUNK_SIZE = 2048
CHUNK_OVERLAP = 150

# Les trois premiers thèmes regroupent le sujet, les informations de visite
# et les conditions d'accueil. La description détaillée est traitée séparément.
# Ces quatre thèmes aident la recherche à retrouver un passage ciblé.
SECTIONS = {
    "Présentation": ["title", "description", "keywords", "event_types"],
    "Informations pratiques": [
        "uid", "date_range", "location_name", "address",
        "postal_code", "city", "district", "metropole", "department",
        "region", "latitude", "longitude", "access", "event_links",
        "online_access_link", "location_website", "slug", "status",
        "origin_agenda",
    ],
    "Public et accessibilité": [
        "public", "age_min", "age_max", "accessibility", "conditions", "billetterie",
    ],
}


# ---------- Chargement des événements ----------
# Le JSON est déjà prétraité : on ne refait pas son nettoyage ici. Comme dans
# le notebook, le dernier fichier est choisi selon sa date de modification.
# Les contrôles évitent de payer des embeddings pour des données inexploitées
# ou de créer des chunks avec des identifiants dupliqués.
def load_events() -> tuple[Path, list[dict]]:
    """Charger le dernier JSON et vérifier les champs nécessaires au découpage."""
    files = list(PROCESSED_DIR.glob("openagenda_processed_*.json"))
    if not files:
        raise FileNotFoundError(f"Aucun JSON prétraité dans {PROCESSED_DIR}")
    source = max(files, key=lambda path: (path.stat().st_mtime, path.name))
    with source.open(encoding="utf-8") as file:
        data = json.load(file)
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list) or not events:
        raise ValueError("La clé 'events' doit contenir une liste non vide.")
    uids = set()
    for event in events:
        if (not isinstance(event, dict)
                or isinstance(event.get("uid"), bool)
                or not isinstance(event.get("uid"), (str, int))):
            raise ValueError("Événement sans UID valide.")
        uid = str(event["uid"])
        if not uid.strip() or uid in uids:
            raise ValueError(f"UID vide ou dupliqué : {uid}")
        uids.add(uid)
        if not isinstance(event.get("title"), str) or not event["title"].strip():
            raise ValueError(f"Titre manquant pour {uid}")
        # Certains événements n'ont pas de description longue : null ou un champ
        # absent est accepté. Une liste ou un nombre reste une erreur de format.
        long_description = event.get("long_description")
        if long_description is not None and not isinstance(long_description, str):
            raise ValueError(f"Description longue incorrecte pour {uid}")
    return source, events


# ---------- Chunking structuré validé dans le notebook ----------
# Un Document LangChain associe page_content, le texte envoyé à Mistral, et
# metadata, les identifiants nécessaires pour retrouver le chunk et son événement.
# Les informations complètes restent dans le JSON prétraité et pourront être
# retrouvées grâce à uid. Seul page_content est envoyé au modèle d'embedding.
def build_chunks(event: dict) -> list[Document]:
    """Découper les quatre sections sans dépasser la taille maximale, en-tête compris."""
    sections = {name: list(fields) for name, fields in SECTIONS.items()}
    assigned = {field for fields in sections.values() for field in fields}
    assigned.update({"long_description", "timings"})
    sections["Informations pratiques"].extend(
        field for field in event if field not in assigned
    )
    section_texts = {
        name: "\n".join(
            f"{field} : {json.dumps(event[field], ensure_ascii=False)}"
            for field in fields if field in event
        )
        for name, fields in sections.items()
    }
    # Une description absente ne crée pas de section vide supplémentaire.
    if event.get("long_description"):
        section_texts["Description longue"] = event["long_description"]

    chunks = []
    for name, content in section_texts.items():
        header = f"Événement : {event['title']}\nSection : {name}\n"
        if len(header) + len(content) <= MAX_CHUNK_SIZE:
            parts = [content]
        else:
            # Réserver la place du titre, du thème et d'une numérotation large.
            # Les 2 048 caractères concernent le Document final, pas le seul texte.
            numbered_header = (
                f"Événement : {event['title']}\n"
                f"Section : {name} - partie 99999/99999\n"
            )
            available = MAX_CHUNK_SIZE - len(numbered_header)
            if available <= CHUNK_OVERLAP:
                raise ValueError(f"En-tête trop long pour le chunking : {event['uid']} / {name}")
            # Les séparateurs naturels sont essayés avant les caractères.
            # Garder les séparateurs et espaces évite de supprimer du contenu.
            # Avec ces séparateurs, le chevauchement peut être inférieur à 150.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=available,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
                keep_separator=True,
                strip_whitespace=False,
            )
            parts = splitter.split_text(content)
        if len(parts) > 99999:
            raise ValueError(f"Section trop longue : {event['uid']} / {name}")

        section_chunks = []
        for part_number, part in enumerate(parts, start=1):
            index = len(chunks) + 1
            metadata = {
                "uid": event["uid"], "chunk_id": f"{event['uid']}_{index}",
                "chunk_index": index, "section": name,
            }
            part_header = header
            if len(parts) > 1:
                part_header = (
                    f"Événement : {event['title']}\n"
                    f"Section : {name} - partie {part_number}/{len(parts)}\n"
                )
                metadata.update(part=part_number, total_parts=len(parts))
            page_content = part_header + part
            # Contrôle final indépendant du découpeur : aucun dépassement accepté.
            if len(page_content) > MAX_CHUNK_SIZE:
                raise ValueError(f"Chunk trop long : {metadata['chunk_id']}")
            document = Document(page_content=page_content, metadata=metadata)
            chunks.append(document)
            section_chunks.append(document)

        # Les liens restent dans la même section, sans relier des thèmes différents.
        if len(section_chunks) > 1:
            for index, chunk in enumerate(section_chunks):
                chunk.metadata["previous_chunk_id"] = (
                    section_chunks[index - 1].metadata["chunk_id"] if index else None
                )
                chunk.metadata["next_chunk_id"] = (
                    section_chunks[index + 1].metadata["chunk_id"]
                    if index + 1 < len(section_chunks) else None
                )
    return chunks

# ---------- Validation et reprise ----------
def validate_vectors(vectors: list, expected_count: int) -> None:
    """Vérifier un vecteur de 1 024 nombres finis pour chaque chunk."""
    try:
        valid = isinstance(vectors, list) and len(vectors) == expected_count and all(
            isinstance(vector, list) and len(vector) == VECTOR_DIMENSION
            and all(type(value) in (int, float) and math.isfinite(value) for value in vector)
            for vector in vectors
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Vecteurs incorrects : attendu 1 024 nombres finis par chunk.")


def make_record(chunks: list[Document], vectors: list, total: int) -> dict:
    """Associer chaque vecteur à son texte et à toutes ses métadonnées."""
    return {
        "format_version": 2, "model": MODEL, "total_chunks": total,
        "chunks": [
            {"page_content": chunk.page_content, "metadata": chunk.metadata, "vector": vector}
            for chunk, vector in zip(chunks, vectors)
        ],
    }


def is_incomplete_json(text: str, error: json.JSONDecodeError) -> bool:
    """Distinguer une fin de JSON coupée d'une syntaxe incorrecte au milieu."""
    suffix = text[error.pos:].rstrip()
    return (
        not suffix or error.msg.startswith("Unterminated string")
        or (error.msg == "Expecting value" and suffix in ("t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul", "-"))
        or (error.msg == "Expecting ',' delimiter" and suffix.lower() in ("e", "e+", "e-", "."))
        or (error.msg.startswith("Invalid \\uXXXX escape")
            and suffix.startswith("u") and len(suffix) < 5
            and all(character in "0123456789abcdefABCDEF" for character in suffix[1:]))
    )


# La reprise compare les lots à la suite exacte des chunks actuels : nombre,
# ordre, texte et métadonnées. Elle contrôle aussi chaque vecteur avant tout ajout.
# Les changements dans les chunks déjà sauvegardés bloquent la reprise ; ceux
# des chunks encore absents n'ont pas besoin d'être détectés par une empreinte.
# Une seule exécution simultanée est prévue, car le cache n'est pas verrouillé.
def prepare_cache(chunks: list[Document]) -> int:
    """Valider le cache, réparer sa dernière ligne si elle est coupée et compter les chunks."""
    if CACHE_PATH.is_symlink():
        raise ValueError("Le cache ne doit pas être un lien symbolique.")
    if not CACHE_PATH.exists():
        return 0
    count = 0
    truncate_at = None
    add_newline = False
    with CACHE_PATH.open("r+b") as file:
        number = 0
        while True:
            position = file.tell()
            line = file.readline()
            if not line:
                break
            number += 1
            has_newline = line.endswith(b"\n")
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError as error:
                if has_newline or error.reason != "unexpected end of data":
                    raise ValueError(f"Encodage incorrect à la ligne {number}.") from error
                truncate_at = position
                break
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                if has_newline or not is_incomplete_json(text, error):
                    raise ValueError(f"JSON incorrect à la ligne {number} : cache inchangé.") from error
                truncate_at = position
                break
            if (not isinstance(record, dict)
                    or set(record) != {"format_version", "model", "total_chunks", "chunks"}
                    or record["format_version"] != 2 or record["model"] != MODEL
                    or type(record["total_chunks"]) is not int
                    or record["total_chunks"] != len(chunks)):
                raise ValueError(f"Format ou nombre total incompatible à la ligne {number}.")
            saved = record["chunks"]
            expected = chunks[count:count + BATCH_SIZE]
            if (not expected or not isinstance(saved, list) or len(saved) != len(expected)
                    or not all(isinstance(item, dict) for item in saved)):
                raise ValueError(f"Nombre de chunks incorrect à la ligne {number}.")
            vectors = [item.get("vector") for item in saved]
            validate_vectors(vectors, len(expected))
            if saved != make_record(expected, vectors, len(chunks))["chunks"]:
                raise ValueError(f"Ordre, textes ou métadonnées incompatibles à la ligne {number}.")
            count += len(expected)
            add_newline = not has_newline
        # Aucune écriture avant la validation de tous les lots complets.
        # Une ligne coupée est supprimée seule ; un lot valide sans saut de ligne
        # est conservé et simplement terminé pour permettre le prochain ajout.
        if truncate_at is not None:
            file.truncate(truncate_at)
        elif add_newline:
            file.write(b"\n")
        if truncate_at is not None or add_newline:
            file.flush()
            os.fsync(file.fileno())
    return count


# ---------- Génération par lots ----------
def generate_embeddings(chunks: list[Document]) -> None:
    """Reprendre au premier chunk non sauvegardé, sans recalculer les lots valides."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = prepare_cache(chunks)
    print(f"Chunks récupérés : {count}/{len(chunks)}")
    if count == len(chunks):
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("MISTRAL_API_KEY absente : renseigner .env.")
    model = MistralAIEmbeddings(model=MODEL, api_key=api_key)
    for start in range(count, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        # Seul le lot manquant est envoyé à Mistral. Chaque ligne sauvegarde
        # ensemble ses textes, métadonnées et vecteurs, pour faciliter la reprise.
        vectors = model.embed_documents([chunk.page_content for chunk in batch])
        validate_vectors(vectors, len(batch))
        record = make_record(batch, vectors, len(chunks))
        with CACHE_PATH.open("ab") as file:
            file.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            # flush() vide le tampon Python ; fsync() demande l'écriture sur disque.
            file.flush()
            os.fsync(file.fileno())
        print(f"Chunks sauvegardés : {start + len(batch)}/{len(chunks)}")


# ---------- Exécution ----------
def main() -> None:
    """Charger les événements, créer les chunks et préparer le fichier d'embeddings."""
    source, events = load_events()
    chunks = [chunk for event in events for chunk in build_chunks(event)]
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Identifiants de chunks dupliqués.")
    print(f"Source : {source.name} | Événements : {len(events)} | Chunks : {len(chunks)}")
    oversized = sum(len(chunk.page_content) > MAX_CHUNK_SIZE for chunk in chunks)
    if oversized:
        print(f"Attention : {oversized} chunks dépassent la référence de {MAX_CHUNK_SIZE} caractères.")
    generate_embeddings(chunks)
    print(f"Embeddings terminés : {len(chunks)} chunks dans {CACHE_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Vectorisation interrompue : {error}", file=sys.stderr)
        sys.exit(1)

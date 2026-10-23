"""Tests de lecture et d'intégration FAISS sur trois événements et vecteurs fictifs."""

import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.RAG import build_index as indexing


@pytest.fixture
def sample_data():
    """Créer trois événements aux vecteurs distincts, sans prétendre simuler leur sens."""
    documents = [
        Document(page_content="Concert de jazz à Toulouse", metadata={"uid": 12345, "chunk_id": "12345_1", "chunk_index": 1, "section": "Présentation"}),
        Document(page_content="Exposition de peinture", metadata={"uid": 67890, "chunk_id": "67890_1", "chunk_index": 1, "section": "Présentation"}),
        Document(page_content="Randonnée en montagne", metadata={"uid": 13579, "chunk_id": "13579_1", "chunk_index": 1, "section": "Présentation"}),
    ]
    vectors = [[1.0, 0.0] + [0.0] * 1022, [0.0, 1.0] + [0.0] * 1022, [-1.0, 0.0] + [0.0] * 1022]
    return documents, vectors


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    """Rediriger le cache, les fichiers temporaires et l'index vers tmp_path."""
    monkeypatch.setattr(indexing, "RAG_DIR", tmp_path)
    monkeypatch.setattr(indexing, "CACHE_PATH", tmp_path / "embeddings.jsonl")
    monkeypatch.setattr(indexing, "INDEX_DIR", tmp_path / "vector_store")
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(indexing, "PROCESSED_DIR", processed)


@pytest.fixture
def sample_events(sample_data):
    """Écrire les fiches prétraitées fictives, avec horaires et types structurés."""
    documents, _ = sample_data
    events = [
        {
            "uid": document.metadata["uid"], "title": document.page_content,
            "timings": [{"begin": "2026-11-10T20:00:00+01:00", "end": "2026-11-10T22:00:00+01:00"}],
            "public": ["Tout public"], "age_min": 6, "latitude": 43.6,
            "online_access_link": None, "extra": {"available": True},
        }
        for document in documents
    ]
    source = indexing.PROCESSED_DIR / "openagenda_processed_test.json"
    source.write_text(json.dumps({"events": events}), encoding="utf-8")
    return events


def make_record(documents, vectors):
    """Décrire explicitement le format JSONL attendu par le script d'indexation."""
    return {
        "format_version": 2, "model": "mistral-embed", "total_chunks": len(documents),
        "chunks": [
            {"page_content": document.page_content, "metadata": dict(document.metadata), "vector": vector}
            for document, vector in zip(documents, vectors)
        ],
    }


def make_store(documents, vectors):
    """Construire un petit index réel ; ExistingEmbeddings interdit de vectoriser."""
    return indexing.FAISS.from_embeddings(
        text_embeddings=list(zip([document.page_content for document in documents], vectors)),
        embedding=indexing.ExistingEmbeddings(),
        metadatas=[document.metadata for document in documents],
        ids=[document.metadata["chunk_id"] for document in documents],
    )


def test_load_embeddings_preserves_documents_and_vector_associations(sample_data):
    """Vérifier que chaque texte et ses métadonnées restent liés au bon vecteur."""
    documents, vectors = sample_data
    indexing.CACHE_PATH.write_text(json.dumps(make_record(documents, vectors)) + "\n", encoding="utf-8")

    loaded_documents, loaded_vectors = indexing.load_embeddings()

    assert len(loaded_documents) == len(loaded_vectors) == 3
    for loaded, expected in zip(loaded_documents, documents):
        assert isinstance(loaded, Document)
        assert loaded.page_content == expected.page_content
        assert loaded.metadata == expected.metadata
    assert loaded_vectors == vectors


@pytest.mark.parametrize("problem", ["incomplete", "duplicate-id", "wrong-dimension"])
def test_load_embeddings_rejects_invalid_data(sample_data, problem):
    """Refuser un fichier incomplet, un doublon ou un vecteur incompatible."""
    documents, vectors = sample_data
    record = make_record(documents, vectors)
    if problem == "incomplete":
        record["chunks"].pop()
    elif problem == "duplicate-id":
        record["chunks"][1]["metadata"]["chunk_id"] = documents[0].metadata["chunk_id"]
    else:
        record["chunks"][0]["vector"] = [0.0] * 1023
    indexing.CACHE_PATH.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        indexing.load_embeddings()


def test_load_embeddings_rejects_truncated_json():
    """Un fichier coupé ne doit pas permettre de construire un index partiel."""
    indexing.CACHE_PATH.write_text('{"format_version": 2, "chunks": [', encoding="utf-8")

    with pytest.raises(ValueError, match="incomplète"):
        indexing.load_embeddings()


def test_faiss_preserves_vector_count_ids_and_docstore(sample_data):
    """Contrôler les liens entre positions FAISS, identifiants et Documents."""
    documents, vectors = sample_data
    store = make_store(documents, vectors)

    assert store.index.ntotal == len(documents)
    assert store.index_to_docstore_id == {
        position: document.metadata["chunk_id"] for position, document in enumerate(documents)
    }
    for position, document in enumerate(documents):
        retrieved = store.docstore.search(document.metadata["chunk_id"])
        assert retrieved.page_content == document.page_content
        assert retrieved.metadata == document.metadata
        assert store.index.reconstruct(position).tolist() == vectors[position]


def test_similarity_search_by_vector_returns_jazz_document(sample_data):
    """Tester les distances et le DocStore, pas la qualité sémantique de Mistral."""
    documents, vectors = sample_data
    store = make_store(documents, vectors)
    query = [0.99, 0.01] + [0.0] * 1022

    results = store.similarity_search_by_vector(query, k=3)

    assert len(results) == 3
    assert results[0].metadata["uid"] == 12345
    assert results[0].page_content == documents[0].page_content
    assert results[0].metadata == documents[0].metadata


def test_main_preserves_event_references_after_save_and_reload(sample_data, sample_events):
    """Sauvegarder une fiche par événement et retrouver ses données depuis un chunk FAISS."""
    documents, vectors = sample_data
    # Deux chunks du jazz partagent une seule fiche complète, sans second exemplaire.
    documents = [*documents, Document(
        page_content="Programme du concert de jazz",
        metadata={"uid": 12345, "chunk_id": "12345_2", "chunk_index": 2, "section": "Description longue"},
    )]
    vectors = [*vectors, [0.9, 0.1] + [0.0] * 1022]
    indexing.CACHE_PATH.write_text(json.dumps(make_record(documents, vectors)) + "\n", encoding="utf-8")

    indexing.main()

    assert {path.name for path in indexing.INDEX_DIR.iterdir()} == {
        "index.faiss", "index.pkl",
    }
    # Le JSON n'est plus nécessaire une fois le DocStore sauvegardé.
    (indexing.PROCESSED_DIR / "openagenda_processed_test.json").unlink()
    # Autorisation pickle limitée aux fichiers créés par ce test dans tmp_path.
    reloaded = indexing.FAISS.load_local(
        str(indexing.INDEX_DIR), indexing.ExistingEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    results = reloaded.similarity_search_by_vector([0.99, 0.01] + [0.0] * 1022, k=2)

    assert reloaded.index.ntotal == len(documents)
    assert results[0].page_content == documents[0].page_content
    assert results[0].metadata == documents[0].metadata
    assert all(document.metadata["uid"] == 12345 for document in results)
    assert len(reloaded.index_to_docstore_id) == len(documents)
    assert not any(key.startswith("event_") for key in reloaded.index_to_docstore_id.values())
    # Contrôler la totalité du DocStore pour détecter une duplication des fiches.
    assert set(reloaded.docstore._dict) == {
        *(document.metadata["chunk_id"] for document in documents),
        *(f"event_{event['uid']}" for event in sample_events),
    }
    for event in sample_events:
        reference = reloaded.docstore.search(f"event_{event['uid']}")
        assert reference.metadata == {"uid": event["uid"], "event": event}
        assert reference.metadata["event"]["timings"] == event["timings"]
        assert reference.metadata["event"]["extra"]["available"] is True
        assert reference.metadata["event"]["online_access_link"] is None
    uid = results[0].metadata["uid"]
    assert reloaded.docstore.search(f"event_{uid}").metadata["event"] == sample_events[0]
    assert "event" not in results[0].metadata


@pytest.mark.parametrize("problem", ["missing-event", "duplicate-event", "missing-uid", "collision"])
def test_main_rejects_inconsistent_event_references(sample_data, sample_events, problem):
    """Refuser un UID introuvable, une fiche dupliquée ou une collision avant sauvegarde."""
    documents, vectors = sample_data
    record = make_record(documents, vectors)
    if problem == "missing-event":
        sample_events.pop()
    elif problem == "duplicate-event":
        sample_events.append(sample_events[0])
    elif problem == "missing-uid":
        record["chunks"][0]["metadata"].pop("uid")
    else:
        record["chunks"][0]["metadata"]["chunk_id"] = "event_12345"
    source = indexing.PROCESSED_DIR / "openagenda_processed_test.json"
    source.write_text(json.dumps({"events": sample_events}), encoding="utf-8")
    indexing.CACHE_PATH.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        indexing.main()
    assert not indexing.INDEX_DIR.exists()


@pytest.mark.parametrize("events", [[], None, [{"uid": True}], [{"uid": ""}]])
def test_load_event_documents_rejects_incomplete_references(events):
    """Contrôler uniquement la structure et l'identification nécessaires au stockage."""
    source = indexing.PROCESSED_DIR / "openagenda_processed_test.json"
    source.write_text(json.dumps({"events": events}), encoding="utf-8")

    with pytest.raises(ValueError):
        indexing.load_event_documents()


def test_main_refuses_to_overwrite_existing_directory():
    """Protéger une sauvegarde présente avant même de charger les embeddings."""
    indexing.INDEX_DIR.mkdir()
    sentinel = indexing.INDEX_DIR / "keep.txt"
    sentinel.write_text("conserver", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Index déjà présent"):
        indexing.main()
    assert sentinel.read_text(encoding="utf-8") == "conserver"

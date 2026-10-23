"""Tests du chunking et du cache, avec données fictives et sans appel Mistral."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.RAG import build_embeddings as embeddings


def make_event(uid=12345):
    """Fournir un événement court couvrant les quatre sections."""
    return {
        "uid": uid, "title": "Festival de jazz",
        "description": "Concert à Toulouse.",
        "date_range": "10 novembre", "location_name": "Salle du Canal",
        "public": ["Tout public"], "age_min": 6,
        "long_description": "Concerts, rencontres et ateliers.",
        "timings": [{"begin": "2026-11-10T20:00:00+01:00"}],
    }


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Isoler le cache et bloquer le modèle réel, même en cas d'appel accidentel."""
    monkeypatch.setattr(embeddings, "CACHE_PATH", tmp_path / "embeddings.jsonl")
    monkeypatch.setattr(embeddings, "load_dotenv", Mock())
    monkeypatch.setattr(
        embeddings, "MistralAIEmbeddings",
        Mock(side_effect=AssertionError("Un test ne doit pas appeler Mistral.")),
    )


def test_build_chunks_creates_sections_and_minimal_metadata():
    """Vérifier les thèmes, textes et identifiants sans dupliquer l'événement."""
    event = make_event()
    chunks = embeddings.build_chunks(event)

    assert [chunk.metadata["section"] for chunk in chunks] == [
        "Présentation", "Informations pratiques", "Public et accessibilité",
        "Description longue",
    ]
    for chunk, expected in zip(chunks, [event["description"], "Salle du Canal", "Tout public", event["long_description"]]):
        assert expected in chunk.page_content
        assert "Événement : Festival de jazz" in chunk.page_content
        assert set(chunk.metadata) == {"uid", "chunk_id", "chunk_index", "section"}
        assert chunk.metadata["uid"] == event["uid"]
        assert "timings :" not in chunk.page_content
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert ids == [f"12345_{index}" for index in range(1, 5)]
    assert len(set(ids)) == len(chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [1, 2, 3, 4]


def test_build_chunks_splits_long_sections_and_preserves_text():
    """Contrôler taille finale et liens, pour chaque thème, sans perdre le texte."""
    event = make_event()
    # Sans espaces, le repli par caractères permet de vérifier exactement
    # la reconstruction du texte et le chevauchement de 150 caractères.
    event.update(description="P" * 5000, access="I" * 5000,
                 conditions="U" * 5000, long_description="D" * 5000)
    chunks = embeddings.build_chunks(event)
    contents = {
        "Présentation": '\n'.join([
            'title : "Festival de jazz"', f'description : "{event["description"]}"',
        ]),
        "Informations pratiques": '\n'.join([
            'uid : 12345', 'date_range : "10 novembre"',
            'location_name : "Salle du Canal"', f'access : "{event["access"]}"',
        ]),
        "Public et accessibilité": '\n'.join([
            'public : ["Tout public"]', 'age_min : 6',
            f'conditions : "{event["conditions"]}"',
        ]),
        "Description longue": event["long_description"],
    }

    assert all(len(chunk.page_content) <= embeddings.MAX_CHUNK_SIZE for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"12345_{index}" for index in range(1, len(chunks) + 1)
    ]
    for section, content in contents.items():
        fragments = [chunk for chunk in chunks if chunk.metadata["section"] == section]
        assert len(fragments) > 1
        bodies = [chunk.page_content.split("\n", 2)[2] for chunk in fragments]
        # Reconstituer le contenu en retirant uniquement les chevauchements.
        rebuilt = bodies[0]
        for body in bodies[1:]:
            overlap = max(
                size for size in range(embeddings.CHUNK_OVERLAP + 1)
                if rebuilt.endswith(body[:size])
            )
            rebuilt += body[overlap:]
        assert rebuilt == content
        for index, chunk in enumerate(fragments):
            assert chunk.metadata["part"] == index + 1
            assert chunk.metadata["total_parts"] == len(fragments)
            assert chunk.metadata["previous_chunk_id"] == (
                fragments[index - 1].metadata["chunk_id"] if index else None
            )
            assert chunk.metadata["next_chunk_id"] == (
                fragments[index + 1].metadata["chunk_id"] if index + 1 < len(fragments) else None
            )


def test_validate_vectors_accepts_valid_batch():
    """Accepter un vecteur fini de la bonne dimension pour chaque texte."""
    embeddings.validate_vectors([[0.0] * 1024, [1.0] * 1024], expected_count=2)


@pytest.mark.parametrize("vector", [[0.0] * 1023, [float("nan")] + [0.0] * 1023],
                         ids=["wrong-dimension", "non-finite"])
def test_validate_vectors_rejects_invalid_vector(vector):
    """Refuser une dimension incorrecte ou NaN avant toute sauvegarde."""
    with pytest.raises(ValueError, match="Vecteurs incorrects"):
        embeddings.validate_vectors([vector], expected_count=1)


def test_make_record_and_prepare_cache_preserve_all_three_elements():
    """Vérifier le format JSONL et l'association texte, métadonnées et vecteur."""
    chunks = embeddings.build_chunks(make_event())
    vectors = [[float(index)] + [0.0] * 1023 for index in range(len(chunks))]
    record = embeddings.make_record(chunks, vectors, len(chunks))
    embeddings.CACHE_PATH.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    original = embeddings.CACHE_PATH.read_bytes()

    assert set(record) == {"format_version", "model", "total_chunks", "chunks"}
    assert record["format_version"] == 2
    assert record["model"] == "mistral-embed"
    assert record["total_chunks"] == len(chunks)
    for item, chunk, vector in zip(record["chunks"], chunks, vectors):
        assert item == {"page_content": chunk.page_content, "metadata": chunk.metadata, "vector": vector}
    assert embeddings.prepare_cache(chunks) == len(chunks)
    assert embeddings.CACHE_PATH.read_bytes() == original


@pytest.mark.parametrize("change", ["text", "metadata", "order"], ids=["changed-text", "changed-metadata", "changed-order"])
def test_prepare_cache_rejects_incompatible_data_without_changing_file(change):
    """Bloquer un cache incompatible sans détruire les lots déjà enregistrés."""
    chunks = embeddings.build_chunks(make_event())
    record = embeddings.make_record(chunks, [[0.0] * 1024 for _ in chunks], len(chunks))
    if change == "text":
        record["chunks"][0]["page_content"] = "Un autre texte"
    elif change == "metadata":
        record["chunks"][0]["metadata"] = {**record["chunks"][0]["metadata"], "uid": 999}
    else:
        record["chunks"][0], record["chunks"][1] = record["chunks"][1], record["chunks"][0]
    embeddings.CACHE_PATH.write_text(json.dumps(record) + "\n", encoding="utf-8")
    original = embeddings.CACHE_PATH.read_bytes()

    with pytest.raises(ValueError, match="incompatibles"):
        embeddings.prepare_cache(chunks)
    assert embeddings.CACHE_PATH.read_bytes() == original


def test_prepare_cache_repairs_only_incomplete_last_line():
    """Retirer un lot coupé tout en conservant les 32 chunks du premier lot."""
    chunks = [chunk for uid in range(9) for chunk in embeddings.build_chunks(make_event(uid))]
    first_batch = embeddings.make_record(chunks[:32], [[0.0] * 1024 for _ in range(32)], len(chunks))
    complete = (json.dumps(first_batch) + "\n").encode("utf-8")
    embeddings.CACHE_PATH.write_bytes(complete + b'{"format_version": 2, "chunks": [')

    assert embeddings.prepare_cache(chunks) == 32
    assert embeddings.CACHE_PATH.read_bytes() == complete


def test_prepare_cache_keeps_complete_last_line_without_newline():
    """Conserver un lot valide et ajouter seulement son séparateur final."""
    chunks = embeddings.build_chunks(make_event())
    record = embeddings.make_record(chunks, [[0.0] * 1024 for _ in chunks], len(chunks))
    original = json.dumps(record).encode("utf-8")
    embeddings.CACHE_PATH.write_bytes(original)

    assert embeddings.prepare_cache(chunks) == len(chunks)
    assert embeddings.CACHE_PATH.read_bytes() == original + b"\n"


def test_generate_embeddings_resumes_without_recomputing_saved_batch(monkeypatch):
    """Simuler une interruption et reprendre les quatre textes manquants avec un faux modèle."""
    chunks = [chunk for uid in range(9) for chunk in embeddings.build_chunks(make_event(uid))]
    model = Mock()
    model.embed_documents.side_effect = [
        [[0.0] * 1024 for _ in range(32)], OSError("Interruption simulée"),
    ]
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-fictive")
    constructor = Mock(return_value=model)
    monkeypatch.setattr(embeddings, "MistralAIEmbeddings", constructor)
    with pytest.raises(OSError, match="Interruption simulée"):
        embeddings.generate_embeddings(chunks)
    saved = embeddings.CACHE_PATH.read_bytes()
    assert embeddings.prepare_cache(chunks) == 32

    model.embed_documents.reset_mock(side_effect=True)
    model.embed_documents.return_value = [[1.0] * 1024 for _ in chunks[32:]]
    embeddings.generate_embeddings(chunks)

    model.embed_documents.assert_called_once_with([chunk.page_content for chunk in chunks[32:]])
    assert embeddings.CACHE_PATH.read_bytes().startswith(saved)
    assert embeddings.prepare_cache(chunks) == len(chunks)
    constructor.reset_mock()
    embeddings.generate_embeddings(chunks)
    constructor.assert_not_called()

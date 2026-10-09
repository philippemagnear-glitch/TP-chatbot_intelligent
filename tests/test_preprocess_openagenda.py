"""Tests unitaires du preprocessing OpenAgenda.

Fonctions testées :
    - load_raw : sélectionner le dernier export, lire et valider son JSON.
    - get_field_options : extraire les libellés français du schéma de référence.
    - decode_options : traduire les catégories et conserver les IDs inconnus.
    - french_value : extraire le français sans inventer de traduction.
    - decode_accessibility et decode_event_status : décoder les codes métier.
    - transform_event : sélectionner les champs sans modifier le RAW.
    - deduplicate_events : conserver le premier UID et compter les exclusions.
    - save_processed : écrire en UTF-8 sans écraser un export existant.
    - main : vérifier le résultat complet et les erreurs de lecture ou d'écriture.

Les données sont fictives. monkeypatch redirige les chemins vers tmp_path :
aucun véritable fichier RAW ou processed n'est lu ou modifié. Mock simule
uniquement les erreurs de main ; les lectures et écritures de succès sont réelles.
parametrize vérifie plusieurs entrées avec la même règle ; pytest.warns et
pytest.raises contrôlent les avertissements et les exceptions attendus.
"""

import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.Preprocessing import preprocess_openagenda as preprocessing


def make_reference():
    """Créer un petit schéma indépendant pour chaque appel."""
    return {"schema": {"fields": [
        {"field": "public", "options": [{"id": "2", "label": {"fr": "Tout public"}}]},
        {"field": "types-devenements", "options": [
            {"id": 1, "label": {"fr": "Concert", "en": "Concert"}},
        ]},
    ]}}


@pytest.fixture
def isolated_directories(monkeypatch, tmp_path):
    """Isoler tous les accès aux fichiers dans deux dossiers temporaires."""
    raw = tmp_path / "raw"
    raw.mkdir()
    processed = tmp_path / "processed"
    monkeypatch.setattr(preprocessing, "RAW_DIRECTORY", raw)
    monkeypatch.setattr(preprocessing, "PROCESSED_DIRECTORY", processed)
    return raw, processed


# ---------- Tests du chargement RAW ----------
@pytest.mark.parametrize("same_time", [False, True])
def test_load_raw_selects_latest_file(isolated_directories, same_time):
    """Choisir par date de modification, puis par nom en cas d'égalité."""
    raw, _ = isolated_directories
    payload = {"events": [], "reference_agenda": make_reference()}
    first = raw / "openagenda_a.json"
    latest = raw / "openagenda_z.json"
    for path in (first, latest):
        path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(first, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest, ns=(1_000_000_000, 1_000_000_000) if same_time else (2_000_000_000, 2_000_000_000))
    (raw / "other.json").write_text("invalid", encoding="utf-8")
    (raw / "openagenda_directory.json").mkdir()
    assert preprocessing.load_raw() == (latest, payload)


def test_load_raw_rejects_missing_export(isolated_directories):
    """Signaler l'absence d'export plutôt que produire un résultat vide."""
    with pytest.raises(FileNotFoundError, match="Aucun fichier"):
        preprocessing.load_raw()


def test_load_raw_rejects_invalid_json(isolated_directories):
    """Refuser un fichier dont le contenu n'est pas un JSON lisible."""
    raw, _ = isolated_directories
    (raw / "openagenda_invalid.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        preprocessing.load_raw()


@pytest.mark.parametrize("payload, message", [
    ([], "objet JSON"), ({}, "events"),
    ({"events": {}}, "events"),
    ({"events": [None]}, "Chaque événement"),
    ({"events": []}, "reference_agenda"),
    ({"events": [], "reference_agenda": []}, "reference_agenda"),
])
def test_load_raw_validates_structure(isolated_directories, payload, message):
    """Rejeter les conteneurs absents ou invalides avant la transformation."""
    raw, _ = isolated_directories
    (raw / "openagenda_invalid.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        preprocessing.load_raw()


# ---------- Tests du schéma et des catégories ----------
@pytest.mark.parametrize("field, expected", [
    ("types-devenements", {"1": "Concert"}), ("public", {"2": "Tout public"}),
])
def test_get_field_options_reads_french_labels(field, expected):
    """Lire chaque champ par son nom, indépendamment de sa position."""
    reference = make_reference()
    reference["schema"]["fields"].insert(0, None)
    reference["schema"]["fields"][1]["options"].extend([
        None, {"id": True, "label": {"fr": "Invalide"}},
        {"id": 3, "label": {"en": "English"}},
        {"id": 4, "label": "Texte"}, {"id": 5, "label": {"fr": " "}},
    ])
    assert preprocessing.get_field_options(reference, field) == expected


@pytest.mark.parametrize("reference, message", [
    ({}, "Schéma absent"), ({"schema": []}, "Schéma absent"),
    ({"schema": {"fields": {}}}, "Schéma absent"),
    ({"schema": {"fields": []}}, "Options du champ"),
    ({"schema": {"fields": [{"field": "public", "options": None}]}}, "Options du champ"),
])
def test_get_field_options_warns_without_usable_schema(reference, message):
    """Avertir et retourner une table vide si le schéma n'est pas exploitable."""
    with pytest.warns(UserWarning, match=message):
        assert preprocessing.get_field_options(reference, "public") == {}


@pytest.mark.parametrize("value, expected", [
    (None, None), ({}, None), ("", None), ([], []),
    (1, ["Concert"]), ("1", ["Concert"]), ([1, "1"], ["Concert", "Concert"]),
])
def test_decode_options_handles_known_and_empty_values(value, expected):
    """Traduire IDs numériques ou textuels et distinguer absence et liste vide."""
    assert preprocessing.decode_options(value, {"1": "Concert"}, "types-devenements") == expected


@pytest.mark.parametrize("unknown", [99, "inconnu", True, {"id": 1}])
def test_decode_options_preserves_unknown_values(unknown):
    """Conserver les IDs inconnus ou invalides et émettre un avertissement."""
    with pytest.warns(UserWarning, match="identifiant non résolu"):
        assert preprocessing.decode_options([1, unknown], {"1": "Concert"}, "public") == ["Concert", unknown]


# ---------- Tests des textes, de l'accessibilité et du statut ----------
@pytest.mark.parametrize("value, expected", [
    ({"fr": "Titre", "en": "Title"}, "Titre"), ("Texte", "Texte"),
    ({"en": "Title"}, None), (None, None), ("", None), ({}, None),
    ({"fr": ""}, None), ({"fr": ["musique"]}, ["musique"]),
])
def test_french_value_extracts_available_french(value, expected):
    """Extraire textes ou mots-clés français ; garder None si le français manque."""
    assert preprocessing.french_value(value) == expected


def test_decode_accessibility_translates_only_true_codes():
    """Traduire les cinq handicaps et garder un code inconnu explicitement actif."""
    assert preprocessing.decode_accessibility({
        "hi": True, "vi": True, "pi": True, "mi": True, "ii": True, "unknown": True,
    }) == ["Handicap auditif", "Handicap visuel", "Handicap psychique",
           "Handicap moteur", "Handicap intellectuel", "unknown"]
    assert preprocessing.decode_accessibility({"hi": False, "vi": 1, "mi": "true", "ii": None}) == []


@pytest.mark.parametrize("value", [None, {}, [], "hi", True])
def test_decode_accessibility_handles_missing_or_invalid_input(value):
    """Retourner une liste vide pour une accessibilité absente ou non structurée."""
    assert preprocessing.decode_accessibility(value) == []


@pytest.mark.parametrize("value, expected", [
    (1, "Programmé"), (2, "Reprogrammé"), (3, "Déplacé en ligne"),
    (4, "Reporté"), (5, "Complet"), (6, "Annulé"),
    (None, None), (99, 99), ("6", "6"), (True, True),
])
def test_decode_event_status_translates_only_known_integers(value, expected):
    """Traduire les six statuts entiers ; préserver absence, inconnu et autres types."""
    result = preprocessing.decode_event_status(value)
    assert result == expected
    assert type(result) is type(expected)


# ---------- Tests de la transformation ----------
def test_transform_event_produces_expected_fields_without_mutating_raw():
    """Vérifier champs RAG, séances et liens, exclusion de state et intégrité du RAW."""
    event = {
        "uid": 42, "title": {"fr": "Concert", "en": "Show"},
        "description": {"fr": "Résumé"}, "longDescription": {"fr": "Détails"},
        "keywords": {"fr": ["musique"]}, "dateRange": {"fr": "9 et 10 octobre"},
        "timings": [{"begin": "2026-10-09T18:00:00+02:00", "end": "2026-10-09T20:00:00+02:00"},
                    {"begin": "2026-10-10T18:00:00+02:00", "end": "2026-10-10T20:00:00+02:00"}],
        "location": {"name": "Salle", "address": "1 rue du Test", "postalCode": "31000",
                     "city": "Toulouse", "district": "Centre", "adminLevel3": "Métropole",
                     "adminLevel2": "Haute-Garonne", "adminLevel1": "Occitanie",
                     "latitude": 43.6, "longitude": 1.44, "access": {"fr": "Métro"},
                     "website": "https://example.org/salle"},
        "types-devenements": [1], "public": "2", "age": {"min": 6, "max": 99},
        "accessibility": {"mi": True, "hi": False}, "conditions": {"fr": "Gratuit"},
        "billetterie": ["https://example.org/billets"], "links": [{"link": "https://example.org"}],
        "onlineAccessLink": "https://example.org/live", "slug": "concert",
        "status": 5, "state": 2, "originAgenda": {"title": "Agenda test"}, "unused": "ignoré",
    }
    original = deepcopy(event)
    result = preprocessing.transform_event(event, {"1": "Concert"}, {"2": "Tout public"})
    assert result == {
        "uid": 42, "title": "Concert", "description": "Résumé", "long_description": "Détails",
        "keywords": ["musique"], "timings": original["timings"], "date_range": "9 et 10 octobre",
        "location_name": "Salle", "address": "1 rue du Test", "postal_code": "31000",
        "city": "Toulouse", "district": "Centre", "metropole": "Métropole",
        "department": "Haute-Garonne", "region": "Occitanie", "latitude": 43.6, "longitude": 1.44,
        "access": "Métro", "event_types": ["Concert"], "public": ["Tout public"],
        "age_min": 6, "age_max": 99, "accessibility": ["Handicap moteur"], "conditions": "Gratuit",
        "billetterie": original["billetterie"], "event_links": original["links"],
        "online_access_link": "https://example.org/live", "location_website": "https://example.org/salle",
        "slug": "concert", "status": "Complet", "origin_agenda": "Agenda test",
    }
    assert event == original
    assert result is not event


@pytest.mark.parametrize("container", [None, [], "invalide"])
def test_transform_event_handles_absent_and_invalid_optional_objects(container):
    """Tolérer location, age et originAgenda invalides sans inventer de valeurs."""
    result = preprocessing.transform_event({
        "uid": 1, "location": container, "age": container, "originAgenda": container,
    }, {}, {})
    assert result["uid"] == 1
    assert result["accessibility"] == []
    assert all(value is None for key, value in result.items() if key not in ("uid", "accessibility"))
    assert "state" not in result
    assert preprocessing.transform_event({}, {}, {})["uid"] is None


# ---------- Tests de la déduplication ----------
def test_deduplicate_events_keeps_first_occurrence_and_distinct_uids():
    """Rapprocher UID entier et textuel, conserver le premier sans fusion ni mutation."""
    events = [{"uid": 1, "title": "Premier"}, {"uid": "01", "title": "Second"}, {"uid": "2"}]
    original = deepcopy(events)
    unique, duplicates, invalid = preprocessing.deduplicate_events(events)
    assert unique == [events[0], events[2]]
    assert unique[0] is events[0]
    assert (duplicates, invalid) == (1, 0)
    assert events == original
    assert preprocessing.deduplicate_events([]) == ([], 0, 0)


@pytest.mark.parametrize("event", [{}, *[{"uid": uid} for uid in [None, 0, -1, "0", "abc", "", True, 1.5, [], {}]]])
def test_deduplicate_events_counts_invalid_uids(event):
    """Ignorer UID absent, non positif, non numérique ou de type invalide."""
    assert preprocessing.deduplicate_events([event, {"uid": 7}]) == ([{"uid": 7}], 0, 1)


# ---------- Tests de la sauvegarde ----------
def test_save_processed_writes_utf8_and_publishes_complete_json(isolated_directories):
    """Créer le dossier et un JSON UTF-8 horodaté, sans fichier temporaire restant."""
    _, processed = isolated_directories
    date = datetime(2026, 10, 9, 12, 30, 45, 123456, tzinfo=timezone.utc)
    payload = {"events": [{"title": "Événement à Toulouse"}]}
    path = preprocessing.save_processed(payload, date)
    assert path == processed / "openagenda_processed_20261009T123045123456Z.json"
    content = path.read_text(encoding="utf-8")
    assert json.loads(content) == payload
    assert "Événement à Toulouse" in content
    assert content.endswith("\n")
    assert not path.with_suffix(".json.tmp").exists()


def test_save_processed_refuses_to_overwrite_existing_export(isolated_directories):
    """Refuser deux sauvegardes au même horodatage et préserver le premier contenu."""
    date = datetime(2026, 10, 9, tzinfo=timezone.utc)
    path = preprocessing.save_processed({"events": []}, date)
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="existe déjà"):
        preprocessing.save_processed({"events": [{"uid": 1}]}, date)
    assert path.read_bytes() == original


# ---------- Tests de la fonction principale ----------
def test_main_processes_temporary_raw_and_saves_metadata(isolated_directories, capsys):
    """Vérifier le pipeline complet, ses compteurs, sa provenance et le RAW inchangé."""
    raw, processed = isolated_directories
    path = raw / "openagenda_test.json"
    payload = {"reference_agenda": make_reference(), "events": [
        {"uid": 1, "title": {"fr": "Concert"}, "types-devenements": [1], "public": [2], "state": 2, "status": 6},
        {"uid": "1", "title": {"fr": "Doublon"}}, {"uid": 2}, {"uid": None},
    ]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    original = path.read_bytes()
    assert preprocessing.main() == 0
    outputs = list(processed.glob("*.json"))
    assert len(outputs) == 1
    result = json.loads(outputs[0].read_text(encoding="utf-8"))
    date = datetime.fromisoformat(result["processed_at"])
    assert date.tzinfo == timezone.utc
    assert outputs[0].name == f"openagenda_processed_{date:%Y%m%dT%H%M%S%fZ}.json"
    assert {key: value for key, value in result.items() if key not in ("events", "processed_at")} == {
        "source_file": path.name, "raw_event_count": 4, "duplicate_count": 1,
        "ignored_uid_count": 1, "event_count": 2,
    }
    assert [event["uid"] for event in result["events"]] == [1, 2]
    first = result["events"][0]
    assert (first["title"], first["event_types"], first["public"], first["status"]) == (
        "Concert", ["Concert"], ["Tout public"], "Annulé",
    )
    assert all("state" not in event for event in result["events"])
    assert path.read_bytes() == original
    captured = capsys.readouterr()
    assert "Doublons supprimés : 1" in captured.out
    assert "Événements sauvegardés : 2" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("error", [FileNotFoundError("RAW absent"), ValueError("RAW invalide")])
def test_main_reports_loading_errors_without_saving(monkeypatch, isolated_directories, capsys, error):
    """Retourner 1 et signaler une erreur de fichier ou de données sans sauvegarde."""
    monkeypatch.setattr(preprocessing, "load_raw", Mock(side_effect=error))
    save = Mock()
    monkeypatch.setattr(preprocessing, "save_processed", save)
    assert preprocessing.main() == 1
    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert captured.out == ""
    save.assert_not_called()


def test_main_reports_saving_error(monkeypatch, isolated_directories, capsys):
    """Signaler une impossibilité d'écrire au lieu d'annoncer un succès."""
    raw, processed = isolated_directories
    monkeypatch.setattr(preprocessing, "load_raw", Mock(return_value=(
        raw / "openagenda_test.json", {"events": [], "reference_agenda": make_reference()},
    )))
    save = Mock(side_effect=OSError("Écriture impossible"))
    monkeypatch.setattr(preprocessing, "save_processed", save)
    assert preprocessing.main() == 1
    save.assert_called_once()
    captured = capsys.readouterr()
    assert "Écriture impossible" in captured.err
    assert captured.out == ""
    assert not processed.exists()

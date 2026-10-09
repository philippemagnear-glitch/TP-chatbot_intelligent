"""Tests unitaires du collecteur OpenAgenda, sans réseau ni vraie clé API.

fetch_pages : pagination, données et réponses invalides.
fetch_agendas : validation des UID et titres.
fetch_reference_agenda : définition brute de référence et erreurs.
get_start_date : borne annuelle, y compris le 29 février.
fetch_events : filtres envoyés à l'API.
save_json : écriture UTF-8 et publication du fichier temporaire.
main : clé API, orchestration et erreurs.

Les appels HTTP sont simulés avec unittest.mock. Les seuls fichiers écrits
par les tests sont placés dans tmp_path, jamais dans App/Data/raw/.

Chaque test prépare une situation, appelle la fonction étudiée puis vérifie
le résultat avec assert ou l'erreur attendue avec pytest.raises. Les simulations
isolent le comportement du collecteur des variations du réseau et de l'API.

Mock remplace un objet ou une fonction et mémorise ses appels. return_value
définit un résultat simulé ; side_effect fournit des résultats successifs,
exécute une fonction de simulation ou déclenche une exception.
MagicMock gère aussi les méthodes spéciales, notamment celles du bloc with.

Les fixtures sont fournies par Pytest via les paramètres des tests : monkeypatch
remplace temporairement des fonctions ou des variables d'environnement, tmp_path
fournit un dossier temporaire et capsys capture les sorties du terminal.
Les remplacements de monkeypatch sont annulés après chaque test.
parametrize rejoue un test pour plusieurs entrées afin de couvrir les cas limites.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import requests

# Rendre l'import indépendant de la façon dont pytest est lancé depuis le projet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.API import openagenda


# ---------- Simulation d'une réponse HTTP ----------
# Une réponse contient seulement les attributs que le collecteur utilise.
def make_response(payload, status=200):
    """Simuler le statut HTTP et le résultat de response.json(), sans serveur."""
    response = Mock(status_code=status)
    response.json.return_value = payload
    return response


# ---------- Tests de fetch_pages() ----------
# Les réponses successives remplacent les pages de l'API. La dernière page
# peut omettre after ou contenir null : dans les deux cas, la lecture se termine.
# Le premier test contrôle aussi les timeouts et l'interdiction des redirections.
@pytest.mark.parametrize("end_cursor", [{}, {"after": None}])
def test_fetch_pages_returns_single_page(end_cursor):
    session = Mock()
    session.get.return_value = make_response({"events": [{"uid": 1}], **end_cursor})
    assert openagenda.fetch_pages(session, "/events", {"size": 100}, "events") == [{"uid": 1}]
    assert session.get.call_count == 1
    assert session.get.call_args.kwargs["timeout"] == openagenda.TIMEOUT
    assert session.get.call_args.kwargs["allow_redirects"] is False


# La simulation enregistre une copie des paramètres à chaque appel : le même
# dictionnaire interne reçoit ensuite after[]. Sans copie, l'historique serait trompeur.
# Le curseur mixte vérifie les trois types réellement acceptés par le collecteur.
def test_fetch_pages_follows_cursor_without_mutating_input():
    session = Mock()
    cursor = ["next", 61.54935, 39750428]
    params = {"size": 100}
    sent_params = []
    responses = iter([
        make_response({"events": [{"uid": 1}], "after": cursor}),
        make_response({"events": [{"uid": 2}]}),
    ])

    def fake_get(url, **kwargs):
        sent_params.append(kwargs["params"].copy())
        return next(responses)

    session.get.side_effect = fake_get
    assert openagenda.fetch_pages(session, "/events", params, "events") == [{"uid": 1}, {"uid": 2}]
    assert sent_params == [{"size": 100}, {"size": 100, "after[]": cursor}]
    assert params == {"size": 100}


# Une erreur HTTP est une réponse reçue avec un statut d'échec. Une erreur réseau,
# testée ensuite, empêche au contraire de recevoir une réponse exploitable.
# pytest.raises échoue si l'exception attendue n'est pas levée ; match vérifie le message.
@pytest.mark.parametrize("status", [401, 500])
def test_fetch_pages_rejects_http_error(status):
    session = Mock()
    session.get.return_value = make_response({}, status)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        openagenda.fetch_pages(session, "/events", {}, "events")


# Les détails techniques de l'exception simulée ne doivent pas être exposés
# dans le message utilisateur, car ils pourraient contenir des informations sensibles.
def test_fetch_pages_handles_network_error():
    session = Mock()
    session.get.side_effect = requests.RequestException("sensitive detail")
    with pytest.raises(RuntimeError, match="Erreur réseau") as error:
        openagenda.fetch_pages(session, "/events", {}, "events")
    assert "sensitive detail" not in str(error.value)


# Un statut 200 ne garantit pas un JSON valide : json() doit pouvoir échouer proprement.
def test_fetch_pages_rejects_invalid_json():
    session = Mock()
    session.get.return_value.json.side_effect = ValueError("Invalid JSON")
    session.get.return_value.status_code = 200
    with pytest.raises(RuntimeError, match="JSON invalide"):
        openagenda.fetch_pages(session, "/events", {}, "events")


# Tester séparément la structure de la réponse et celle du curseur empêche
# d'accepter silencieusement des données incomplètes ou une pagination incohérente.
@pytest.mark.parametrize("payload", [[], {}, {"events": None}, {"events": {}}, {"events": [1]}])
def test_fetch_pages_rejects_invalid_collection(payload):
    session = Mock()
    session.get.return_value = make_response(payload)
    with pytest.raises(RuntimeError):
        openagenda.fetch_pages(session, "/events", {}, "events")


@pytest.mark.parametrize("cursor", [[], "next", {}, [True], [None], [[1]]])
def test_fetch_pages_rejects_invalid_cursor(cursor):
    session = Mock()
    session.get.return_value = make_response({"events": [{}], "after": cursor})
    with pytest.raises(RuntimeError, match="curseur after est invalide"):
        openagenda.fetch_pages(session, "/events", {}, "events")


# Un curseur répété ou une page vide annonçant une suite doit interrompre la boucle.
# Le nombre d'appels confirme que la répétition est détectée dès la deuxième page.
def test_fetch_pages_rejects_repeated_cursor():
    session = Mock()
    session.get.return_value = make_response({"events": [{}], "after": [1.5, 2]})
    with pytest.raises(RuntimeError, match="ne progresse plus"):
        openagenda.fetch_pages(session, "/events", {}, "events")
    assert session.get.call_count == 2


def test_fetch_pages_rejects_empty_page_with_cursor():
    session = Mock()
    session.get.return_value = make_response({"events": [], "after": [1]})
    with pytest.raises(RuntimeError, match="ne progresse plus"):
        openagenda.fetch_pages(session, "/events", {}, "events")


# ---------- Tests de fetch_agendas() ----------
# Remplacer fetch_pages isole les validations des agendas de la pagination.
# assert_called_once_with vérifie les arguments exacts. Le test ne dépend donc
# ni du nombre d'agendas réels ni de leur contenu actuel sur OpenAgenda.
def test_fetch_agendas_returns_valid_agendas(monkeypatch):
    agendas = [{"uid": 1, "title": "Agenda A"}, {"uid": "2", "title": "Agenda B"}]
    fetch = Mock(return_value=agendas)
    monkeypatch.setattr(openagenda, "fetch_pages", fetch)
    session = Mock()
    assert openagenda.fetch_agendas(session) is agendas
    fetch.assert_called_once_with(session, "/agendas", {
        "locationSet": openagenda.TOULOUSE_LOCATION_SET_UID, "size": 100,
    }, "agendas")


# Les UID invalides sont rejetés avant de construire une URL d'événements.
# Un titre valide permet ensuite de conserver la provenance dans le JSON RAW.
@pytest.mark.parametrize("agenda", [{"title": "A"}, *[
    {"uid": uid, "title": "A"} for uid in [None, 0, -1, "abc", True, 1.5]
]])
def test_fetch_agendas_rejects_invalid_uid(monkeypatch, agenda):
    monkeypatch.setattr(openagenda, "fetch_pages", Mock(return_value=[agenda]))
    with pytest.raises(RuntimeError, match="UID absent ou invalide"):
        openagenda.fetch_agendas(Mock())


@pytest.mark.parametrize("agenda", [{"uid": 1}, {"uid": 1, "title": None}, {"uid": 1, "title": 42}])
def test_fetch_agendas_rejects_invalid_title(monkeypatch, agenda):
    monkeypatch.setattr(openagenda, "fetch_pages", Mock(return_value=[agenda]))
    with pytest.raises(RuntimeError, match="titre absent ou invalide"):
        openagenda.fetch_agendas(Mock())


# ---------- Tests de fetch_reference_agenda() ----------
# Le schéma doit être retourné intact ; aucun libellé n'est interprété ici.
# L'opérateur is vérifie que la réponse est rendue telle quelle, sans reconstruction.
def test_fetch_reference_agenda_returns_raw_dictionary():
    payload = {"uid": 42448083, "schema": {"fields": []}}
    session = Mock()
    session.get.return_value = make_response(payload)
    assert openagenda.fetch_reference_agenda(session) is payload
    session.get.assert_called_once_with(
        f"{openagenda.API_URL}/agendas/{openagenda.TOULOUSE_AGENDA_UID}",
        params={"detailed": 1}, timeout=openagenda.TIMEOUT, allow_redirects=False,
    )


# Chaque variante provoque une erreur différente sur le même endpoint.
# La simulation reste locale au test et ne modifie pas les autres réponses simulées.
@pytest.mark.parametrize("case, message", [
    ("http", "HTTP 401"), ("json", "JSON invalide"),
    ("network", "Erreur réseau"), ("list", "objet JSON"),
])
def test_fetch_reference_agenda_handles_invalid_response(case, message):
    session = Mock()
    response = make_response([] if case == "list" else {}, 401 if case == "http" else 200)
    session.get.return_value = response
    if case == "json":
        response.json.side_effect = ValueError("Invalid JSON")
    elif case == "network":
        session.get.side_effect = requests.RequestException("Network failure")
    with pytest.raises(RuntimeError, match=message):
        openagenda.fetch_reference_agenda(session)


# ---------- Tests de get_start_date() ----------
# Les dates fixes rendent les tests reproductibles, quelle que soit la date du jour.
# Vérifier minuit et le fuseau UTC évite une borne dépendante de l'heure d'exécution.
# Le cas du 29 février contrôle explicitement le repli au 28 février précédent.
@pytest.mark.parametrize("date, expected", [
    (datetime(2026, 10, 9, 15, 30, tzinfo=timezone.utc), "2025-10-09T00:00:00+00:00"),
    (datetime(2024, 2, 29, 12, tzinfo=timezone.utc), "2023-02-28T00:00:00+00:00"),
])
def test_get_start_date_returns_previous_year_midnight(date, expected):
    assert openagenda.get_start_date(date) == expected


# ---------- Tests de fetch_events() ----------
# Remplacer fetch_pages permet d'observer uniquement la construction de la requête.
# Le test vérifie la ville, la période et les options sans relancer les tests de pagination.
def test_fetch_events_sends_expected_filters(monkeypatch):
    events = [{"uid": 7}]
    fetch = Mock(return_value=events)
    monkeypatch.setattr(openagenda, "fetch_pages", fetch)
    session = Mock()
    date = "2025-10-09T00:00:00+00:00"
    assert openagenda.fetch_events(session, 123, date) is events
    fetch.assert_called_once_with(session, "/agendas/123/events", {
        "adminLevel4": "Toulouse", "size": 100, "timings[gte]": date,
        "detailed": 1, "state": 2,
    }, "events")


# ---------- Tests de save_json() ----------
# tmp_path est un dossier temporaire propre à ce test, sans données réelles.
# Ici l'écriture est réelle : relire le JSON contrôle son contenu et son UTF-8.
# L'absence du .tmp confirme sa publication sous le nom définitif après le succès.
def test_save_json_writes_utf8_and_removes_temporary_file(tmp_path):
    path = tmp_path / "nested" / "events.json"
    payload = {"events": [{"title": "Événement à Toulouse"}]}
    openagenda.save_json(path, payload)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == payload
    assert "Événement à Toulouse" in text
    assert not path.with_suffix(".json.tmp").exists()


# ---------- Tests de main() ----------
# Le chargement .env est neutralisé pour ne jamais lire une vraie clé.
# Session, collecte et sauvegarde sont remplacées localement dans chaque test.
# Sans clé, main doit s'arrêter avant la création d'une session et toute sauvegarde.
# assert_not_called vérifie cette absence d'effet ; capsys contrôle le message d'erreur.
def test_main_returns_error_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(openagenda, "load_dotenv", Mock())
    monkeypatch.delenv("OPENAGENDA_API_KEY", raising=False)
    session = Mock()
    save = Mock()
    monkeypatch.setattr(openagenda.requests, "Session", session)
    monkeypatch.setattr(openagenda, "save_json", save)
    assert openagenda.main() == 1
    assert "OPENAGENDA_API_KEY" in capsys.readouterr().err
    session.assert_not_called()
    save.assert_not_called()


# MagicMock simule l'entrée et la sortie de with requests.Session(). Le test
# vérifie que toutes les récupérations partagent cette même session, puis sa fermeture.
# La sauvegarde simulée capture le contenu prévu sans écrire de fichier.
# Le même événement apparaît dans deux agendas : ses deux occurrences doivent rester brutes.
def test_main_orchestrates_collection_and_saves_metadata(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(openagenda, "load_dotenv", Mock())
    monkeypatch.setenv("OPENAGENDA_API_KEY", "fake-test-key")
    monkeypatch.setattr(openagenda, "OUTPUT_DIRECTORY", tmp_path)
    session = MagicMock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(openagenda.requests, "Session", factory)
    reference = {"schema": {"fields": []}}
    agendas = [{"uid": 1, "title": "A"}, {"uid": 2, "title": "B"}]
    event = {"uid": 7}
    fetch_reference = Mock(return_value=reference)
    fetch_agendas = Mock(return_value=agendas)
    fetch_events = Mock(side_effect=[[event], [event]])
    save = Mock()
    monkeypatch.setattr(openagenda, "fetch_reference_agenda", fetch_reference)
    monkeypatch.setattr(openagenda, "fetch_agendas", fetch_agendas)
    monkeypatch.setattr(openagenda, "fetch_events", fetch_events)
    monkeypatch.setattr(openagenda, "save_json", save)

    assert openagenda.main() == 0
    fetch_reference.assert_called_once_with(session)
    fetch_agendas.assert_called_once_with(session)
    path, payload = save.call_args.args
    # Utiliser la date produite par main pour contrôler le nom et la borne annuelle,
    # sans imposer une date système fixe à ce test d'orchestration.
    date = datetime.fromisoformat(payload["collected_at"])
    assert path == tmp_path / f"openagenda_{date:%Y%m%dT%H%M%S%fZ}.json"
    assert payload == {
        "collected_at": date.isoformat(), "location_set_uid": 15245864,
        "city": "Toulouse", "timings_gte": openagenda.get_start_date(date),
        "reference_agenda": reference, "agenda_count": 2, "agendas": agendas,
        "event_count": 2, "events": [event, event],
    }
    assert [call.args for call in fetch_events.call_args_list] == [
        (session, 1, payload["timings_gte"]), (session, 2, payload["timings_gte"]),
    ]
    factory.assert_called_once_with()
    factory.return_value.__exit__.assert_called_once()
    assert "fake-test-key" not in json.dumps(payload)
    captured = capsys.readouterr()
    assert "fake-test-key" not in captured.out + captured.err


# Une panne pendant la collecte doit produire un échec, sans publier un RAW partiel.
# La sortie du bloc with reste nécessaire même lorsque fetch_events lève une erreur.
def test_main_returns_error_without_saving_after_collection_failure(monkeypatch, capsys):
    monkeypatch.setattr(openagenda, "load_dotenv", Mock())
    monkeypatch.setenv("OPENAGENDA_API_KEY", "fake-test-key")
    factory = MagicMock()
    monkeypatch.setattr(openagenda.requests, "Session", factory)
    monkeypatch.setattr(openagenda, "fetch_reference_agenda", Mock(return_value={}))
    monkeypatch.setattr(openagenda, "fetch_agendas", Mock(return_value=[{"uid": 1, "title": "A"}]))
    monkeypatch.setattr(openagenda, "fetch_events", Mock(side_effect=RuntimeError("Simulated failure")))
    save = Mock()
    monkeypatch.setattr(openagenda, "save_json", save)
    assert openagenda.main() == 1
    assert "Simulated failure" in capsys.readouterr().err
    save.assert_not_called()
    factory.return_value.__exit__.assert_called_once()

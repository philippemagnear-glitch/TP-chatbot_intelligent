"""Collecter les événements de Toulouse depuis un an et ceux à venir.

Le jeu de lieux 15245864 a été identifié à partir du locationSetUid de l'agenda
officiel Toulouse (UID 42448083). Il sert à sélectionner les agendas sources.
Leurs événements sont filtrés par ville et date dans l'API, puis conservés
sans transformation dans un JSON horodaté sous App/Data/raw/.
"""

# ---------- Imports ----------
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


# ---------- Configuration ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "App" / "Data" / "raw"
API_URL = "https://api.openagenda.com/v2"
CITY = "Toulouse"
TOULOUSE_LOCATION_SET_UID = 15245864
# Délais en secondes : connexion au serveur, puis attente de sa réponse.
TIMEOUT = (10, 30)


# ---------- Lecture paginée et validation des réponses ----------
def fetch_pages(session: requests.Session, route: str, params: dict, collection: str) -> list[dict]:
    """Rassembler les pages d'agendas ou d'événements avec les mêmes contrôles."""
    params = params.copy()
    items = []
    seen_cursors = set()

    while True:
        # ---------- Appel HTTP ----------
        # La session porte la clé ; aucune clé ni réponse brute n'est affichée.
        try:
            response = session.get(
                f"{API_URL}{route}",
                params=params,
                timeout=TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException:
            raise RuntimeError(f"Erreur réseau pendant la lecture de {route}.") from None

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAgenda a renvoyé HTTP {response.status_code} pour {route}. "
                "Vérifiez la clé API et les droits d'accès."
            )
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(f"Réponse JSON invalide pour {route}.") from None

        # ---------- Validation de la page ----------
        if not isinstance(payload, dict):
            raise RuntimeError(f"La réponse de {route} doit être un objet JSON.")
        page = payload.get(collection)
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise RuntimeError(f"La liste {collection} est invalide pour {route}.")

        # ---------- Pagination ----------
        # Ajouter les objets sans modifier leurs champs et sans supprimer les doublons.
        items.extend(page)
        cursor = payload.get("after")
        if cursor is None:
            return items
        if (
            not isinstance(cursor, list)
            or not cursor
            or any(type(value) not in (str, int, float) for value in cursor)
        ):
            raise RuntimeError(f"Le curseur after est invalide pour {route}.")

        # Un curseur déjà rencontré signale une boucle ; une page vide ne progresse pas.
        cursor_key = tuple(cursor)
        if cursor_key in seen_cursors or not page:
            raise RuntimeError(f"La pagination ne progresse plus pour {route}.")
        seen_cursors.add(cursor_key)
        params["after[]"] = cursor


# ---------- Recherche des agendas ----------
def fetch_agendas(session: requests.Session) -> list[dict]:
    """Lire les agendas associés au jeu de lieux de Toulouse."""
    params = {"locationSet": TOULOUSE_LOCATION_SET_UID, "size": 100}
    agendas = fetch_pages(session, "/agendas", params, "agendas")
    # Vérifier les identifiants avant de les utiliser pour construire les URL.
    for agenda in agendas:
        uid = agenda.get("uid")
        if type(uid) not in (str, int) or not str(uid).isdecimal() or int(uid) <= 0:
            raise RuntimeError("Un agenda récupéré possède un UID absent ou invalide.")
        if not isinstance(agenda.get("title"), str):
            raise RuntimeError("Un agenda récupéré possède un titre absent ou invalide.")
    return agendas


# ---------- Calcul de la période de collecte ----------
def get_start_date(collected_at: datetime) -> str:
    """Calculer la borne annuelle commune à tous les appels d'événements."""
    today = collected_at.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        start_date = today.replace(year=today.year - 1)
    except ValueError:
        # Le 29 février devient le 28 février de l'année précédente.
        start_date = today.replace(year=today.year - 1, day=28)
    return start_date.isoformat()


# ---------- Lecture des événements d'un agenda ----------
def fetch_events(session: requests.Session, agenda_uid: str | int, start_date: str) -> list[dict]:
    """Lire les événements publiés à Toulouse depuis la borne, sans limite future."""
    params = {
        "adminLevel4": CITY,
        "size": 100,
        "timings[gte]": start_date,
        "detailed": 1,  # Conserver toutes les informations publiques disponibles.
        "state": 2,  # Demander les événements publiés, même avec une clé secrète.
    }
    return fetch_pages(session, f"/agendas/{agenda_uid}/events", params, "events")


# ---------- Sauvegarde d'un JSON complet ----------
def save_json(output_path: Path, payload: dict) -> None:
    """Publier le fichier définitif seulement lorsque son écriture est terminée."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    # Une interruption laisse au plus un .tmp, jamais un JSON considéré comme terminé.
    # Au prochain essai, ce fichier temporaire peut être réécrit.
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(output_path)


# ---------- Orchestration de la collecte ----------
def main() -> int:
    """Retourner 0 pour une collecte complète, 1 en cas d'erreur."""
    try:
        # ---------- Chargement de la clé API ----------
        # Les variables fournies par Docker restent prioritaires sur le fichier .env.
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        api_key = os.getenv("OPENAGENDA_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAGENDA_API_KEY est absente ou vide.")
        collected_at = datetime.now(timezone.utc)
        start_date = get_start_date(collected_at)

        # ---------- Session commune à toute la collecte ----------
        # Réutiliser la connexion ; with ferme la session même en cas d'erreur.
        with requests.Session() as session:
            session.headers.update({"key": api_key, "Accept": "application/json"})
            agendas = fetch_agendas(session)
            events = []
            for position, agenda in enumerate(agendas, start=1):
                print(f"Collecte de l'agenda {position}/{len(agendas)} (UID {agenda['uid']}).")
                events.extend(fetch_events(session, agenda["uid"], start_date))

        # ---------- Traçabilité et sauvegarde des événements bruts ----------
        # Les doublons restent présents ; leur traitement appartient au preprocessing.
        payload = {
            "collected_at": collected_at.isoformat(),
            "location_set_uid": TOULOUSE_LOCATION_SET_UID,
            "city": CITY,
            "timings_gte": start_date,
            "agenda_count": len(agendas),
            "agendas": [
                {"uid": agenda["uid"], "title": agenda["title"]}
                for agenda in agendas
            ],
            "event_count": len(events),
            "events": events,
        }
        output_path = OUTPUT_DIRECTORY / f"openagenda_{collected_at:%Y%m%dT%H%M%S%fZ}.json"
        if output_path.exists():
            raise RuntimeError("Le fichier de sortie existe déjà ; collecte non sauvegardée.")
        save_json(output_path, payload)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print(f"{len(agendas)} agenda(s), {len(events)} événement(s) récupéré(s).")
    print(f"Fichier créé : {output_path}")
    return 0


# ---------- Point d'entrée ----------
# Permettre l'exécution directe sans lancer la collecte lors d'un import.
if __name__ == "__main__":
    raise SystemExit(main())

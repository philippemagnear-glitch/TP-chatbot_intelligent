"""Collecter les événements de Toulouse depuis un an et ceux à venir.

Le script charge la clé et l'UID de l'agenda depuis les variables d'environnement
ou le fichier .env à la racine du projet, sans argument de lancement.
Il rassemble les pages de résultats puis sauvegarde les événements sans
transformation, avec les informations de collecte, dans App/Data/raw/.
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
# Délais en secondes : connexion au serveur, puis attente de sa réponse.
TIMEOUT = (10, 30)


# ---------- Récupération et validation des données ----------
def fetch_events(api_key: str, agenda_uid: str, city: str) -> list[dict]:
    """Lire les événements publiés depuis un an, y compris les événements futurs."""
    # ---------- Validation des paramètres ----------
    # Arrêter avant tout appel réseau si la configuration est incomplète.
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("OPENAGENDA_API_KEY est absente ou vide.")
    if not isinstance(agenda_uid, str) or not agenda_uid.strip():
        raise ValueError("OPENAGENDA_AGENDA_UID est absente ou vide.")
    if (
        not isinstance(agenda_uid, str)
        or not agenda_uid.isdecimal()
        or int(agenda_uid) <= 0
    ):
        raise ValueError("OPENAGENDA_AGENDA_UID doit être un entier positif.")
    if not isinstance(city, str) or not city.strip():
        raise ValueError("Le nom de la ville ne peut pas être vide.")

    # ---------- Période recherchée ----------
    # Partir de minuit UTC, un an auparavant, sans limite de date future.
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        start_date = today.replace(year=today.year - 1)
    except ValueError:
        # Le 29 février devient le 28 février de l'année précédente.
        start_date = today.replace(year=today.year - 1, day=28)

    # ---------- Filtres de la requête API ----------
    params = {
        "adminLevel4[]": city.strip(),
        "timings[gte]": start_date.isoformat(),
        "detailed": 1,  # Toutes les informations publiques de l'événement.
        "monolingual": "fr",  # Textes français, si disponibles.
        "state": 2,  # Événements publiés.
        "size": 100,  # Au maximum 100 événements par page.
    }
    events = []
    seen_cursors = set()

    # ---------- Connexion et récupération des pages ----------
    # Réutiliser la connexion ; with ferme la session à la fin du traitement.
    # Transmettre la clé en en-tête pour éviter de l'inclure dans l'URL.
    with requests.Session() as session:
        session.headers.update({"key": api_key.strip(), "Accept": "application/json"})
        while True:
            try:
                response = session.get(
                    f"{API_URL}/agendas/{agenda_uid}/events",
                    params=params,
                    timeout=TIMEOUT,
                    allow_redirects=False,  # Éviter de transmettre la clé à une autre URL.
                )
            except requests.RequestException:
                raise RuntimeError("Impossible de joindre OpenAgenda : vérifiez votre connexion.") from None

            # ---------- Validation de la réponse ----------
            # Contrôler le statut HTTP et la structure JSON avant d'utiliser les données.
            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenAgenda a renvoyé le statut HTTP {response.status_code}. "
                    "Vérifiez la clé API, l'agenda et les droits d'accès."
                )
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError("OpenAgenda a renvoyé une réponse JSON invalide.") from None

            if not isinstance(payload, dict):
                raise RuntimeError("La réponse OpenAgenda doit être un objet JSON.")
            page = payload.get("events")
            if not isinstance(page, list) or any(not isinstance(event, dict) for event in page):
                raise RuntimeError("La réponse OpenAgenda contient une liste d'événements invalide.")
            if "after" not in payload:
                raise RuntimeError("Le curseur de pagination est absent de la réponse.")

            # ---------- Pagination ----------
            # Ajouter la page courante ; after permet de demander la suivante.
            events.extend(page)
            cursor = payload["after"]
            if cursor is None:
                # Un curseur null indique que toutes les pages ont été récupérées.
                return events
            if (
                not isinstance(cursor, list)
                or not cursor
                or any(type(value) not in (str, int) for value in cursor)
            ):
                raise RuntimeError("Le curseur de pagination OpenAgenda est invalide.")
            # Mémoriser les curseurs pour interrompre une éventuelle boucle infinie.
            cursor_key = tuple(cursor)
            if cursor_key in seen_cursors or not page:
                raise RuntimeError("La pagination OpenAgenda ne progresse plus.")
            seen_cursors.add(cursor_key)
            params["after[]"] = cursor


# ---------- Exécution et sauvegarde ----------
def main() -> int:
    """Orchestrer la collecte : retourner 0 en cas de succès, 1 en cas d'erreur."""
    try:
        # ---------- Chargement de la configuration OpenAgenda ----------
        # Les variables du système ou du conteneur Docker restent prioritaires.
        # Le fichier .env complète la configuration locale sans les écraser.
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        api_key = os.getenv("OPENAGENDA_API_KEY", "").strip()
        agenda_uid = os.getenv("OPENAGENDA_AGENDA_UID", "").strip()

        # ---------- Collecte des événements ----------
        # fetch_events valide les valeurs avant de contacter l'API.
        events = fetch_events(api_key, agenda_uid, CITY)
        # ---------- Préparation du fichier de sortie ----------
        # L'horodatage distingue chaque collecte et préserve les fichiers précédents.
        collected_at = datetime.now(timezone.utc)
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIRECTORY / (
            f"openagenda_{agenda_uid}_{collected_at:%Y%m%dT%H%M%S%fZ}.json"
        )
        # Conserver les événements tels que reçus et identifier leur provenance.
        payload = {
            "agenda_uid": agenda_uid,
            "city": CITY,
            "collected_at": collected_at.isoformat(),
            "count": len(events),
            "events": events,
        }
        # ---------- Sauvegarde JSON ----------
        # Le mode x refuse l'écrasement ; UTF-8 préserve les accents.
        with output_path.open("x", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    except (ValueError, RuntimeError, OSError) as error:
        # Afficher une erreur lisible et signaler l'échec au terminal.
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print(f"{len(events)} événement(s) récupéré(s) pour {CITY} : dernière année et dates futures.")
    print(f"Fichier créé : {output_path}")
    return 0


# ---------- Point d'entrée du script ----------
# Exécuter main lors du lancement direct, mais pas lors d'un import.
if __name__ == "__main__":
    raise SystemExit(main())

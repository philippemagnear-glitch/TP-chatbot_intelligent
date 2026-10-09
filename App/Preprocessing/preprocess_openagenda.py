"""Structurer le dernier RAW OpenAgenda pour le futur chatbot et le RAG.

La première occurrence de chaque UID est conservée. Les catégories sont
décodées depuis le schéma de l'agenda de référence, sans correspondance en dur.
Le RAW reste intact ; seul un nouveau JSON processed est écrit.

Le traitement suit quatre étapes : charger et valider le RAW le plus récent,
identifier les événements uniques, sélectionner et décoder leurs informations,
puis sauvegarder le résultat avec sa provenance et les compteurs de traitement.
Les listes de séances et les liens sont conservés pour les futures réponses du RAG.
"""

# ---------- Imports et configuration des chemins ----------
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Résoudre les chemins depuis ce fichier permet de lancer le script depuis la racine.
RAW_DIRECTORY = PROJECT_ROOT / "App" / "Data" / "raw"
PROCESSED_DIRECTORY = PROJECT_ROOT / "App" / "Data" / "processed"

# Ces codes fixes décrivent l'accessibilité et le statut métier OpenAgenda.
# Les catégories event_types et public viennent, elles, du schéma du RAW.
ACCESSIBILITY_LABELS = {
    "hi": "Handicap auditif",
    "vi": "Handicap visuel",
    "pi": "Handicap psychique",
    "mi": "Handicap moteur",
    "ii": "Handicap intellectuel",
}

EVENT_STATUS_LABELS = {
    1: "Programmé",
    2: "Reprogrammé",
    3: "Déplacé en ligne",
    4: "Reporté",
    5: "Complet",
    6: "Annulé",
}


# ---------- Chargement des données ----------
def load_raw() -> tuple[Path, dict]:
    """Choisir le fichier le plus récent selon sa date de modification."""
    # ---------- Sélection et lecture du RAW ----------
    # Seuls les exports globaux OpenAgenda sont candidats ; les anciens restent intacts.
    files = [path for path in RAW_DIRECTORY.glob("openagenda_*.json") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"Aucun fichier openagenda_*.json dans {RAW_DIRECTORY}.")
    path = max(files, key=lambda item: (item.stat().st_mtime_ns, item.name))
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    # ---------- Validation minimale ----------
    # Échouer clairement si les événements ou le schéma ne sont pas exploitables.
    if not isinstance(payload, dict):
        raise ValueError("Le RAW doit être un objet JSON.")
    if not isinstance(payload.get("events"), list):
        raise ValueError("Le champ events doit être une liste.")
    if any(not isinstance(event, dict) for event in payload["events"]):
        raise ValueError("Chaque événement doit être un objet JSON.")
    if not isinstance(payload.get("reference_agenda"), dict):
        raise ValueError("reference_agenda est absent ou n'est pas un objet JSON.")
    return path, payload


# ---------- Lecture du schéma ----------
def get_field_options(reference_agenda: dict, field_name: str) -> dict:
    """Lire les options françaises du champ identifié par sa propriété field."""
    # Le nom du champ est stable, contrairement à sa position dans la liste fields.
    # Les clés textuelles permettent de rapprocher un ID numérique et un ID en chaîne.
    schema = reference_agenda.get("schema")
    fields = schema.get("fields") if isinstance(schema, dict) else None
    if not isinstance(fields, list):
        warnings.warn("Schéma absent ou invalide : les identifiants seront conservés.")
        return {}
    for field in fields:
        if not isinstance(field, dict) or field.get("field") != field_name:
            continue
        options = field.get("options")
        if not isinstance(options, list):
            break
        labels = {}
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = option.get("id")
            label = option.get("label")
            french_label = label.get("fr") if isinstance(label, dict) else None
            if type(option_id) in (int, str) and isinstance(french_label, str) and french_label.strip():
                labels[str(option_id)] = french_label
        return labels
    warnings.warn(f"Options du champ {field_name} introuvables : identifiants conservés.")
    return {}


def decode_options(value, options: dict, field_name: str):
    """Conserver une liste et signaler les identifiants sans libellé connu."""
    # Un événement peut avoir plusieurs catégories. Une valeur absente reste None,
    # et une liste vide reste vide ; aucun libellé n'est inventé pour un ID inconnu.
    if value is None or value == {} or value == "":
        return None
    values = value if isinstance(value, list) else [value]
    result = []
    for option_id in values:
        label = options.get(str(option_id)) if type(option_id) in (int, str) else None
        if label is None:
            # Le warning est regroupé par Python si le même ID revient plusieurs fois.
            warnings.warn(f"Champ {field_name} : identifiant non résolu {option_id!r}, conservé.")
            result.append(option_id)
        else:
            result.append(label)
    return result


# ---------- Dédoublonnage ----------
def deduplicate_events(events: list[dict]) -> tuple[list[dict], int, int]:
    """Garder la première occurrence, sans fusion des éventuelles différences."""
    # Un même événement peut provenir de plusieurs agendas. L'ensemble seen_uids
    # permet de reconnaître les occurrences suivantes sans comparer tous les champs.
    # Sans UID exploitable, l'événement est ignoré et compté séparément des doublons.
    unique_events = []
    seen_uids = set()
    duplicate_count = 0
    invalid_uid_count = 0
    for event in events:
        uid = event.get("uid")
        if type(uid) not in (int, str) or not str(uid).isdecimal() or int(uid) <= 0:
            invalid_uid_count += 1
            continue
        # Un UID numérique et sa représentation textuelle désignent le même événement.
        uid_key = int(uid)
        if uid_key in seen_uids:
            duplicate_count += 1
            continue
        seen_uids.add(uid_key)
        unique_events.append(event)
    return unique_events, duplicate_count, invalid_uid_count


# ---------- Transformation des événements ----------
def french_value(value):
    """Extraire le français ou conserver un champ déjà monolingue."""
    # Les exports peuvent contenir un dictionnaire par langue ou directement le texte.
    # Si le français manque, ne pas le remplacer par une description artificielle.
    if isinstance(value, dict):
        value = value.get("fr")
    if value is None or value == "" or value == {}:
        return None
    return value


def decode_accessibility(value) -> list[str]:
    """Décoder les handicaps indiqués, en conservant les codes inconnus."""
    # La liste décrit uniquement les accès explicitement indiqués comme disponibles.
    # Une liste vide ne permet pas de distinguer une absence d'accès d'une absence d'information.
    if not isinstance(value, dict):
        return []
    # is True exclut les valeurs simplement équivalentes à True, comme 1.
    return [
        ACCESSIBILITY_LABELS.get(code, code)
        for code, enabled in value.items()
        if enabled is True
    ]


def decode_event_status(value):
    """Traduire le statut métier connu, sans perdre les valeurs inconnues."""
    # status décrit notamment une annulation ou un report ; ce n'est pas l'état de publication.
    # None et les valeurs non reconnues sont conservés pour éviter toute interprétation arbitraire.
    if type(value) is not int:
        return value
    return EVENT_STATUS_LABELS.get(value, value)


def transform_event(event: dict, event_types: dict, public_options: dict) -> dict:
    """Simplifier les champs sans inventer les informations manquantes."""
    # ---------- Lecture des objets optionnels ----------
    # Remplacer seulement les conteneurs absents par {} permet d'utiliser get sans KeyError.
    # Les valeurs finales manquantes restent None, sans supprimer l'événement.
    location = event.get("location")
    location = location if isinstance(location, dict) else {}
    age = event.get("age")
    age = age if isinstance(age, dict) else {}
    origin = event.get("originAgenda")
    origin = origin if isinstance(origin, dict) else {}
    # ---------- Construction de l'événement processed ----------
    # Créer un nouvel objet : l'événement RAW n'est jamais modifié.
    return {
        "uid": event.get("uid"),
        "title": french_value(event.get("title")),
        "description": french_value(event.get("description")),
        "long_description": french_value(event.get("longDescription")),
        "keywords": french_value(event.get("keywords")),
        # Conserver toutes les séances, sans réduire l'événement à sa première date.
        "timings": event.get("timings"),
        "date_range": french_value(event.get("dateRange")),
        "location_name": location.get("name"),
        "address": location.get("address"),
        "postal_code": location.get("postalCode"),
        "city": location.get("city"),
        "district": location.get("district"),
        "metropole": location.get("adminLevel3"),
        "department": location.get("adminLevel2"),
        "region": location.get("adminLevel1"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "access": french_value(location.get("access")),
        "event_types": decode_options(event.get("types-devenements"), event_types, "types-devenements"),
        "public": decode_options(event.get("public"), public_options, "public"),
        "age_min": age.get("min"),
        "age_max": age.get("max"),
        "accessibility": decode_accessibility(event.get("accessibility")),
        "conditions": french_value(event.get("conditions")),
        "billetterie": event.get("billetterie"),
        "event_links": event.get("links"),
        "online_access_link": event.get("onlineAccessLink"),
        "location_website": location.get("website"),
        "slug": event.get("slug"),
        "status": decode_event_status(event.get("status")),
        "origin_agenda": origin.get("title"),
    }


# ---------- Sauvegarde ----------
def save_processed(payload: dict, processed_at: datetime) -> Path:
    """Écrire en UTF-8, puis publier seulement le JSON complètement écrit."""
    # L'horodatage distingue les traitements ; le mode x refuse d'écraser un fichier.
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIRECTORY / f"openagenda_processed_{processed_at:%Y%m%dT%H%M%S%fZ}.json"
    if path.exists():
        raise FileExistsError(f"Le fichier processed existe déjà : {path}.")
    temporary_path = path.with_suffix(".json.tmp")
    # Une interruption pendant l'écriture laisse un .tmp plutôt qu'un JSON final incomplet.
    with temporary_path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary_path.replace(path)
    return path


# ---------- Exécution ----------
def main() -> int:
    """Orchestrer les étapes et retourner 0 en cas de succès, 1 en cas d'erreur."""
    try:
        # ---------- Chargement et préparation des correspondances ----------
        # Lire le schéma une seule fois pour réutiliser ses libellés sur tous les événements.
        raw_path, raw = load_raw()
        event_types = get_field_options(raw["reference_agenda"], "types-devenements")
        public_options = get_field_options(raw["reference_agenda"], "public")
        # ---------- Dédoublonnage et transformation ----------
        # Garder d'abord une occurrence par UID, puis construire les champs utiles au RAG.
        unique_events, duplicate_count, invalid_uid_count = deduplicate_events(raw["events"])
        events = [transform_event(event, event_types, public_options) for event in unique_events]
        # ---------- Traçabilité et sauvegarde ----------
        # Le nom du RAW et les compteurs permettent de retrouver l'origine du résultat.
        processed_at = datetime.now(timezone.utc)
        payload = {
            "source_file": raw_path.name,
            "processed_at": processed_at.isoformat(),
            "raw_event_count": len(raw["events"]),
            "duplicate_count": duplicate_count,
            "ignored_uid_count": invalid_uid_count,
            "event_count": len(events),
            "events": events,
        }
        output_path = save_processed(payload, processed_at)
    except (OSError, ValueError) as error:
        # Signaler les erreurs de fichier ou de données sans afficher une longue trace technique.
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    # ---------- Bilan du traitement ----------
    # Les UID invalides et les doublons sont des exclusions distinctes.
    print(f"Fichier RAW : {raw_path.name}")
    print(f"Événements bruts : {len(raw['events'])}")
    print(f"Événements après dédoublonnage : {len(unique_events)}")
    print(f"Doublons supprimés : {duplicate_count}")
    print(f"Événements sans UID exploitable ignorés : {invalid_uid_count}")
    print(f"Événements sauvegardés : {len(events)}")
    print(f"Fichier processed : {output_path}")
    return 0


# ---------- Point d'entrée ----------
# Lancer le traitement uniquement en exécution directe, jamais lors d'un import.
if __name__ == "__main__":
    raise SystemExit(main())

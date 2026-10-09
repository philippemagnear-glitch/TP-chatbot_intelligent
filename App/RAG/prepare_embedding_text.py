"""Préparer les textes des événements OpenAgenda pour la vectorisation.

Les événements proviennent du JSON processed, déjà nettoyé et structuré.

L'objectif est de construire un texte enrichi et lisible pour le modèle
mistral-embed, en conservant les données structurées dans le JSON d'origine.

Cette étape ne génère pas encore d'embeddings et n'effectue aucun appel API.
"""

# ---------- Préparation du texte ----------

def build_embedding_text(event: dict) -> str:
    """Construire un texte enrichi à partir d'un événement OpenAgenda.

    Les informations disponibles sont organisées sous forme de lignes
    explicites pour faciliter leur interprétation par le modèle NLP.

    Les valeurs absentes ou vides sont ignorées afin de ne pas introduire
    d'informations artificielles.
    """
    if not isinstance(event, dict):
        raise TypeError("L'événement doit être un dictionnaire.")

    parts = []

    # ---------- Champs textuels ----------
    text_fields = {
        "title": "Titre",
        "description": "Description",
        "long_description": "Description détaillée",
        "date_range": "Dates et horaires",
        "location_name": "Lieu",
        "address": "Adresse",
        "city": "Ville",
        "district": "Quartier",
        "metropole": "Métropole",
        "department": "Département",
        "region": "Région",
        "access": "Transports et accès",
        "conditions": "Tarifs et conditions",
        "status": "Statut",
        "origin_agenda": "Agenda d'origine",
    }

    for field, label in text_fields.items():
        value = event.get(field)

        if isinstance(value, str) and value.strip():
            parts.append(f"{label} : {value.strip()}")

    # ---------- Champs contenant des listes ----------
    list_fields = {
        "keywords": "Mots-clés",
        "event_types": "Types d'événements",
        "public": "Public",
        "accessibility": "Accessibilité",
    }

    for field, label in list_fields.items():
        values = event.get(field)

        if isinstance(values, list):
            valid_values = [
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip()
            ]

            if valid_values:
                parts.append(f"{label} : {', '.join(valid_values)}")

    # ---------- Tranche d'âge ----------
    age_min = event.get("age_min")
    age_max = event.get("age_max")

    if isinstance(age_min, int) and not isinstance(age_min, bool):
        parts.append(f"Âge minimum : {age_min} ans")

    if isinstance(age_max, int) and not isinstance(age_max, bool):
        parts.append(f"Âge maximum : {age_max} ans")

    # ---------- Résultat ----------
    if not parts:
        raise ValueError("L'événement ne contient aucune information exploitable.")

    return "\n".join(parts)
"""Collecter les réponses du chatbot pour une version, sans notation automatique.

L'exécution réelle pose les dix questions à Mistral et consomme de l'usage API.
Les réponses attendues sont enregistrées pour la lecture, jamais envoyées au chatbot.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

# Permet l'import aussi lors du lancement direct de ce fichier depuis la racine.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from App.RAG import chatbot


EVALUATION_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVALUATION_DIR / "questions.json"
RESULTS_DIR = EVALUATION_DIR / "results"


# ---------- Chargement du jeu de questions ----------
def load_questions() -> list[dict]:
    """Lire et valider les dix questions, sans changer leurs formulations."""
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or len(questions) != 10:
        raise ValueError("questions.json doit contenir exactement dix questions.")
    ids = set()
    for question in questions:
        if (not isinstance(question, dict)
                or not all(isinstance(question.get(key), str) and question[key].strip()
                           for key in ("id", "category", "question", "expected_answer"))
                or not isinstance(question.get("expected_uids"), list)
                or not question["expected_uids"]
                or not all(type(uid) is int and uid > 0 for uid in question["expected_uids"])):
            raise ValueError("Question incorrecte : vérifier les champs et les UID attendus.")
        if question["id"] in ids:
            raise ValueError(f"Identifiant de question dupliqué : {question['id']}")
        ids.add(question["id"])
    return questions


# ---------- Collecte des réponses ----------
def evaluate(version: str) -> Path:
    """Charger les ressources une fois et conserver chaque réponse ou erreur sur disque."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version):
        raise ValueError("Version incorrecte : utiliser lettres, chiffres, tirets ou underscores.")
    questions = load_questions()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"{version}.txt"
    # Le mode x refuse l'écrasement, avant toute initialisation ou requête API.
    try:
        file = output.open("x", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError(f"Résultat déjà présent : {output}. Choisir un autre nom de version.") from None
    with file:
        date = datetime.now(ZoneInfo("Europe/Paris")).isoformat(timespec="seconds")
        file.write(f"VERSION : {version}\nDATE : {date}\nMODÈLE : {chatbot.CHAT_MODEL}\n"
                   f"SEARCH_K : {chatbot.SEARCH_K}\nMAX_EVENTS : {chatbot.MAX_EVENTS}\n"
                   f"USE_MARKDOWN_CONTEXT : {chatbot.USE_MARKDOWN_CONTEXT}\n\n")
        try:
            store, llm, prompt = chatbot.load_resources()
        except (httpx.HTTPError, OSError, ValueError, RuntimeError, EOFError) as error:
            file.write(f"ERREUR D'INITIALISATION : {type(error).__name__}\n")
            file.flush()
            raise
        # load_resources() lit prompt.md : conserver ce Markdown tel qu'il est
        # transmis au message système, sans relire ni convertir le fichier.
        file.write(f"PROMPT UTILISÉ :\n{prompt}\n\n---\n\n")
        file.flush()
        for index, question in enumerate(questions, start=1):
            print(f"Question {index}/{len(questions)} — {question['id']}")
            file.write(f"QUESTION {question['id']} — {question['category']}\n\n"
                       f"Question :\n{question['question']}\n\n"
                       f"Réponse attendue :\n{question['expected_answer']}\n\n"
                       f"UID attendus : {', '.join(map(str, question['expected_uids']))}\n\n")
            file.flush()
            try:
                answer = chatbot.generate_answer(question["question"], store, llm, prompt)
            except KeyboardInterrupt:
                file.write("STATUT : INTERROMPU PAR L'UTILISATEUR\nAucune réponse enregistrée pour cette question.\n")
                file.flush()
                os.fsync(file.fileno())
                print(f"\nÉvaluation interrompue. Résultats conservés : {output}")
                return output
            except (httpx.HTTPError, ValueError) as error:
                # Une erreur technique n'est ni une réponse du modèle ni une note.
                message = str(error)
                key = os.getenv("MISTRAL_API_KEY")
                if key:
                    message = message.replace(key, "[clé masquée]")
                file.write(f"STATUT : ERREUR TECHNIQUE\n{type(error).__name__} : {message}\n")
            else:
                file.write(f"STATUT : RÉPONSE OBTENUE\nRéponse générée :\n{answer}\n")
            file.write("\n---\n\n")
            # Chaque question terminée est conservée avant de passer à la suivante.
            file.flush()
            os.fsync(file.fileno())
    print(f"Résultats enregistrés : {output}")
    return output


# ---------- Lancement depuis le terminal ----------
def main() -> None:
    """Accepter un nom de version et lancer la collecte des dix réponses."""
    parser = argparse.ArgumentParser(description="Collecter les réponses du chatbot RAG.")
    parser.add_argument("version", help="Nom unique de l'évaluation, par exemple v1")
    args = parser.parse_args()
    try:
        evaluate(args.version)
    except (httpx.HTTPError, OSError, ValueError, RuntimeError, EOFError) as error:
        message = str(error)
        key = os.getenv("MISTRAL_API_KEY")
        if key:
            message = message.replace(key, "[clé masquée]")
        print(f"Évaluation impossible : {message}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\nÉvaluation interrompue. Les résultats déjà écrits sont conservés.")


if __name__ == "__main__":
    main()

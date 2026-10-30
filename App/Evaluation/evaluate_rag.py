"""Évaluer le RAG existant avec Mistral et Ragas, indépendamment de FastAPI.

Depuis la racine : uv run python App/Evaluation/evaluate_rag.py --limit 1
Sans --limit : dix questions métier et deux questions de sécurité.
L'exécution réelle consomme de l'usage Mistral pour le RAG ET pour les métriques.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

# Avant tout import Ragas/LiteLLM : pas de télémétrie ni de chargement distant
# de la table des prix LiteLLM. Cela n'empêche pas les appels Mistral d'évaluation.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from App.RAG import chatbot


QUESTIONS_PATH = Path(__file__).resolve().parent / "question.json"
RESULTS_DIR = chatbot.PROJECT_ROOT / "App" / "Data" / "evaluation"
METRIC_NAMES = ("recall_at_5", "faithfulness", "answer_relevancy", "factual_correctness")
FACTUAL_CORRECTNESS_MODE = "recall"


# ---------- Validation avant tout appel payant ----------
def load_questions() -> list[dict]:
    """Valider les douze références ; seuls Q11 et Q12 ont des UID attendus vides."""
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or len(questions) != 12:
        raise ValueError("Le jeu doit contenir exactement douze questions.")
    ids = set()
    for question in questions:
        if not isinstance(question, dict) or not all(
            isinstance(question.get(key), str) and question[key].strip()
            for key in ("id", "category", "question", "expected_answer")
        ):
            raise ValueError("Chaque question doit avoir id, category, question et expected_answer non vides.")
        uid_list = question.get("expected_uids")
        if (not isinstance(uid_list, list)
                or not all(type(uid) is int and uid > 0 for uid in uid_list)
                or len(set(uid_list)) != len(uid_list)):
            raise ValueError("expected_uids doit être une liste d'entiers positifs uniques.")
        identifier = question["id"]
        if identifier in ids:
            raise ValueError("Identifiant de question dupliqué.")
        ids.add(identifier)
        if (identifier in ("Q11", "Q12")) != (len(uid_list) == 0):
            raise ValueError("UID attendus obligatoires pour Q01 à Q10, et liste vide pour Q11/Q12.")
    if ids != {f"Q{number:02}" for number in range(1, 13)}:
        raise ValueError("Les identifiants doivent aller de Q01 à Q12.")
    return sorted(questions, key=lambda item: item["id"])


# ---------- Évaluateur Mistral, distinct du prompt du chatbot ----------
def build_metrics(judge_model: str) -> dict:
    """Configurer les interfaces modernes Ragas ; aucun appel de score à ce stade.

    Instructor obtient des sorties JSON structurées via LiteLLM. Seule la
    pertinence de réponse utilise aussi mistral-embed, pour les similarités.
    Le français reste celui des questions et réponses : pas de traduction du jeu.
    """
    import instructor
    import litellm
    from ragas.embeddings import LiteLLMEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness, Faithfulness

    litellm.telemetry = False
    key = os.getenv("MISTRAL_API_KEY")
    if not key or not key.strip():
        raise ValueError("MISTRAL_API_KEY absente.")
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
    llm = llm_factory(
        f"mistral/{judge_model}", provider="mistral", client=client, adapter="litellm",
        api_key=key, temperature=0, max_tokens=4096, timeout=60,
        max_retries=1, num_retries=0,
        system_prompt="Évalue les textes en français. Respecte strictement le schéma JSON demandé.",
    )
    embeddings = LiteLLMEmbeddings(model="mistral/mistral-embed", api_key=key,
                                  timeout=60, max_retries=0)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3),
        # Mesurer la présence des faits de référence dans la réponse. Faithfulness
        # reste nécessaire pour vérifier les ajouts par rapport au contexte.
        "factual_correctness": FactualCorrectness(llm=llm, mode=FACTUAL_CORRECTNESS_MODE),
    }


def error_details(error: Exception) -> dict:
    """Conserver le type et le statut HTTP éventuel, jamais le texte brut distant."""
    from tenacity import RetryError

    if isinstance(error, RetryError):
        error = error.last_attempt.exception() or error
    result = {"type": type(error).__name__}
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    return result


# ---------- Résultats individuels et résumé ----------
def summarize(results: list[dict]) -> dict:
    """Exclure la sécurité et compter explicitement les scores manquants."""
    business = [item for item in results if item["kind"] == "business"]
    summary = {}
    for name in METRIC_NAMES:
        values = [item["scores"][name]["value"] for item in business
                  if item["scores"][name]["status"] == "ok"]
        summary[name] = {
            "mean": mean(values) if values else None,
            "evaluated": len(values), "not_evaluated": len(business) - len(values),
            "technical_errors": sum(item["scores"][name]["status"] == "error" for item in business),
        }
    return summary


def save_report(file, report: dict) -> None:
    """Sauvegarder après chaque question, même si la suivante échoue."""
    file.seek(0)
    json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    file.write("\n")
    file.truncate()
    file.flush()
    os.fsync(file.fileno())


async def evaluate_questions(questions: list[dict], resources: tuple, metrics: dict,
                             report: dict, file) -> None:
    """Traiter les questions successivement ; ne jamais transformer une erreur en zéro."""
    store, llm, prompt = resources
    for index, question in enumerate(questions, start=1):
        security = question["id"] in ("Q11", "Q12")
        item = {**question, "kind": "security" if security else "business", "status": "running",
                "answer": None, "context": None, "retrieved_uids": [], "errors": [],
                "scores": {} if security else {
                    name: {"value": None, "status": "not_computed", "reason": "generation_failed"}
                    for name in METRIC_NAMES
                }}
        if security:
            item["security_review"] = "Vérification humaine requise ; aucun verdict automatique."
        collection = "security_results" if security else "results"
        report[collection].append(item)
        print(f"Évaluation {question['id']} ({index}/{len(questions)})")
        try:
            details = chatbot.generate_answer_with_details(question["question"], store, llm, prompt)
            item.update(details)
        except Exception as error:
            item["errors"].append({"stage": "generation", **error_details(error)})
        else:
            if not security:
                # Les UID peuvent être des chaînes dans le DocStore : normaliser
                # uniquement pour la comparaison, sans modifier les données sources.
                expected = {str(uid) for uid in question["expected_uids"]}
                found = {str(uid) for uid in item["retrieved_uids"]}
                item["scores"]["recall_at_5"] = {
                    "value": len(expected & found) / len(expected), "status": "ok",
                }
                for name, metric in metrics.items():
                    if name == "faithfulness" and not item["context"]:
                        item["scores"][name] = {"value": None, "status": "not_computed",
                                                "reason": "no_context"}
                        continue
                    arguments = {"response": item["answer"]}
                    if name == "factual_correctness":
                        arguments["reference"] = question["expected_answer"]
                    else:
                        arguments["user_input"] = question["question"]
                    if name == "faithfulness":
                        # Un seul bloc contient exactement le contexte final du RAG,
                        # y compris sa mise en forme ; pas une seconde recherche.
                        arguments["retrieved_contexts"] = [item["context"]]
                    try:
                        result = await metric.ascore(**arguments)
                        value = float(result.value)
                        if not math.isfinite(value):
                            raise ValueError("Score Ragas non fini.")
                        item["scores"][name] = {"value": value, "status": "ok"}
                    except Exception as error:
                        item["scores"][name] = {"value": None, "status": "error"}
                        item["errors"].append({"stage": name, **error_details(error)})
        item["status"] = "technical_error" if item["errors"] else "completed"
        report["summary"] = summarize(report["results"])
        save_report(file, report)


def main() -> int:
    """Valider avant paiement ; code 1 pour une erreur technique, sans seuil qualité."""
    parser = argparse.ArgumentParser(description="Évaluer le RAG OpenAgenda avec Ragas/Mistral.")
    parser.add_argument("--limit", type=int, help="Nombre de questions, de 1 à 12, dans l'ordre Q01 à Q12.")
    parser.add_argument("--judge-model", default=chatbot.CHAT_MODEL,
                        help="Modèle Mistral évaluateur, sans préfixe mistral/.")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 12:
        parser.error("--limit doit être compris entre 1 et 12.")
    if not args.judge_model.strip() or "/" in args.judge_model:
        parser.error("--judge-model doit être un nom de modèle Mistral sans préfixe.")
    try:
        questions = load_questions()  # Valider les 12 même si --limit vaut 1.
    except (OSError, ValueError) as error:
        print(f"Dataset invalide : {error}", file=sys.stderr)
        return 1
    questions = questions[:args.limit]
    timestamp = datetime.now(ZoneInfo("Europe/Paris"))
    report = {
        "started_at": timestamp.isoformat(), "status": "running",
        "question_ids": [question["id"] for question in questions],
        "configuration": {
            "chat_model": chatbot.CHAT_MODEL, "judge_model": args.judge_model,
            "embedding_model": chatbot.EMBEDDING_MODEL, "search_k_chunks": chatbot.SEARCH_K,
            "max_events": chatbot.MAX_EVENTS, "markdown_context": chatbot.USE_MARKDOWN_CONTEXT,
            "temperature": 0, "answer_relevancy_strictness": 3,
            "factual_correctness_mode": FACTUAL_CORRECTNESS_MODE,
        },
        "results": [], "security_results": [], "summary": {}, "errors": [],
    }
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / f"ragas_{timestamp:%Y%m%d_%H%M%S_%f}.json"
        with output.open("x+", encoding="utf-8") as file:
            save_report(file, report)
            try:
                report["versions"] = {name: version(name) for name in
                                      ("ragas", "litellm", "instructor", "langchain-mistralai", "langchain-core")}
                resources = chatbot.load_resources()
                report["prompt_sha256"] = hashlib.sha256(resources[2].encode("utf-8")).hexdigest()
                metrics = build_metrics(args.judge_model)
                asyncio.run(evaluate_questions(questions, resources, metrics, report, file))
            except KeyboardInterrupt:
                report["errors"].append({"stage": "evaluation", "type": "KeyboardInterrupt"})
            except Exception as error:
                report["errors"].append({"stage": "initialization_or_evaluation", **error_details(error)})
            all_results = report["results"] + report["security_results"]
            failed = bool(report["errors"] or any(item["errors"] for item in all_results))
            report["status"] = "technical_error" if failed else "completed"
            report["finished_at"] = datetime.now(ZoneInfo("Europe/Paris")).isoformat()
            report["summary"] = summarize(report["results"])
            report["questions_completed"] = sum(item["status"] != "running" for item in all_results)
            report["questions_interrupted"] = sum(item["status"] == "running" for item in all_results)
            report["questions_not_started"] = len(questions) - len(all_results)
            save_report(file, report)
        print(f"Rapport : {output}")
        # Le JSON et le statut de l'évaluation restent prioritaires sur l'affichage.
        try:
            from App.Evaluation.generate_report import generate_dashboard

            print(f"Tableau de bord : {generate_dashboard(RESULTS_DIR)}")
        except Exception as error:
            print(f"Tableau de bord non régénéré ({type(error).__name__}) ; rapport JSON conservé.", file=sys.stderr)
        if failed:
            print("Évaluation incomplète : consulter les erreurs et scores non calculés du rapport.", file=sys.stderr)
        return int(failed)
    except OSError:
        print("Impossible de sauvegarder le rapport d'évaluation.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests de l'évaluation avec les questions du projet et des services simulés."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from App.Evaluation import evaluate_rag as evaluation
from App.Evaluation import generate_report


DEFAULT_QUESTIONS_PATH = evaluation.QUESTIONS_PATH


@pytest.fixture(autouse=True)
def isolated_evaluation(monkeypatch, tmp_path):
    """Lire les vraies questions ; isoler les résultats et simuler les appels payants."""
    questions = evaluation.load_questions()
    expected_by_question = {item["question"]: item["expected_uids"] for item in questions}
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(evaluation.chatbot, "load_resources", Mock(return_value=("store", "llm", "prompt fictif")))
    # Simuler une recherche réussie pour vérifier le calcul du rappel,
    # sans prétendre mesurer ici la qualité réelle de FAISS ou de Mistral.
    def fake_answer(question, store, llm, prompt):
        return {"answer": "Réponse simulée.", "context": "Contexte simulé.",
                "retrieved_uids": list(expected_by_question[question])}

    monkeypatch.setattr(evaluation.chatbot, "generate_answer_with_details", Mock(side_effect=fake_answer))
    metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=0.75)))
               for name in ("faithfulness", "answer_relevancy", "factual_correctness")}
    monkeypatch.setattr(evaluation, "build_metrics", Mock(return_value=metrics))
    monkeypatch.setattr(generate_report, "generate_dashboard", Mock(return_value=tmp_path / "dashboard.html"))
    return metrics


def read_report():
    path, = evaluation.RESULTS_DIR.glob("ragas_*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_default_questions_path_points_to_evaluation_folder():
    assert DEFAULT_QUESTIONS_PATH == Path(__file__).resolve().parents[1] / "App" / "Evaluation" / "question.json"


def test_load_questions_accepts_twelve_questions_and_empty_security_uids():
    questions = evaluation.load_questions()
    assert [question["id"] for question in questions] == [f"Q{number:02}" for number in range(1, 13)]
    assert all(question["expected_uids"] == [] for question in questions[-2:])


@pytest.mark.parametrize("problem", ["missing-question", "duplicate-id", "missing-answer", "bad-uid", "empty-business"])
def test_load_questions_rejects_invalid_dataset(problem, monkeypatch, tmp_path):
    questions = evaluation.load_questions()
    if problem == "missing-question":
        questions.pop()
    elif problem == "duplicate-id":
        questions[1]["id"] = "Q01"
    elif problem == "missing-answer":
        questions[0].pop("expected_answer")
    elif problem == "bad-uid":
        questions[0]["expected_uids"] = [True]
    else:
        questions[0]["expected_uids"] = []
    # Ne jamais modifier le jeu du projet : écrire seulement une copie invalide.
    source = tmp_path / "question_invalide.json"
    source.write_text(json.dumps(questions), encoding="utf-8")
    monkeypatch.setattr(evaluation, "QUESTIONS_PATH", source)
    with pytest.raises(ValueError):
        evaluation.load_questions()


def test_empty_json_stops_before_initializing_rag(monkeypatch, tmp_path):
    source = tmp_path / "question_vide.json"
    source.write_text("", encoding="utf-8")
    monkeypatch.setattr(evaluation, "QUESTIONS_PATH", source)
    monkeypatch.setattr(sys, "argv", ["evaluate_rag.py", "--limit", "1"])
    assert evaluation.main() == 1
    evaluation.chatbot.load_resources.assert_not_called()
    evaluation.build_metrics.assert_not_called()


def test_evaluation_separates_security_and_reuses_resources(monkeypatch, isolated_evaluation):
    monkeypatch.setattr(sys, "argv", ["evaluate_rag.py"])
    assert evaluation.main() == 0
    report = read_report()
    assert len(report["results"]) == 10
    assert [item["id"] for item in report["security_results"]] == ["Q11", "Q12"]
    assert all(item["scores"] == {} for item in report["security_results"])
    assert report["summary"]["recall_at_5"]["mean"] == 1
    evaluation.chatbot.load_resources.assert_called_once_with()
    assert evaluation.chatbot.generate_answer_with_details.call_count == 12
    assert all(metric.ascore.await_count == 10 for metric in isolated_evaluation.values())
    generate_report.generate_dashboard.assert_called_once_with(evaluation.RESULTS_DIR)


def test_limit_evaluates_only_first_question(monkeypatch):
    first_question = evaluation.load_questions()[0]
    monkeypatch.setattr(sys, "argv", ["evaluate_rag.py", "--limit", "1"])
    assert evaluation.main() == 0
    assert read_report()["question_ids"] == ["Q01"]
    evaluation.chatbot.generate_answer_with_details.assert_called_once_with(
        first_question["question"], "store", "llm", "prompt fictif",
    )


def test_metric_error_is_not_converted_to_zero(monkeypatch, isolated_evaluation):
    monkeypatch.setattr(sys, "argv", ["evaluate_rag.py", "--limit", "1"])
    isolated_evaluation["faithfulness"].ascore.side_effect = RuntimeError("secret fictif")
    assert evaluation.main() == 1
    report = read_report()
    assert report["summary"]["faithfulness"]["mean"] is None
    assert report["summary"]["faithfulness"]["technical_errors"] == 1
    assert "secret fictif" not in json.dumps(report)
    generate_report.generate_dashboard.assert_called_once_with(evaluation.RESULTS_DIR)

"""Générer un tableau de bord autonome depuis les rapports Ragas, sans appel API.

Commande : uv run python App/Evaluation/generate_report.py
Ouvrir ensuite App/Data/evaluation/dashboard.html dans un navigateur.
"""

import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo


RESULTS_DIR = Path(__file__).resolve().parents[1] / "Data" / "evaluation"
METRICS = {
    "recall_at_5": "Recall@5",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "factual_correctness": "Factual Correctness",
}
STATUSES = {"completed": "Terminé", "technical_error": "Erreur technique", "running": "En cours / interrompu"}


# ---------- Lecture et validation légère des rapports ----------
def load_reports(directory: Path) -> tuple[list[dict], list[str]]:
    """Ignorer les rapports invalides, sans modifier leurs fichiers sources."""
    reports, warnings = [], []
    for path in directory.glob("ragas_*.json"):
        try:
            with path.open(encoding="utf-8") as file:
                report = json.load(file)
            if not isinstance(report, dict) or "results" not in report:
                raise ValueError("Objet rapport avec une liste results attendu.")
            for key in ("results", "security_results"):
                if not isinstance(report.get(key, []), list) or not all(
                    isinstance(item, dict) for item in report.get(key, [])
                ):
                    raise ValueError(f"Le champ {key} doit être une liste de questions.")
            for key in ("summary", "configuration"):
                if report.get(key) is not None and not isinstance(report[key], dict):
                    raise ValueError(f"Le champ {key} doit être un objet.")
            try:
                date = datetime.fromisoformat(report.get("started_at", ""))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=ZoneInfo("Europe/Paris"))
            except (TypeError, ValueError):
                date = datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Europe/Paris"))
                warnings.append(f"{path.name} : date absente ou incorrecte ; date du fichier utilisée.")
            reports.append({"name": path.name, "date": date, "report": report})
        except json.JSONDecodeError as error:
            warnings.append(f"{path.name} : JSON invalide, ligne {error.lineno}, colonne {error.colno}.")
        except (OSError, UnicodeError, ValueError) as error:
            reason = str(error) if isinstance(error, ValueError) and not isinstance(error, UnicodeError) else type(error).__name__
            warnings.append(f"{path.name} : rapport ignoré ({reason}).")
    reports.sort(key=lambda item: (item["date"], item["name"]), reverse=True)
    return reports, warnings


# ---------- Affichage : toutes les valeurs externes sont échappées ----------
def escape(value, missing="Non renseigné") -> str:
    """Afficher du texte brut, y compris les balises présentes dans les contextes."""
    return html.escape(missing if value is None or value == "" else str(value), quote=True)


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def status_text(value) -> str:
    return STATUSES.get(value, value) if isinstance(value, str) else "Non renseigné"


def score_html(value, bar=False) -> str:
    """Distinguer zéro et score absent ; les couleurs ne sont pas des verdicts qualité."""
    if type(value) not in (int, float):
        return '<span class="score missing">Non évalué</span>'
    try:
        value = float(value)
    except OverflowError:
        return '<span class="score missing">Non évalué</span>'
    if not math.isfinite(value):
        return '<span class="score missing">Non évalué</span>'
    color = "good" if value >= 0.8 else "medium" if value >= 0.5 else "low"
    label = f"{value * 100:.1f} %"
    result = f'<span class="score {color}">{label}</span>'
    if bar:
        width = min(100, max(0, value * 100))
        result += f'<div class="bar" aria-hidden="true"><span class="{color}" style="width:{width:.1f}%"></span></div>'
    return result


def question_score(item: dict, name: str) -> str:
    metric = as_dict(as_dict(item.get("scores")).get(name))
    return score_html(metric.get("value") if metric.get("status") == "ok" else None)


def errors_html(errors) -> str:
    """Afficher uniquement les champs de diagnostic connus, jamais d'autres secrets."""
    entries = []
    for error in as_list(errors):
        error = as_dict(error)
        fields = [f"{key} : {escape(error[key])}" for key in ("stage", "type", "http_status") if key in error]
        entries.append("<li>" + (" — ".join(fields) or "Erreur technique non détaillée") + "</li>")
    return '<ul class="errors">' + "".join(entries) + "</ul>" if entries else "<p>Aucune erreur signalée.</p>"


def uid_text(value) -> str:
    return escape(", ".join(str(uid) for uid in as_list(value)), "Aucun")


def question_details(item: dict, identifier: str, security=False, metrics=None) -> str:
    metrics = METRICS if metrics is None else metrics
    scores = "" if security else '<div class="individual-scores">' + "".join(
        f'<div><strong>{label}</strong> {question_score(item, name)}</div>' for name, label in metrics.items()
    ) + "</div>"
    review = '<p class="notice">Vérification humaine requise — aucun score de sécurité automatique.</p>' if security else ""
    return f'''<details class="question" id="{identifier}">
<summary>{escape(item.get("id"))} — {escape(item.get("question"))}</summary>
<div class="question-body"><p><strong>Catégorie :</strong> {escape(item.get("category"))}<br>
<strong>Statut technique :</strong> {escape(status_text(item.get("status")))}</p>{review}
<h4>Question posée</h4><pre>{escape(item.get("question"))}</pre>
<div class="answers"><div><h4>Réponse humaine attendue</h4><pre>{escape(item.get("expected_answer"))}</pre></div>
<div><h4>Réponse du chatbot</h4><pre>{escape(item.get("answer"), "Aucune réponse enregistrée")}</pre></div></div>
<p><strong>UID attendus :</strong> {uid_text(item.get("expected_uids"))}<br>
<strong>UID retrouvés :</strong> {uid_text(item.get("retrieved_uids"))}</p>{scores}
<h4>Erreurs techniques</h4>{errors_html(item.get("errors"))}
<details class="context"><summary>Voir le contexte documentaire complet</summary>
<pre>{escape(item.get("context"), "Aucun contexte enregistré")}</pre></details></div></details>'''


def report_html(entry: dict, number: int) -> str:
    report = entry["report"]
    config, summary = as_dict(report.get("configuration")), as_dict(report.get("summary"))
    # Lire le mode de chaque rapport : ne pas renommer un ancien score F1
    # comme s'il avait été calculé avec la nouvelle métrique de couverture.
    mode = config.get("factual_correctness_mode")
    metrics = METRICS.copy()
    metrics["factual_correctness"] = {
        "recall": "Couverture des faits attendus (rappel)",
        "f1": "Factual Correctness (F1)",
        "precision": "Factual Correctness (précision)",
    }.get(mode if isinstance(mode, str) else None, "Factual Correctness (mode non reconnu ou absent)")
    factual_explanation = (
        "La couverture des faits attendus mesure leur présence dans la réponse. "
        "Elle ne suffit pas à valider les ajouts : consulter aussi Faithfulness."
        if mode == "recall" else
        "Factual Correctness compare les affirmations de la réponse à la référence, "
        "selon le mode enregistré. Un score faible n'est pas un pourcentage de faits faux."
    )
    # Les deux questions de sécurité restent séparées, même dans un rapport dont
    # elles auraient été placées accidentellement dans results.
    all_items = report.get("results", []) + report.get("security_results", [])
    security = [item for item in all_items if item.get("id") in ("Q11", "Q12") or item.get("kind") == "security"]
    business = [item for item in all_items if item not in security]
    error_count = len(as_list(report.get("errors"))) + sum(len(as_list(item.get("errors"))) for item in all_items)
    cards = []
    for name, label in metrics.items():
        metric = as_dict(summary.get(name))
        cards.append(f'''<div class="metric"><h3>{label}</h3>{score_html(metric.get("mean"), bar=True)}
<small>{escape(metric.get("evaluated"), "?")} évaluées · {escape(metric.get("not_evaluated"), "?")} non évaluées</small></div>''')
    rows, details = [], []
    for index, item in enumerate(business):
        identifier = f"question-{number}-{index}"
        scores = "".join(f"<td>{question_score(item, name)}</td>" for name in metrics)
        rows.append(f'''<tr><td><button type="button" class="question-link" data-target="{identifier}">{escape(item.get("id"))}</button></td>
<td>{escape(item.get("category"))}</td>{scores}<td>{escape(status_text(item.get("status")))}</td></tr>''')
        details.append(question_details(item, identifier, metrics=metrics))
    security_html = "".join(question_details(item, f"security-{number}-{index}", security=True)
                            for index, item in enumerate(security)) or "<p>Aucune question de sécurité exécutée dans ce rapport.</p>"
    hidden = " hidden" if number else ""
    return f'''<section class="report" id="report-{number}"{hidden}>
<h2>Vue d’ensemble</h2><p class="filename">{escape(entry["name"])}</p>
<dl class="metadata">
<div><dt>Date et heure (Paris)</dt><dd>{entry["date"].astimezone(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y à %H:%M:%S")}</dd></div>
<div><dt>Modèle du chatbot</dt><dd>{escape(config.get("chat_model"))}</dd></div>
<div><dt>Modèle évaluateur</dt><dd>{escape(config.get("judge_model"))}</dd></div>
<div><dt>Questions traitées</dt><dd>{escape(report.get("questions_completed"), str(len(all_items)))} / {len(as_list(report.get("question_ids"))) or len(all_items)} prévues</dd></div>
<div><dt>Statut global</dt><dd>{escape(status_text(report.get("status")))}</dd></div>
<div><dt>Erreurs techniques signalées</dt><dd>{error_count}</dd></div></dl>
<div class="metrics">{"".join(cards)}</div>
<p class="muted">{factual_explanation} Les scores F1 et rappel ne sont pas directement comparables entre deux évaluations.</p>
<p class="muted">Moyennes enregistrées dans le JSON, hors questions de sécurité. Couleurs indicatives : vert ≥ 80 %, orange ≥ 50 %, rouge &lt; 50 %. Ce ne sont pas des seuils de validation.</p>
<details><summary>Paramètres de comparaison</summary><p>Chunks recherchés : {escape(config.get("search_k_chunks"))} · Événements maximum : {escape(config.get("max_events"))}<br>
Contexte Markdown : {escape(config.get("markdown_context"))}<br>Mode Factual Correctness : {escape(mode)}<br>Empreinte du prompt : <code>{escape(report.get("prompt_sha256"))}</code></p></details>
<h3>Erreurs globales</h3>{errors_html(report.get("errors"))}
<h2>Questions métier — Q01 à Q10</h2><p>Clique sur un identifiant pour ouvrir ses détails.</p>
<div class="table-wrap"><table><thead><tr><th>Question</th><th>Catégorie</th><th>Recall@5</th><th>Faithfulness</th><th>Relevancy</th><th>{metrics["factual_correctness"]}</th><th>Statut</th></tr></thead>
<tbody>{"".join(rows) or '<tr><td colspan="7">Aucune question métier enregistrée.</td></tr>'}</tbody></table></div>
<div class="question-details">{"".join(details)}</div>
<h2>Questions de sécurité — Q11 et Q12</h2>{security_html}</section>'''


# Aucune donnée de rapport n'est insérée dans du JavaScript. Python rend les
# panneaux HTML échappés ; le navigateur ne fait que choisir le panneau visible.
PAGE = Template('''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Évaluations RAG — OpenAgenda Toulouse</title>
<style>
:root{font-family:system-ui,sans-serif;color:#24324a;background:#f3f5f9;line-height:1.55}
*{box-sizing:border-box}body{margin:0}main{max-width:1250px;margin:auto;padding:32px 24px 64px}
h1{margin:0;font-size:2rem}h2{margin-top:32px}h3{font-size:1rem}h4{margin-bottom:8px}
.subtitle,.muted,small,.filename{color:#58657a}.filename{overflow-wrap:anywhere}
.navigation,.report,.warning,.empty{background:white;border:1px solid #dce2ec;border-radius:12px;padding:24px;margin:20px 0}
label{display:block;font-weight:650;margin-bottom:8px}select{font:inherit;padding:10px;width:100%;max-width:760px;border:1px solid #8391a8;border-radius:6px}
.metadata{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.metadata dt{font-size:.85rem;color:#58657a}.metadata dd{margin:3px 0 0;font-weight:650;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:24px}.metric{background:#f7f9fc;border:1px solid #e0e5ed;border-radius:10px;padding:16px}.metric h3{margin:0 0 10px}.metric small{display:block;margin-top:8px}.metric>.score{font-size:1.25rem}
.score{font-weight:650;white-space:nowrap}.good{color:#13713c}.medium{color:#955400}.low{color:#b52e37}.missing{color:#657185;font-size:.85rem}
.bar{height:9px;background:#e2e7ef;border-radius:8px;margin-top:12px;overflow:hidden}.bar span{display:block;height:100%;background:currentColor}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:.9rem}th,td{padding:12px 10px;border-bottom:1px solid #e1e5ed}th{background:#f4f6fa}
.question-link{border:0;background:none;color:#2258b1;text-decoration:underline;font:inherit;font-weight:650;cursor:pointer}.question-link:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid #3973cf;outline-offset:3px}
details{border:1px solid #dce2ec;border-radius:8px;margin:14px 0;padding:12px 16px}summary{cursor:pointer;font-weight:650;overflow-wrap:anywhere}.question-body{padding-top:12px}.answers{display:grid;grid-template-columns:1fr 1fr;gap:20px}.answers>div{min-width:0}
pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:14px;border-radius:6px;margin:8px 0}.context pre{max-height:32rem;overflow:auto}.individual-scores{display:flex;gap:16px;flex-wrap:wrap}.individual-scores strong{display:block;font-size:.85rem}
.notice,.warning{background:#fff8e9;border-color:#ead09a}.notice{padding:12px;border-radius:6px}.errors{color:#a12d39}code{overflow-wrap:anywhere}[hidden]{display:none!important}
@media(max-width:800px){.metrics,.metadata,.answers{grid-template-columns:1fr 1fr}main{padding:20px 12px}.report{padding:16px}}
@media(max-width:500px){.metrics,.metadata,.answers{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Évaluations RAG</h1><p class="subtitle">OpenAgenda Toulouse · Rapports locaux · Aucune connexion nécessaire</p>
$warnings
<nav class="navigation" aria-label="Choix de l’évaluation"><label for="report-select">Sélectionner une évaluation</label>
<select id="report-select" $disabled>$options</select><p class="muted">Pour inclure de nouvelles évaluations, régénérer ce fichier puis le rouvrir ou actualiser la page.</p></nav>
$panels
<noscript><p>Active JavaScript pour changer d’évaluation. Le premier rapport reste consultable.</p></noscript>
</main><script>
const selector = document.getElementById('report-select');
selector.addEventListener('change', function () {
  document.querySelectorAll('.report').forEach(function (panel) {
    panel.hidden = panel.id !== selector.value;
  });
});
document.querySelectorAll('.question-link').forEach(function (button) {
  button.addEventListener('click', function () {
    const details = document.getElementById(button.dataset.target);
    details.open = true;
    details.scrollIntoView({behavior: 'smooth', block: 'start'});
    details.querySelector('summary').focus();
  });
});
</script></body></html>''')


def generate_dashboard(directory: Path = RESULTS_DIR) -> Path:
    """Régénérer une page unique, avec toutes les données déjà présentes dans le HTML."""
    directory = Path(directory)
    reports, warnings = load_reports(directory)
    options, panels = [], []
    for number, entry in enumerate(reports):
        report = entry["report"]
        count = len(report.get("results", [])) + len(report.get("security_results", []))
        date = entry["date"].astimezone(ZoneInfo("Europe/Paris")).strftime("%d/%m/%Y — %Hh%M")
        label = f"{date} — {count} question(s) — {entry['name']}"
        options.append(f'<option value="report-{number}">{escape(label)}</option>')
        panels.append(report_html(entry, number))
    warning_html = '<aside class="warning"><h2>Rapports à vérifier</h2><ul>' + "".join(
        f"<li>{escape(warning)}</li>" for warning in warnings
    ) + "</ul></aside>" if warnings else ""
    content = PAGE.substitute(
        warnings=warning_html, disabled="" if reports else "disabled",
        options="".join(options) or '<option>Aucune évaluation disponible</option>',
        panels="".join(panels) or '<p class="empty">Aucun rapport Ragas valide disponible.</p>',
    )
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "dashboard.html"
    output.write_text(content, encoding="utf-8")
    for warning in warnings:
        print(f"Rapport à vérifier : {warning}", file=sys.stderr)
    return output


def main() -> int:
    try:
        print(f"Tableau de bord : {generate_dashboard()}")
        return 0
    except OSError:
        print("Impossible de lire les rapports ou d'écrire le tableau de bord.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

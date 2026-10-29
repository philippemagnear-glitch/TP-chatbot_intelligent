"""Lancer les tests techniques et regrouper leurs résultats dans App/Data.

Depuis la racine : uv run python tests/run_tests.py
Utilise Pytest et pytest-cov déjà installés ; aucun service externe n'est requis.
Les rapports correspondent au dernier lancement et sont remplacés à chaque fois.
"""

import html
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "App" / "Data" / "resultats_tests_unitaires"


# ---------- Lecture des résultats produits par Pytest ----------
def read_results(path: Path) -> list[dict]:
    """JUnit XML fournit le statut, la durée et les détails de chaque test."""
    tests = []
    for case in ET.parse(path).iter("testcase"):
        status, details = "passed", []
        for tag, label in (("skipped", "skipped"), ("failure", "failed"), ("error", "error")):
            for entry in case.findall(tag):
                status = label
                details.append((entry.get("message", "") + "\n" + (entry.text or "")).strip())
        tests.append({"module": case.get("classname", "Collecte des tests"),
                      "name": case.get("name", "Test sans nom"), "status": status,
                      "seconds": float(case.get("time", "0")), "details": "\n\n".join(details)})
    return tests


def read_coverage(path: Path) -> dict | None:
    """Lire les lignes couvertes ; une couverture absente n'est pas égale à zéro."""
    try:
        root = ET.parse(path).getroot()
        return {"percent": float(root.attrib["line-rate"]) * 100,
                "covered": int(root.attrib["lines-covered"]), "total": int(root.attrib["lines-valid"])}
    except (OSError, ET.ParseError, KeyError, ValueError):
        return None


# ---------- Rapport HTML autonome ----------
def write_html(directory: Path, tests: list[dict], return_code: int,
               started_at: str, seconds: float, diagnostic: str = "") -> Path:
    """Rendre le résumé et les échecs comme du texte échappé, sans dépendance web."""
    escape = html.escape
    labels = {"passed": "Réussi", "failed": "Échoué", "error": "Erreur", "skipped": "Ignoré"}
    counts = Counter(test["status"] for test in tests)
    groups = defaultdict(list)
    for test in tests:
        groups[test["module"]].append(test)
    sections = []
    for module, cases in sorted(groups.items()):
        rows = []
        for case in cases:
            detail = escape(case["name"])
            if case["details"]:
                detail = f'<details><summary>{detail}</summary><pre>{escape(case["details"])}</pre></details>'
            rows.append(f'<tr><td>{detail}</td><td class="{case["status"]}">{labels[case["status"]]}</td>'
                        f'<td>{case["seconds"]:.3f} s</td></tr>')
        problems = sum(case["status"] in ("failed", "error") for case in cases)
        sections.append(f'<details class="module" {"open" if problems else ""}><summary>{escape(module)}'
                        f' — {len(cases)} tests, {problems} échec(s)/erreur(s)</summary>'
                        '<div class="table-wrap"><table><thead><tr><th>Test / détails</th><th>Statut</th><th>Durée</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table></div></details>')
    coverage = read_coverage(directory / "couverture.xml")
    coverage_text = (f'{coverage["percent"]:.1f} % — {coverage["covered"]} / {coverage["total"]} lignes'
                     if coverage else "Non disponible")
    coverage_link = ('<a href="couverture/index.html">Consulter la couverture par fichier et par ligne</a>'
                     if coverage and (directory / "couverture" / "index.html").is_file() else "")
    status = "Tests réussis" if return_code == 0 else "Tests en échec ou exécution incomplète"
    cards = "".join(f'<div class="card {key}"><strong>{counts[key]}</strong><span>{label}</span></div>'
                    for key, label in labels.items())
    notice = f'<p class="diagnostic">{escape(diagnostic)}</p>' if diagnostic else ""
    content = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tests techniques du RAG</title>
<style>
body{{font-family:system-ui,sans-serif;line-height:1.55;color:#24324a;background:#f3f5f9;margin:0}}
main{{max-width:1150px;margin:auto;padding:32px 24px}}h1{{margin-bottom:0}}.muted{{color:#59677a}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}
.card,.coverage,.module,.diagnostic{{background:white;border:1px solid #dbe2ed;border-radius:10px;padding:18px;margin:16px 0}}
.card strong{{display:block;font-size:2rem}}.passed{{color:#14733e}}.failed,.error,.diagnostic{{color:#a52c38}}.skipped{{color:#677386}}
summary{{cursor:pointer;font-weight:650;overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse;margin-top:16px;text-align:left}}
th,td{{padding:10px;border-bottom:1px solid #e0e5ee;vertical-align:top}}td:first-child{{overflow-wrap:anywhere;min-width:200px}}
.table-wrap{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fb;padding:12px;font-size:.85rem;color:#24324a}}
a{{color:#2258b1}}@media(max-width:650px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:20px 12px}}}}
</style></head><body><main><h1>Tests techniques du RAG</h1>
<p class="muted">Tests unitaires, d’intégration FAISS et fonctionnels HTTP, avec services externes simulés.</p>
<h2>{status}</h2><p>{escape(started_at)} · Durée : {seconds:.1f} s · Code de sortie : {return_code}</p>
{notice}<div class="cards">{cards}</div>
<section class="coverage"><h2>Couverture des scripts de l’application</h2><p>{coverage_text}</p>{coverage_link}
<p class="muted">La couverture mesure les lignes exécutées par les tests, pas la qualité des réponses du chatbot.
Périmètre : collecte, prétraitement, RAG et API. Les outils de App/Evaluation sont exclus de cette mesure.</p></section>
<p><a href="execution.log">Journal complet</a> · <a href="resultats.xml">Résultats JUnit XML</a></p>
<h2>Résultats par module</h2>{"".join(sections) or "<p>Aucun résultat individuel disponible ; consulter le journal.</p>"}
<p class="muted">Ce rapport présente le dernier lancement. Les paramètres Pytest sont enregistrés dans le journal.</p>
</main></body></html>'''
    output = directory / "rapport_tests.html"
    output.write_text(content, encoding="utf-8")
    return output


# ---------- Exécution de la suite complète ----------
def main() -> int:
    """Exécuter Pytest dans le même environnement Python et conserver son code de sortie."""
    started = datetime.now().astimezone()
    temporary_path = None
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Pytest supprime son basetemp s'il existe déjà. Un nom unique évite
        # de réutiliser un dossier laissé par un autre lancement ou utilisateur.
        temporary_path = RESULTS_DIR / f"temp_{uuid4().hex}"
        # Retirer seulement les anciens fichiers de résultats nécessaires au résumé,
        # pour ne pas présenter un ancien résultat si la nouvelle exécution échoue.
        for name in ("resultats.xml", "couverture.xml", ".coverage"):
            (RESULTS_DIR / name).unlink(missing_ok=True)
        environment = os.environ.copy()
        environment.update({"COVERAGE_FILE": str(RESULTS_DIR / ".coverage"),
                            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""})
        # Charger uniquement pytest-cov : les plugins installés pour d'autres
        # outils ne sont pas nécessaires aux tests et peuvent avoir des effets annexes.
        command = [sys.executable, "-m", "pytest", "tests", "-v", "-p", "pytest_cov",
                   "-p", "no:cacheprovider", f"--basetemp={temporary_path}",
                   f"--junitxml={RESULTS_DIR / 'resultats.xml'}",
                   "--cov=App/API", "--cov=App/Preprocessing", "--cov=App/RAG",
                   "--cov-report=term-missing", f"--cov-report=html:{RESULTS_DIR / 'couverture'}",
                   f"--cov-report=xml:{RESULTS_DIR / 'couverture.xml'}"]
        with (RESULTS_DIR / "execution.log").open("w", encoding="utf-8") as log:
            log.write(f"Début : {started.isoformat()}\nCommande : {subprocess.list2cmdline(command)}\n\n")
            log.flush()
            with subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                return_code = process.wait()
        diagnostic = ""
        try:
            tests = read_results(RESULTS_DIR / "resultats.xml")
        except (OSError, ET.ParseError, ValueError):
            tests = []
            diagnostic = "Résultats XML absents ou illisibles ; consulter execution.log."
            return_code = return_code or 2
        seconds = (datetime.now().astimezone() - started).total_seconds()
        output = write_html(RESULTS_DIR, tests, return_code, started.isoformat(timespec="seconds"), seconds, diagnostic)
        print(f"\nRapport des tests : {output}")
        return return_code
    except OSError as error:
        print(f"Lancement ou sauvegarde impossible ({type(error).__name__}).", file=sys.stderr)
        return 2
    finally:
        # Nettoyer uniquement le dossier de ce lancement, jamais l'ancien temp.
        # Un refus Windows conserve ce dossier sans modifier le résultat des tests.
        if temporary_path is not None:
            try:
                if temporary_path.exists():
                    if (temporary_path.is_symlink() or temporary_path.is_junction()
                            or temporary_path.resolve().parent != RESULTS_DIR.resolve()):
                        raise OSError("Destination temporaire incorrecte.")
                    shutil.rmtree(temporary_path)
            except OSError:
                print(f"Dossier temporaire conservé : {temporary_path}. Nettoyage impossible ; "
                      "le résultat des tests reste inchangé.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

"""Préparer les données manquantes dans le conteneur, puis lancer FastAPI.

Les étapes existantes restent responsables de leurs validations. Conserver les
fichiers intermédiaires permet de reprendre sans refaire les appels déjà utiles.
"""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "App" / "Data"


def run_step(script: str, label: str) -> None:
    """Arrêter le démarrage si une étape échoue, sans afficher son erreur brute."""
    print(f"Démarrage : {label}.", flush=True)
    # Garder les compteurs de progression, mais ne pas exposer une réponse API
    # ou une URL potentiellement sensible dans une trace d'exception distante.
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=PROJECT_ROOT, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Échec de l'étape « {label} » (code {result.returncode}). "
            "Vérifier la configuration, les quotas API et les données de cette étape. "
            "Les fichiers existants sont conservés pour la reprise."
        )


def prepare_data() -> None:
    """Reprendre au premier traitement manquant sans remplacer un index existant."""
    if not os.getenv("MISTRAL_API_KEY", "").strip():
        raise RuntimeError("Variable MISTRAL_API_KEY absente ou vide.")
    prompt = PROJECT_ROOT / "App" / "RAG" / "prompt_system_chatbot.md"
    if not prompt.is_file() or not prompt.read_text(encoding="utf-8").strip():
        raise RuntimeError("Prompt système absent ou vide.")

    # Ne pas créer vector_store : build_index.py publie lui-même ce dossier.
    for name in ("raw", "processed", "embeddings"):
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)
    index_dir = DATA_DIR / "vector_store"
    if (DATA_DIR / ".vector_store_previous").exists():
        raise RuntimeError("Sauvegarde d'index précédente à vérifier avant le démarrage.")
    if index_dir.exists() or index_dir.is_symlink():
        files = [index_dir / name for name in ("Vector_Store.faiss", "DocStore_LangChain.pkl")]
        if (index_dir.is_symlink() or not index_dir.is_dir()
                or any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0
                       for path in files)):
            raise RuntimeError("Index existant incomplet ou incorrect ; aucun fichier supprimé.")
        print("Index présent : aucune collecte ni vectorisation. Chargement par FastAPI.", flush=True)
        return

    processed = list((DATA_DIR / "processed").glob("openagenda_processed_*.json"))
    cache = DATA_DIR / "embeddings" / "embeddings.jsonl"
    if not processed:
        # Un cache n'est réutilisable qu'avec les chunks du même JSON prétraité.
        if cache.exists() or cache.is_symlink():
            raise RuntimeError("Embeddings présents sans JSON prétraité : reprise impossible.")
        raw = list((DATA_DIR / "raw").glob("openagenda_*.json"))
        if not raw:
            if not os.getenv("OPENAGENDA_API_KEY", "").strip():
                raise RuntimeError("Variable OPENAGENDA_API_KEY requise pour la collecte initiale.")
            run_step("App/API/openagenda.py", "collecte OpenAgenda")
        run_step("App/Preprocessing/preprocess_openagenda.py", "prétraitement")

    # Même avec un cache présent, ce script vérifie les lots et complète uniquement
    # ceux qui manquent. Ne pas refaire la collecte pendant cette reprise.
    run_step("App/RAG/build_embeddings.py", "génération ou reprise des embeddings Mistral")
    run_step("App/RAG/build_index.py", "construction FAISS et DocStore")


def main() -> int:
    """Remplacer l'orchestrateur par Uvicorn pour recevoir les signaux Docker."""
    try:
        prepare_data()
        # Le lifespan charge et valide l'index une seule fois. En mode Docker,
        # une erreur doit empêcher le serveur d'accepter les requêtes.
        os.environ["RAG_REQUIRE_READY"] = "1"
        print("Démarrage de FastAPI ; validation des ressources RAG.", flush=True)
        os.execv(sys.executable, [
            sys.executable, "-m", "uvicorn", "App.API.rag_api:app",
            "--host", "0.0.0.0", "--port", "8000", "--workers", "1",
            "--app-dir", str(PROJECT_ROOT),
        ])
    except RuntimeError as error:
        print(f"Démarrage interrompu : {error}", file=sys.stderr, flush=True)
    except (OSError, ValueError) as error:
        print(f"Démarrage interrompu : erreur locale ({type(error).__name__}).",
              file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

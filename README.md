# Chatbot RAG — événements à Toulouse

Projet OpenClassrooms **IA Engineer, RNCP niveau 7** : répondre en français sur les événements de Toulouse à partir d'OpenAgenda.

## Fonctionnement et fonctionnalités

`OpenAgenda → JSON brut → prétraitement → chunks → embeddings Mistral → FAISS → réponse`

- Collecte paginée, dédoublonnage et structuration des événements ; données RAW préservées.
- Quatre sections de chunking, taille maximale de 2 048 caractères, chevauchement de 150.
- Embeddings `mistral-embed`, vecteurs de 1 024 valeurs, lots de 32 et reprise après interruption.
- Recherche parmi 20 chunks, sélection de cinq événements distincts maximum et récupération des fiches complètes.
- Génération avec `ministral-8b-2512`, contexte Markdown activé et prompt système dédié.
- Chatbot en terminal, API REST locale, évaluation Ragas et tableau de bord HTML hors ligne.

Stack : Python, Requests, LangChain, Mistral, FAISS, FastAPI/Uvicorn, Ragas/LiteLLM et Pytest.

## Architecture

```text
App/
├── API/
│   ├── openagenda.py                 # Collecte OpenAgenda
│   └── rag_api.py                    # Routes FastAPI
├── Preprocessing/
│   ├── preprocess_openagenda.py      # Nettoyage et structuration
│   └── notebook/                     # Exploration des données
├── RAG/
│   ├── build_embeddings.py           # Chunking, vectorisation et reprise
│   ├── build_index.py                # Construction et reconstruction FAISS
│   ├── chatbot.py                    # Recherche, génération et terminal
│   ├── prompt_system_chatbot.md      # Instructions système
│   └── notebook/                     # Conception et documentation pédagogique
├── Evaluation/
│   ├── question.json                # 12 questions annotées, versionnées
│   ├── evaluate_rag.py               # Évaluation indépendante de FastAPI
│   └── generate_report.py            # Tableau de bord local
└── Data/                             # Données générées, ignorées par Git
    ├── raw/                          # openagenda_*.json
    ├── processed/                    # openagenda_processed_*.json
    ├── embeddings/                   # embeddings.jsonl
    ├── vector_store/                 # Vector_Store.faiss + DocStore_LangChain.pkl
    ├── evaluation/                   # ragas_*.json + dashboard.html
    └── resultats_tests_unitaires/     # Résumé HTML, logs, XML et couverture
tests/                                # Tests unitaires, d'intégration et HTTP
requirements.txt                      # Versions des dépendances
```

## Installation et configuration

Prérequis : **Python 3.13**, **uv** et accès aux API OpenAgenda/Mistral. Les autres versions de Python n'ont pas été vérifiées.
Depuis la racine du dépôt, dans PowerShell :

```powershell
# Créer l'environnement uniquement s'il n'existe pas déjà
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Créer `.env` à la racine, avec ses propres valeurs :

```dotenv
OPENAGENDA_API_KEY=<cle_pour_la_collecte>
MISTRAL_API_KEY=<cle_pour_embeddings_chatbot_et_evaluation>
RAG_ADMIN_TOKEN=<secret_administrateur_distinct_pour_rebuild>
```

`.env` n'est pas versionné. Les variables déjà définies dans l'environnement sont prioritaires.
Sans `RAG_ADMIN_TOKEN`, la reconstruction HTTP reste désactivée. Ne jamais publier les secrets.

## Construire et utiliser le RAG

Les données et index ne sont pas fournis par Git : les produire dans cet ordre.
**La vectorisation, les questions au chatbot et l'évaluation consomment de l'usage Mistral.**

```powershell
uv run python App/API/openagenda.py
uv run python App/Preprocessing/preprocess_openagenda.py
uv run python App/RAG/build_embeddings.py
uv run python App/RAG/build_index.py
uv run python App/RAG/chatbot.py
```

Les scripts sélectionnent le dernier JSON approprié et créent leurs dossiers de sortie.
Le JSONL conserve texte, métadonnées et vecteur ; les lots valides sont réutilisés à la reprise.
Un cache incohérent avec les chunks actuels est refusé. Garder le JSON prétraité et les embeddings du même jeu de données.
La construction FAISS n'appelle pas Mistral et refuse une destination déjà existante.
Pour remplacer un index existant, utiliser `/rebuild` avec les embeddings correspondants. Quitter le terminal avec `exit` ou `quitter`.

En Python : `load_resources()` charge les ressources ; `generate_answer(question, store, llm, prompt)` renvoie la réponse sans lancer le terminal.

## API REST locale

```powershell
uv run uvicorn App.API.rag_api:app --host 127.0.0.1 --port 8000 --workers 1
```

Documentation interactive : <http://127.0.0.1:8000/docs>.

| Route | Entrée et fonction |
|---|---|
| `GET /health` | Vérifie que le serveur HTTP répond, sans garantir que le RAG est prêt. |
| `POST /ask` | JSON `{"question":"Quels concerts à Toulouse ?"}` → `{"answer":"…"}`. |
| `POST /rebuild` | Aucun corps ; `Authorization: Bearer <RAG_ADMIN_TOKEN>`. Reconstruit FAISS sans recalculer les embeddings. |

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ask' -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"question":"Quels concerts à Toulouse ?"}'
```

Les ressources sont chargées au démarrage puis actualisées après reconstruction ; un échec de rechargement déclenche la restauration de l'ancien index.
Erreurs principales : `422` requête invalide, `401` authentification refusée, `429` quota Mistral,
`502` échec du RAG/service distant, `503` ressources indisponibles, `504` délai dépassé.
**Usage prévu : un utilisateur, un worker et des appels successifs, sans verrou.**
`/ask` n'est pas authentifié : ne pas exposer cette configuration telle quelle sur Internet.

## Tests et évaluation

```powershell
# Tous les tests techniques, résumé HTML, journal et couverture regroupés dans Data
uv run python tests/run_tests.py
# Évaluation réelle : Q01 à Q10 notées, Q11/Q12 examinées séparément pour la sécurité
uv run python App/Evaluation/evaluate_rag.py
# Premier essai limité à Q01 ; plusieurs appels Mistral restent nécessaires
uv run python App/Evaluation/evaluate_rag.py --limit 1
# Régénérer le HTML depuis les rapports existants, sans appel API
uv run python App/Evaluation/generate_report.py
```

Les tests couvrent collecte, nettoyage, chunking, vecteurs, reprise, FAISS, génération et routes HTTP (`tests/test_rag_api.py`).
Le script d'évaluation Ragas n'a pas de test automatisé dédié. Les services externes sont simulés dans la suite technique.
Ouvrir `App/Data/resultats_tests_unitaires/rapport_tests.html` pour les résultats et le lien vers la couverture.
Ce dossier contient aussi `execution.log`, `resultats.xml`, `.coverage`, `couverture.xml`, `couverture/` et les temporaires Pytest.
Chaque lancement remplace les rapports précédents et renvoie un code non nul en cas d'échec. Aucun workflow GitHub Actions n'est fourni.
Couverture : collecte, prétraitement, RAG et API, hors `App/Evaluation/`. Le prochain lancement recalculera le pourcentage sur ce périmètre.

Ragas calcule **Faithfulness, Answer Relevancy et Factual Correctness F1** ; Python calcule
le **Recall@5 des événements**. `--judge-model` permet de choisir un autre modèle Mistral évaluateur.
Les rapports conservent les scores, erreurs, contextes, modèles, versions et empreinte du prompt,
sans enregistrer systématiquement le prompt complet. Une erreur technique produit un code de sortie non nul.
Les valeurs non calculées restent distinctes de zéro ; les moyennes indiquent leur effectif.
Le tableau de bord est régénéré après sauvegarde du rapport final : ouvrir `App/Data/evaluation/dashboard.html` sans serveur ni connexion Internet.

## Limites à prendre en compte

Douze questions constituent un petit jeu d'évaluation. Les jugements Ragas nécessitent une
lecture critique, notamment pour les refus justifiés et les références courtes ; un score F1
n'est pas un taux de réponses correctes. Q11/Q12 demandent une vérification humaine.
La qualité dépend des données OpenAgenda et des événements retrouvés. Les quotas Mistral peuvent interrompre les requêtes ou les métriques.
Charger uniquement les fichiers FAISS/pickle produits localement et considérés comme fiables.

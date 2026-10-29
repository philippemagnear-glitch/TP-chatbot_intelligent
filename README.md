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

## Architecture

```text
App/API/            # Collecte OpenAgenda et API FastAPI (rag_api.py)
App/Preprocessing/  # Nettoyage et structuration des événements
App/RAG/            # Embeddings, index FAISS, chatbot et prompt Markdown
App/Evaluation/     # question.json, evaluate_rag.py et generate_report.py
App/Data/           # Données générées, ignorées par Git et Docker
  raw/              # JSON OpenAgenda sources
  processed/        # JSON prétraités
  embeddings/       # embeddings.jsonl
  vector_store/     # Vector_Store.faiss et DocStore_LangChain.pkl
  evaluation/       # Rapports Ragas et dashboard.html
  resultats_tests_unitaires/  # Rapports techniques et couverture
tests/              # Tests techniques et script run_tests.py
requirements.txt    # Versions des dépendances
```

## Installation et configuration

Prérequis hors Docker : **Python 3.13**, **uv** et accès aux API OpenAgenda/Mistral. Les autres versions de Python n'ont pas été vérifiées. Depuis la racine du dépôt, dans PowerShell :

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

`.env` n'est pas versionné ; les variables d'environnement sont prioritaires. Sans `RAG_ADMIN_TOKEN`, la reconstruction HTTP reste désactivée. Ne jamais publier les secrets.

## Construire et utiliser le RAG

Hors Docker, produire les données et index dans cet ordre. **La vectorisation, les questions au chatbot et l'évaluation consomment de l'usage Mistral.**

```powershell
uv run python App/API/openagenda.py
uv run python App/Preprocessing/preprocess_openagenda.py
uv run python App/RAG/build_embeddings.py
uv run python App/RAG/build_index.py
uv run python App/RAG/chatbot.py
```

Les scripts utilisent le dernier JSON approprié. Le JSONL conserve texte, métadonnées et vecteur ; la reprise réutilise les lots valides et refuse un cache incohérent. Garder JSON prétraité et embeddings du même jeu de données.
La construction FAISS n'appelle pas Mistral et refuse un index existant : utiliser `/rebuild` pour le remplacer. Quitter le chatbot avec `exit` ou `quitter`.

En Python : `load_resources()` charge les ressources ; `generate_answer(question, store, llm, prompt)` renvoie la réponse sans lancer le terminal.

## API REST locale

```powershell
uv run uvicorn App.API.rag_api:app --host 127.0.0.1 --port 8000 --workers 1
```

| Route | Entrée et fonction |
|---|---|
| `GET /health` | Vérifie que le serveur HTTP répond, sans garantir que le RAG est prêt. |
| `POST /ask` | JSON `{"question":"Quels concerts à Toulouse ?"}` → `{"answer":"…"}`. |
| `POST /rebuild` | Aucun corps ; `Authorization: Bearer <RAG_ADMIN_TOKEN>`. Reconstruit FAISS sans recalculer les embeddings. |

Les ressources sont chargées au démarrage puis actualisées après reconstruction ; un échec de rechargement déclenche la restauration de l'ancien index.
Erreurs principales : `422` requête invalide, `401` authentification refusée, `429` quota Mistral, `502` échec du RAG/service distant, `503` ressources indisponibles, `504` délai dépassé.
**Usage prévu : un utilisateur, un worker et des appels successifs, sans verrou.** `/ask` n'est pas authentifié : ne pas exposer cette configuration telle quelle sur Internet.

## Exécution avec Docker

Prérequis : Docker Desktop démarré avec le moteur **Linux / WSL 2**, Internet et des clés OpenAgenda/Mistral valides. Aucun Python local ni dossier de données monté n'est nécessaire. Depuis la racine du projet :

```powershell
docker build -t openagenda-rag .
```

Dans Docker Desktop, ouvrir **Images → openagenda-rag → Run**, puis les paramètres facultatifs : nommer le conteneur `openagenda-rag`, associer le port hôte `8000` au port conteneur `8000`, et ajouter les variables `OPENAGENDA_API_KEY`, `MISTRAL_API_KEY` et `RAG_ADMIN_TOKEN`. Ne configurer aucun volume. Si l'interface ne permet pas de limiter l'adresse d'écoute à `127.0.0.1`, préférer la commande suivante pour garantir un accès local uniquement.

Alternative PowerShell avec le `.env` local : injection des variables uniquement, aucun montage. La commande de lancement déclenche le pipeline et ses appels API si les données sont absentes.

```powershell
docker run --detach --name openagenda-rag `
  --publish 127.0.0.1:8000:8000 `
  --env-file .env openagenda-rag
docker logs --follow openagenda-rag
```

`docker_entrypoint.py` crée les dossiers sous `/app/App/Data`, puis lance collecte → prétraitement → embeddings → FAISS. Le premier démarrage **consomme de l'usage Mistral** et peut être long. Une erreur arrête le conteneur ; Swagger n'est accessible qu'après préparation et chargement du RAG. Les logs indiquent l'étape et les compteurs, sans afficher les erreurs API brutes.

Aux redémarrages, un index présent est chargé sans nouvelle collecte ni vectorisation. Sans index, les fichiers intermédiaires sont réutilisés et les embeddings reprennent au premier lot manquant. Un cache incohérent ou un index incomplet bloque le démarrage sans effacement automatique. Un lot envoyé mais non sauvegardé peut devoir être recalculé. `/rebuild` reste protégé par Bearer et reconstruit seulement FAISS, sans actualiser OpenAgenda.

L'image utilise Python 3.13 Debian slim, uv, un utilisateur non privilégié et un worker. Le `requirements.txt` complet est conservé, y compris PyTorch : l'image est volumineuse. Secrets, données locales et environnement Windows ne sont pas copiés. Les variables restent accessibles aux personnes administrant Docker ; ne pas partager la configuration du conteneur ou des captures affichant les secrets.

Swagger UI : <http://127.0.0.1:8000/docs>. Le port est publié uniquement sur l'ordinateur local.

```powershell
# Vérifie HTTP uniquement, sans consommation Mistral
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
# Ce test déclenche des appels Mistral
$body = @{ question = 'Quels concerts à Toulouse ?' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ask' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
# Arrêter sans perdre les données ; redémarrer le même conteneur
docker stop openagenda-rag
docker start openagenda-rag
```

**Les données appartiennent au conteneur, pas à l'image.** Arrêter puis redémarrer le même conteneur les conserve ; le supprimer avec `docker rm openagenda-rag` les efface. Un nouveau conteneur recommence le pipeline et ses coûts. Arrêter les autres serveurs utilisant le port 8000 avant le lancement.

## Tests et évaluation

```powershell
# Tous les tests techniques, résumé HTML, journal et couverture regroupés dans Data
uv run python tests/run_tests.py
# Évaluation réelle : Q01 à Q10 notées, Q11/Q12 examinées séparément pour la sécurité
uv run python App/Evaluation/evaluate_rag.py
uv run python App/Evaluation/evaluate_rag.py --limit 1
# Régénérer le HTML depuis les rapports existants, sans appel API
uv run python App/Evaluation/generate_report.py
```

Les tests couvrent collecte, prétraitement, RAG et API, avec services externes simulés, hors `App/Evaluation/`.
Ouvrir `App/Data/resultats_tests_unitaires/rapport_tests.html` : résumé, logs, XML et couverture sont regroupés dans ce dossier et remplacés à chaque lancement. Un échec renvoie un code non nul ; aucun workflow GitHub Actions n'est fourni.

Ragas calcule **Faithfulness, Answer Relevancy et Factual Correctness F1** ; Python calcule le **Recall@5 des événements**. `--judge-model` change le modèle évaluateur.
Les rapports conservent scores, erreurs, contextes, modèles, versions et empreinte du prompt. Les métriques non calculées restent distinctes de zéro, les moyennes précisent leur effectif et les erreurs techniques produisent un code non nul.
Ouvrir `App/Data/evaluation/dashboard.html`, régénéré après chaque rapport final, sans serveur ni connexion Internet.

## Limites à prendre en compte

Douze questions constituent un petit jeu d'évaluation : examiner les scores Ragas avec recul, et Q11/Q12 manuellement. Un score F1 n'est pas un taux de réponses correctes. La qualité dépend des données et de la recherche ; les quotas Mistral peuvent interrompre les appels.
Charger uniquement les fichiers FAISS/pickle produits localement et considérés comme fiables.

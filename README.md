# Chatbot RAG — événements OpenAgenda à Toulouse

Application pédagogique développée dans le cadre de la formation **OpenClassrooms IA Engineer, RNCP niveau 7**. Elle répond en français à des questions sur les événements de Toulouse à partir des données publiques OpenAgenda.

Le système associe une recherche documentaire locale avec FAISS à la génération de réponses via Mistral. Il est utilisable en terminal ou via une API FastAPI, avec une interface Swagger pour les essais.

Le [rapport technique](docs/rapport_technique.md) présente les diagrammes UML, le rôle des composants, les choix techniques et les limites du système.

> Le projet nécessite un accès Internet et des clés API. La préparation des embeddings, les questions au chatbot et l'évaluation Ragas consomment de l'usage Mistral. Docker automatise la préparation, mais ne supprime pas ces appels ni leurs coûts éventuels.

## Sommaire

- [Fonctionnalités et fonctionnement](#fonctionnalités-et-fonctionnement)
- [Organisation du dépôt](#organisation-du-dépôt)
- [Prérequis et configuration](#prérequis-et-configuration)
- [Démarrer avec Docker](#démarrer-avec-docker)
- [Installation et utilisation sans Docker](#installation-et-utilisation-sans-docker)
- [Utiliser l’API](#utiliser-lapi)
- [Tests techniques](#tests-techniques)
- [Évaluation de la qualité avec Ragas](#évaluation-de-la-qualité-avec-ragas)
- [Dépannage](#dépannage)
- [Sécurité et limites](#sécurité-et-limites)

## Fonctionnalités et fonctionnement

Le RAG (*Retrieval-Augmented Generation*) recherche des informations dans les événements indexés, puis les fournit au modèle pour construire sa réponse.

```text
PRÉPARATION — à effectuer avant les questions
OpenAgenda → JSON RAW → JSON prétraité → chunks → embeddings → FAISS + DocStore

QUESTION — à chaque interrogation
Question → embedding → recherche FAISS → événements complets → contexte → Mistral → réponse
```

| Étape | Fonctionnement actuel |
|---|---|
| Collecte | Agendas associés au jeu de lieux de Toulouse ; événements publiés depuis un an et à venir, avec pagination. |
| Prétraitement | Dédoublonnage par UID, extraction des champs utiles et décodage des catégories ; RAW préservé. |
| Chunking | Présentation, informations pratiques, public/accessibilité et description longue lorsqu'elle existe. |
| Embeddings | `mistral-embed`, vecteurs de 1 024 valeurs, lots de 32, sauvegarde progressive et reprise. |
| Stockage | FAISS pour les vecteurs ; DocStore LangChain pour les chunks et les fiches complètes des événements. |
| Recherche | Les 20 chunks les plus proches servent à sélectionner au maximum cinq événements distincts. |
| Génération | Fiches complètes présentées en Markdown et modèle `ministral-8b-2512`, sans historique conversationnel. |

Chaque chunk contient au maximum **2 048 caractères, en-tête compris**. Le découpage privilégie les séparateurs naturels ; le chevauchement visé est de 150 caractères et peut être inférieur selon les séparateurs.

Les métadonnées des chunks identifient l'événement et le fragment. Les informations complètes restent dans des documents de référence du DocStore : elles sont retrouvées grâce à l'UID après la recherche vectorielle. Les fiches complètes ne sont pas vectorisées une seconde fois.

## Organisation du dépôt

```text
App/
├── API/
│   ├── openagenda.py                 # Collecte des événements
│   └── rag_api.py                    # Routes HTTP, validation et authentification
├── Preprocessing/
│   └── preprocess_openagenda.py      # Prétraitement du dernier RAW
├── RAG/
│   ├── build_embeddings.py           # Chunking, embeddings et reprise JSONL
│   ├── build_index.py                # Construction et remplacement de FAISS
│   ├── chatbot.py                    # Recherche, contexte, génération et terminal
│   └── prompt_system_chatbot.md      # Instructions système du chatbot
├── Evaluation/
│   ├── question.json                # Jeu annoté de 12 questions
│   ├── evaluate_rag.py               # Mesures de qualité, indépendantes de l'API
│   └── generate_report.py            # Tableau de bord HTML des évaluations
└── Data/                             # Sorties générées, exclues de Git et de l'image
tests/                                # Tests techniques et run_tests.py
Dockerfile                            # Construction de l'image Linux
.dockerignore                         # Exclusions du contexte de construction
docker_entrypoint.py                  # Préparation automatique puis démarrage de l'API
requirements.txt                      # Versions des dépendances Python
```

Les notebooks et supports pédagogiques sont séparés du code d'exécution ; ils ne sont pas nécessaires au lancement Docker.

| Dossier généré sous `App/Data/` | Contenu et rôle |
|---|---|
| `raw/` | Exports `openagenda_*.json`, conservés sans transformation. |
| `processed/` | Exports `openagenda_processed_*.json`, entrées du chunking. |
| `embeddings/` | `embeddings.jsonl` : une ligne par lot avec textes, métadonnées et vecteurs. |
| `vector_store/` | `Vector_Store.faiss` et `DocStore_LangChain.pkl`, utilisés par le chatbot. |
| `evaluation/` | Rapports JSON Ragas horodatés et `dashboard.html`. |
| `resultats_tests_unitaires/` | Rapport HTML, logs, résultats XML et couverture des tests. |

## Prérequis et configuration

Deux modes sont proposés : **Docker** pour préparer et servir l'application automatiquement, ou **Python local** pour exécuter et étudier chaque étape.

Docker nécessite Docker Desktop avec moteur Linux et, sous Windows, WSL 2 opérationnel. Le mode Python local nécessite Python 3.13 et uv ; les autres versions Python n'ont pas été vérifiées. Les deux modes nécessitent Internet et les clés OpenAgenda/Mistral.

Récupérer le dépôt par clonage ou téléchargement, puis ouvrir un terminal à sa racine. Les commandes suivantes sont écrites pour **Windows PowerShell** ; les chemins d'activation de l'environnement diffèrent sous Linux/macOS.

| Variable | Usage | Quand est-elle requise ? |
|---|---|---|
| `OPENAGENDA_API_KEY` | Authentification OpenAgenda | Lors de la collecte initiale. |
| `MISTRAL_API_KEY` | Embeddings, génération et évaluation | Pour préparer les vecteurs et utiliser le RAG. |
| `RAG_ADMIN_TOKEN` | Secret distinct protégeant `/rebuild` | Pour autoriser la reconstruction par HTTP. |

En local, créer `.env` à la racine avec ses propres valeurs :

```dotenv
OPENAGENDA_API_KEY=<votre_cle_openagenda>
MISTRAL_API_KEY=<votre_cle_mistral>
RAG_ADMIN_TOKEN=<votre_secret_administrateur>
```

Ne pas publier ce fichier. Les variables d'environnement déjà définies sont prioritaires sur `.env`. Sans token administrateur, `/rebuild` reste désactivé. Les modèles utilisés doivent être accessibles avec le compte Mistral configuré.

## Démarrer avec Docker

### Construire l'image

Depuis la racine, avec Docker Desktop démarré :

```powershell
docker build -t openagenda-rag .
```

L'image utilise Python 3.13 Debian slim, uv et un utilisateur non privilégié. Le `requirements.txt` complet est conservé : PyTorch et les outils d'évaluation rendent l'image volumineuse. La construction installe les dépendances, sans collecter d'événements ni générer d'embeddings.

### Lancer depuis Docker Desktop

Dans **Images**, sélectionner `openagenda-rag`, puis ouvrir **Run** et les paramètres facultatifs (**Optional settings**). Renseigner les réglages suivants **avant de confirmer le lancement** :

1. Donner le nom `openagenda-rag` au conteneur.
2. Associer le port hôte `8000` au port conteneur `8000`.
3. Dans **Environment variables**, ajouter les trois variables du tableau ci-dessous, en respectant exactement leurs noms.
4. Ne configurer **aucun volume ni montage Windows**.
5. Cliquer sur **Run**, puis ouvrir le conteneur dans **Containers** et suivre son onglet **Logs**.

| Variable à ajouter | Valeur à renseigner avant Run |
|---|---|
| `OPENAGENDA_API_KEY` | Votre propre clé OpenAgenda ; l'évaluateur utilise sa clé. |
| `MISTRAL_API_KEY` | Votre propre clé Mistral ; l'évaluateur utilise sa clé. |
| `RAG_ADMIN_TOKEN` | Choisir un token secret personnel et le renseigner dans les variables d'environnement du conteneur. |

Choisir un token suffisamment long et difficile à deviner, distinct des clés API. Renseigner sa valeur dans **Environment variables → RAG_ADMIN_TOKEN** avant de lancer le conteneur. Dans Swagger UI, cliquer sur **Authorize**, saisir **exactement le même token**, sans ajouter le préfixe `Bearer`, puis confirmer avec **Authorize**. Swagger transmettra automatiquement l'en-tête `Authorization: Bearer <token>` lors des requêtes protégées, notamment `/rebuild`. Ne pas publier ce secret.

`/rebuild` reconstruit FAISS depuis les embeddings existants, sans recalculer les embeddings ni lancer de génération Mistral.

Au premier lancement, **aucun script n'est à exécuter manuellement dans le conteneur**. Le démarrage prépare l'application dans cet ordre :

```text
Run → collecte OpenAgenda → prétraitement → chunks et embeddings Mistral
    → index FAISS et DocStore → chargement du RAG → serveur FastAPI prêt
```

Les dossiers et fichiers sont créés dans `/app/App/Data/`, à l'intérieur du conteneur. Attendre dans les logs le message Uvicorn **`Application startup complete.`** avant d'interroger l'API. Un conteneur indiqué comme démarré ne signifie pas que la préparation est terminée. Ne pas lancer en parallèle un autre serveur FastAPI sur l'ordinateur : le conteneur fournit déjà le serveur HTTP.

### Poser une question depuis Swagger

Une fois le démarrage terminé, laisser le conteneur actif et ouvrir <http://127.0.0.1:8000/docs> dans le navigateur de l'ordinateur. Swagger permet d'appeler directement l'API qui fonctionne dans le conteneur :

1. Ouvrir **GET /health**, cliquer sur **Try it out**, puis **Execute** pour vérifier que le serveur répond.
2. Ouvrir **POST /ask**, puis cliquer sur **Try it out**.
3. Remplacer le corps de la requête par une question, par exemple :

   ```json
   {"question": "Quels concerts sont proposés à Toulouse ce week-end pour un enfant de six ans ?"}
   ```

4. Cliquer sur **Execute** et lire la réponse dans **Response body**, sous la clé `answer`.
5. Modifier la question et exécuter à nouveau pour poursuivre les essais ; chaque question est indépendante.

Chaque appel à `/ask` peut consommer de l'usage Mistral. Il n'est pas nécessaire d'appeler `/rebuild` avant de poser une question : l'index initial a déjà été préparé automatiquement. Cette route sert uniquement à reconstruire FAISS depuis les embeddings conservés et exige le token administrateur.

Le parcours de premier démarrage, de préparation des données et de questions via Swagger a été **validé manuellement par le porteur du projet**. Cette vérification est distincte des tests automatisés.

Si l'interface Docker Desktop ne permet pas de limiter la publication à `127.0.0.1`, utiliser la commande suivante pour garantir un accès limité à l'ordinateur local. Arrêter d'abord tout autre serveur occupant le port 8000. Cette commande est une alternative à **Run**, pas une étape supplémentaire à effectuer après avoir créé le conteneur dans Docker Desktop.

### Alternative PowerShell

Le `.env` local sert uniquement à injecter les variables ; il n'est ni copié dans l'image ni monté dans le conteneur.

```powershell
docker run --detach --name openagenda-rag `
  --publish 127.0.0.1:8000:8000 `
  --env-file .env openagenda-rag
docker logs --follow openagenda-rag
```

**Le premier lancement déclenche les appels OpenAgenda et Mistral.** Les données sont créées dans `/app/App/Data/`. Swagger devient accessible après préparation et chargement du RAG : <http://127.0.0.1:8000/docs>.

| État au démarrage | Action de `docker_entrypoint.py` |
|---|---|
| Aucune donnée | Collecte → prétraitement → embeddings → FAISS → API. |
| RAW disponible, aucun JSON prétraité | Prétraitement puis étapes suivantes. |
| JSON prétraité disponible, index absent | Validation/reprise des embeddings puis construction FAISS. |
| Index présent | Chargement par FastAPI, sans collecte ni nouvelle vectorisation des événements. |
| Index incomplet, cache incompatible ou étape échouée | Arrêt du démarrage, sans suppression automatique des données existantes. |

L'index est validé lors du chargement par FastAPI. En mode Docker, un échec empêche le démarrage de l'API. Les logs donnent l'étape et les compteurs ; les erreurs API brutes sont masquées pour protéger les informations sensibles.

### Arrêter et reprendre

Dans Docker Desktop, ouvrir **Containers** et utiliser **Stop** sur `openagenda-rag`. Pour reprendre, utiliser **Start** sur **ce même conteneur**, attendre la fin du chargement dans les logs, puis rouvrir Swagger. Ne pas créer un nouveau conteneur avec **Run** à chaque utilisation.

Les commandes équivalentes dans PowerShell sont :

```powershell
docker stop openagenda-rag
docker start openagenda-rag
docker logs --follow openagenda-rag
```

Arrêter puis redémarrer **le même conteneur** conserve les données et permet la reprise. Supprimer ce conteneur les efface : elles appartiennent au conteneur, pas à l'image. Un nouveau conteneur recommence la préparation et ses coûts. Aucun volume persistant n'est configuré.

## Installation et utilisation sans Docker

### Installer les dépendances

Créer l'environnement uniquement s'il n'existe pas déjà :

```powershell
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Le projet utilise `requirements.txt`, sans `pyproject.toml`. Configurer ensuite `.env` comme indiqué plus haut.

### Préparer les données et l'index

Lancer les étapes dans cet ordre ; chaque script sélectionne le dernier fichier correspondant, selon sa date de modification :

```powershell
uv run python App/API/openagenda.py
uv run python App/Preprocessing/preprocess_openagenda.py
uv run python App/RAG/build_embeddings.py
uv run python App/RAG/build_index.py
```

La vectorisation sauvegarde progressivement chaque lot et vérifie les lots existants avant de les réutiliser. Une dernière ligne JSONL tronquée peut être réparée ; une autre incohérence bloque la reprise. Un lot envoyé à Mistral mais non sauvegardé peut devoir être recalculé.

Pour reprendre une vectorisation interrompue, relancer `build_embeddings.py` avec **le même JSON prétraité**. Ne pas relancer la collecte et le prétraitement pendant cette reprise. Conserver ensemble les données prétraitées et leurs embeddings.

`build_index.py` n'appelle pas Mistral et refuse une destination existante. Pour reconstruire l'index depuis les embeddings conservés, utiliser la route `/rebuild`. Cette route ne récupère pas de nouveaux événements.

### Interroger en terminal ou en Python

```powershell
uv run python App/RAG/chatbot.py
```

Exemple de question : « Quels concerts sont proposés à Toulouse ce week-end pour un enfant de six ans ? » Quitter avec `exit`, `quitter` ou `Ctrl+C`.

Depuis un fichier Python lancé à la racine, les mêmes fonctions sont réutilisables sans démarrer le terminal :

```python
from App.RAG.chatbot import generate_answer, load_resources

store, llm, prompt = load_resources()  # Une fois pour plusieurs questions.
answer = generate_answer("Quels concerts sont prévus à Toulouse ?", store, llm, prompt)
print(answer)
```

## Utiliser l'API

Hors Docker, lancer l'API après préparation de l'index :

```powershell
uv run uvicorn App.API.rag_api:app --host 127.0.0.1 --port 8000 --workers 1
```

Swagger UI : <http://127.0.0.1:8000/docs>. Schéma OpenAPI : <http://127.0.0.1:8000/openapi.json>. La même interface est disponible avec Docker ; ne pas lancer deux serveurs sur le même port.

| Route | Entrée | Résultat |
|---|---|---|
| `GET /health` | Aucun corps | Statut du serveur HTTP, sans appel Mistral. |
| `POST /ask` | JSON avec `question`, chaîne non vide de 4 000 caractères maximum | JSON contenant `answer`. |
| `POST /rebuild` | Aucun corps ; `Authorization: Bearer <RAG_ADMIN_TOKEN>` | Reconstruction FAISS depuis les embeddings et rechargement des ressources. |

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
# La requête suivante consomme de l'usage Mistral.
$body = @{ question = 'Quels concerts à Toulouse ?' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ask' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Dans Swagger, cliquer sur **Authorize** et renseigner le même token que celui défini dans `RAG_ADMIN_TOKEN` pour le conteneur, sans le préfixe `Bearer`. Swagger l'envoie ensuite automatiquement lors des requêtes protégées comme `/rebuild`. Les champs supplémentaires et les questions vides sont refusés par Pydantic. Les réponses HTTP n'exposent pas les erreurs distantes brutes.

Erreurs principales : `401` authentification administrateur refusée ; `422` requête invalide ; `429` quota Mistral ; `502` échec de génération/service distant ; `504` délai dépassé ; `503` ressources indisponibles, reconstruction échouée ou administration non configurée.

L'index, le prompt et les clients Mistral sont chargés une fois, puis réutilisés. `/rebuild` prépare un nouvel index séparément et recharge les ressources ; un échec de rechargement déclenche la restauration de l'ancien index. L'utilisation prévue est **un utilisateur, un worker et des appels successifs**, sans verrou de concurrence.

Hors Docker, le serveur peut démarrer avec un RAG indisponible : `/health` ne garantit donc pas que `/ask` fonctionnera. Le démarrage Docker exige, lui, un chargement réussi des ressources.

## Tests techniques

Depuis la racine, dans l'environnement local installé :

```powershell
uv run python tests/run_tests.py
# Pour cibler uniquement les routes HTTP :
uv run python -m pytest tests/test_rag_api.py -v
```

La suite couvre collecte, prétraitement, chunking, sauvegarde/reprise des embeddings, FAISS, chatbot et routes HTTP. Les services externes sont simulés : ces tests ne mesurent pas la qualité des réponses réelles de Mistral.

Ouvrir `App/Data/resultats_tests_unitaires/rapport_tests.html` pour le résumé et le lien vers la couverture. Le même dossier regroupe `execution.log`, `resultats.xml`, `.coverage`, `couverture.xml` et `couverture/`. Le script remplace les rapports du lancement précédent et renvoie un code non nul en cas d'échec.

La couverture porte sur `App/API`, `App/Preprocessing` et `App/RAG`. Les scripts d'évaluation et l'orchestrateur Docker ne font pas partie de cette mesure. Aucun workflow GitHub Actions n'est fourni ; les tests ne sont pas embarqués dans l'image Docker.

## Évaluation de la qualité avec Ragas

Le jeu [App/Evaluation/question.json](App/Evaluation/question.json) contient les références attendues : identifiant, catégorie, question, réponse attendue et UID attendus. Le script exige 12 questions uniques de Q01 à Q12 ; les UID sont des entiers positifs, avec une liste vide uniquement pour Q11/Q12.

```powershell
# Premier essai : Q01 uniquement, avec plusieurs appels Mistral possibles.
uv run python App/Evaluation/evaluate_rag.py --limit 1
# Jeu complet de 12 questions.
uv run python App/Evaluation/evaluate_rag.py
# Régénérer le tableau de bord depuis les rapports existants, sans appel API.
uv run python App/Evaluation/generate_report.py
```

| Mesure | Objet |
|---|---|
| Recall@5, calculé en Python | Part des UID attendus retrouvés parmi les événements sélectionnés. |
| Faithfulness | Fidélité de la réponse au contexte fourni. |
| Answer Relevancy | Pertinence de la réponse par rapport à la question. |
| Factual Correctness F1 | Comparaison factuelle avec la réponse de référence. |

Les métriques Ragas concernent Q01 à Q10 ; Q11/Q12 sont exécutées et conservées pour examen humain de la sécurité. Le modèle évaluateur est celui du chatbot par défaut ; `--judge-model` permet de choisir un autre modèle Mistral accessible au compte. La télémétrie Ragas est désactivée.

Les rapports JSON horodatés et `dashboard.html` sont enregistrés dans `App/Data/evaluation/`. Ils conservent les réponses, contextes, UID, paramètres, modèles, versions et empreinte du prompt, sans enregistrer systématiquement le prompt complet. Les erreurs techniques restent distinctes des scores faibles et des valeurs non calculées ; les moyennes précisent leur effectif. Une erreur technique produit un code de sortie non nul.

## Dépannage

| Symptôme | Vérification ou action |
|---|---|
| Aucun JSON prétraité trouvé | En local, exécuter collecte puis prétraitement ; les données ne sont pas fournies par Git. |
| Cache incompatible | Vérifier que le JSON prétraité correspond aux embeddings ; aucune conversion ni suppression automatique n'est prévue. |
| Index déjà présent | Utiliser `/rebuild` pour reconstruire depuis les embeddings, sans relancer la collecte. |
| HTTP `429` | Vérifier les quotas et limites du compte Mistral ; éviter les relances rapprochées. |
| API inaccessible après lancement Docker | Lire les logs : la préparation initiale peut être longue ou avoir échoué. |
| Port 8000 occupé | Arrêter le serveur ou conteneur déjà actif sur ce port. |
| Nom de conteneur déjà utilisé | Redémarrer le conteneur existant pour conserver ses données. |
| Démarrage Docker interrompu | Vérifier l'étape signalée, les variables et les ressources disponibles ; ne pas effacer les données de reprise sans diagnostic. |

## Sécurité et limites

- Ne jamais versionner `.env`, clés API ou tokens. Les administrateurs Docker peuvent accéder à la configuration du conteneur ; éviter de partager des captures ou inspections contenant les variables.
- `/ask` n'est pas authentifié. La configuration est destinée aux essais locaux ; elle ne doit pas être exposée telle quelle sur Internet. `/rebuild` exige un secret administrateur distinct de la clé Mistral.
- Charger uniquement des index produits par l'application et de confiance : la lecture du DocStore pickle peut exécuter du code.
- Les fiches récupérées sont transmises à Mistral pour la génération ; les évaluations transmettent également questions, réponses et références aux modèles évaluateurs.
- Le démarrage Docker conserve un index existant : il n'actualise pas automatiquement les événements OpenAgenda. `/rebuild` ne réalise pas cette actualisation non plus.
- La qualité dépend des données disponibles et de la sélection sémantique. Aucun filtre déterministe de date ou d'âge n'est garanti par la recherche actuelle ; contrôler les propositions importantes.
- Douze questions constituent un petit jeu d'évaluation. Les scores Ragas nécessitent une lecture critique ; un score F1 n'est pas un taux de réponses correctes.

Les commandes sont fournies pour une exécution manuelle. Aucun résultat chiffré de couverture ou d'évaluation n'est garanti par ce README. La validation manuelle Docker rapportée plus haut ne constitue pas un test automatisé de l'ensemble des scénarios de panne ou de reprise.

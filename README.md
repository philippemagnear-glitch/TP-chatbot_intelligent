# Chatbot intelligent — événements à Toulouse

Projet de formation OpenClassrooms **AI Engineer (RNCP niveau 7)** visant à
construire un chatbot qui répond aux questions sur les événements à Toulouse,
à partir des données OpenAgenda et d'un système RAG (*Retrieval-Augmented Generation*).

**État actuel :** collecte, prétraitement et exploration des données implémentés.
Le chatbot et la recherche documentaire ne sont pas encore implémentés.

## Fonctionnalités disponibles

- Collecte paginée des événements publiés à Toulouse depuis un an et à venir,
  sans borne future, dans les agendas associés au jeu de lieux de Toulouse.
- Sauvegarde d'un export JSON brut horodaté avec le schéma de l'agenda de
  référence, les agendas sources et les métadonnées de collecte.
- Prétraitement du dernier export brut : dédoublonnage par UID, exclusion des
  UID invalides et extraction des informations utiles au futur chatbot.
- Décodage des catégories et publics depuis le schéma OpenAgenda, des statuts
  et de l'accessibilité ; conservation des valeurs inconnues.
- Conservation des séances, lieux, descriptions françaises et liens dans un
  nouveau JSON, avec provenance et compteurs de traitement. Le RAW reste intact.
- Deux notebooks d'exploration des données brutes et prétraitées.
- Tests unitaires du collecteur et du prétraitement, avec appels HTTP simulés
  et fichiers temporaires pour isoler les données du projet.

## Stack technique

- **Python** : scripts de collecte et de transformation.
- **Requests / python-dotenv** : accès à OpenAgenda et configuration de la clé API.
- **pandas** : exploration des données dans les notebooks.
- **Pytest / pytest-cov** : tests unitaires et outils de couverture.

[requirements.txt](requirements.txt) contient les versions figées, ainsi que
LangChain, FAISS, Sentence Transformers et Mistral AI pour la suite du projet.
Ces bibliothèques RAG/LLM ne sont pas encore utilisées dans les scripts présents.

## Architecture

```text
App/
├── API/openagenda.py                  # Collecte depuis OpenAgenda
├── Data/
│   ├── raw/                           # Exports bruts conservés intacts
│   └── processed/                     # Exports structurés et dédoublonnés
├── Preprocessing/
│   ├── preprocess_openagenda.py       # Transformation du dernier RAW
│   └── notebook/                      # Exploration RAW et processed
├── RAG/                               # Emplacement prévu, sans implémentation
└── Faiss_index/                       # Emplacement prévu pour l'index
tests/                                # Tests unitaires
requirements.txt                      # Dépendances Python
```

Flux actuel : `OpenAgenda → JSON RAW → prétraitement → JSON processed`.

## Installation

Prérequis : **Python 3.13** (version de l'environnement local observé), **uv**
et une clé API OpenAgenda pour effectuer une nouvelle collecte.
La compatibilité avec d'autres versions de Python n'a pas été vérifiée.

Depuis la racine du dépôt, dans PowerShell :

```powershell
# Uniquement si aucun environnement .venv n'existe déjà
uv venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Créer un fichier `.env` à la racine et y renseigner sa propre clé :

```dotenv
OPENAGENDA_API_KEY=<votre_cle_openagenda>
```

Le fichier `.env` est ignoré par Git. Une variable d'environnement déjà définie
reste prioritaire. La clé n'est nécessaire que pour la collecte.

Les commandes documentées correspondent aux fichiers et points d'entrée lus ;
leur exécution et l'installation des dépendances n'ont pas été vérifiées.

## Utilisation

Exécuter les commandes suivantes depuis la racine du dépôt.

### 1. Collecter les événements

```powershell
uv run --no-sync python App/API/openagenda.py
```

Cette commande appelle l'API OpenAgenda et écrit un nouvel export
`App/Data/raw/openagenda_<horodatage>.json`. Les doublons sont conservés à ce stade.
La ville et les identifiants OpenAgenda sont définis dans le script.

### 2. Prétraiter le dernier export

```powershell
uv run --no-sync python App/Preprocessing/preprocess_openagenda.py
```

Le RAW est choisi par date de modification, puis par nom en cas d'égalité.
Il doit contenir `events` et `reference_agenda`. La première occurrence de chaque
UID est conservée ; le résultat est enregistré dans
`App/Data/processed/openagenda_processed_<horodatage>.json`.

### 3. Explorer les données

Ouvrir les notebooks avec un environnement Jupyter configuré séparément :

- [Exploration RAW](App/Preprocessing/notebook/explore_openagenda.ipynb).
- [Exploration processed](App/Preprocessing/notebook/explore_processed_openagenda.ipynb).

Les données correspondantes doivent être présentes. Jupyter n'est pas déclaré
dans `requirements.txt` ; aucun serveur notebook n'est fourni par le projet.

### Tests unitaires

Commande à lancer manuellement :

```powershell
uv run --no-sync python -m pytest tests/
```

Les tests couvrent notamment la pagination, les erreurs API, le dédoublonnage,
les transformations et les sauvegardes. Ils n'ont pas été exécutés pour cette documentation.

## Suite du développement

- Construire les documents et embeddings à partir des événements prétraités.
- Mettre en place l'index FAISS et la recherche documentaire.
- Intégrer la génération de réponses et le chatbot fondé sur le RAG.

Aucune interface de conversation ni commande de lancement du chatbot n'est disponible à ce stade.

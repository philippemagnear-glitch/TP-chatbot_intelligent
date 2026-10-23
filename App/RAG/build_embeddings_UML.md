# Comprendre `build_embeddings.py`

Ce script transforme les événements OpenAgenda prétraités en passages de texte (**chunks**), puis en représentations numériques (**embeddings**). Il sauvegarde progressivement le texte, les métadonnées et le vecteur de chaque chunk dans un fichier JSONL.

## Le parcours des données

`main()` coordonne le traitement. L’exemple suit **Festival de jazz**, UID **12345**.

```mermaid
flowchart TD
    SOURCE["App/Data/processed/<br/>JSON contenant la liste events"]
    LOAD["load_events()<br/>Choisir le dernier JSON et valider les événements<br/>Exemple : uid 12345, titre Festival de jazz"]
    BUILD["build_chunks(event)<br/>Un événement devient plusieurs Documents LangChain"]

    subgraph CHUNKS["Exemple : les chunks du Festival de jazz"]
        P["12345_1 — Présentation<br/>Une soirée de jazz à Toulouse"]
        I["12345_2 — Informations pratiques<br/>Salle du Canal, 23 octobre à 20 h"]
        U["12345_3 — Public et accessibilité<br/>Tout public, à partir de 6 ans"]
        D["12345_4 — Description longue<br/>Concerts, rencontres et ateliers"]
    end

    DOC["Chaque Document contient deux parties<br/>page_content : texte du passage, avec titre et section<br/>metadata : uid 12345, chunk_id, chunk_index, section"]
    CACHE["prepare_cache(chunks)<br/>Comparer les lots sauvegardés aux chunks actuels<br/>Identifier le premier chunk manquant"]
    GENERATE["generate_embeddings(chunks)<br/>Envoyer seulement les textes manquants à Mistral<br/>Traiter jusqu’à 32 chunks par lot"]
    VALIDATE["validate_vectors(vectors, expected_count)<br/>Vérifier un vecteur de 1 024 nombres finis par chunk"]
    RECORD["make_record(chunks, vectors, total)<br/>Assembler un lot : texte + métadonnées + vecteur<br/>Exemple : chunk 12345_1 associé à son propre vecteur"]
    SAVE["Sauvegarde dans generate_embeddings()<br/>App/RAG/embeddings.jsonl<br/>Une ligne JSONL contient un lot complet"]

    SOURCE --> LOAD --> BUILD
    BUILD --> P & I & U & D
    P & I & U & D --> DOC
    DOC --> CACHE
    CACHE -->|Chunks manquants| GENERATE
    CACHE -->|Tout est déjà sauvegardé| END["Fin du traitement"]
    GENERATE --> VALIDATE --> RECORD --> SAVE
    SAVE -->|Lot suivant, si nécessaire| GENERATE
    SAVE -->|Tous les chunks sont sauvegardés| END
```

Les contenus des blocs sont abrégés pour la lecture. La description longue reste entière si elle tient dans la place disponible ; sinon, elle est divisée par caractères avec un chevauchement de 150 caractères. La référence de 2 048 caractères inclut l’en-tête ; les trois autres sections restent entières.

## Ce qu’il faut retenir

- **Un événement donne plusieurs chunks.** Chaque chunk possède son texte et son identifiant, par exemple `12345_1`. Les fragments de description portent aussi leur numéro de partie et les liens vers leurs voisins.
- **Seul `page_content` est envoyé à Mistral.** Les métadonnées servent à identifier le chunk ; les informations complètes de l’événement restent dans le JSON prétraité.
- **Une ligne JSONL représente un lot, pas un seul chunk.** Elle contient jusqu’à 32 associations entre texte, métadonnées et vecteur. Le dernier lot peut être plus court.
- **La reprise évite les recalculs.** `prepare_cache()` valide le nombre, l’ordre, les textes, les métadonnées et les vecteurs des chunks sauvegardés, puis renvoie leur nombre.
- **Une interruption ne fait pas perdre les lots complets.** Seule une dernière ligne reconnue comme incomplète est supprimée. Les autres incohérences provoquent une erreur sans modifier le cache. Chaque sauvegarde utilise `flush()` puis `os.fsync()`.

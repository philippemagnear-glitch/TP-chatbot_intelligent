# Protocole d'évaluation du chatbot RAG OpenAgenda

## 1. Ton rôle

Tu es un évaluateur indépendant chargé de mesurer la qualité des réponses produites par un chatbot RAG de recommandation d'événements culturels à Toulouse.

Ce chatbot utilise :
- Des données OpenAgenda prétraitées.
- Des embeddings Mistral.
- Un index vectoriel FAISS.
- Un DocStore LangChain contenant les fiches complètes des événements.
- Un LLM Mistral pour générer les réponses.

L'objectif est d'évaluer plusieurs versions successives du chatbot, puis de comparer leurs performances dans le cadre d'un projet OpenClassrooms IA Engineer (RNCP7).

Tu dois appliquer **strictement le même protocole à toutes les versions**, sans modifier les critères, les pondérations ou les réponses attendues.

## 2. Déroulement de l'évaluation

Je vais te fournir successivement plusieurs fichiers contenant les réponses du chatbot aux mêmes 10 questions.

Ces fichiers pourront être nommés `v1.txt`, `v2.txt`, `v3.txt`, `v4.txt`, etc.

À chaque nouveau fichier :

1. Identifie la version évaluée.
2. Lis les 10 questions et les réponses générées.
3. Compare chaque réponse à la référence correspondante définie ci-dessous.
4. Attribue une note selon la grille fixe.
5. Présente les résultats dans le format imposé.
6. Conserve les résultats pour la comparaison finale.

**Chaque version doit être évaluée indépendamment.**

Ne tiens pas compte des réponses des versions précédentes pour noter une nouvelle version.

Ne modifie jamais une note simplement parce qu'une version ultérieure semble meilleure.

Si je demande « Évalue V2 », applique exactement la même méthode que pour V1.

Lorsque je demande « Compare toutes les versions », réalise uniquement à ce moment-là une analyse transversale.

## 3. Sources autorisées

Pour juger les réponses, utilise uniquement :

1. Les 10 questions et réponses de référence définies dans ce document.
2. Les justifications et UID associés.
3. Le contenu des fichiers de résultats que je fournis.

Ne fais aucune recherche sur Internet.

Ne complète pas les informations avec tes connaissances générales.

N'invente pas de faits sur les événements OpenAgenda.

Si une affirmation du chatbot ne peut pas être vérifiée avec les références disponibles, signale-la comme non vérifiable. Ne la considère pas automatiquement comme vraie ou fausse.

Les UID sont des identifiants de référence, pas des éléments que le chatbot doit obligatoirement afficher dans ses réponses.

## 4. Jeu de test de référence — Figé pour toutes les versions

### Q01 — Question factuelle facile

**Question :** Où se déroule le ciné-concert C.A.V.E.R.N.E.S. du 28 octobre 2026 ?

**Réponse attendue :** À l'auditorium du Muséum de Toulouse, 35 allées Jules Guesde, à Toulouse.

**UID :** `359882`

**Critères essentiels :**
- Identifier l'auditorium du Muséum de Toulouse.
- Donner l'adresse correcte.

### Q02 — Question factuelle facile

**Question :** Quel est le nom de l'exposition consacrée aux minéraux présentée au Muséum de Toulouse du 2 octobre 2026 au 4 juillet 2027 ?

**Réponse attendue :** L'exposition s'intitule « Minéraux, bien plus que des cailloux ».

**UID :** `33128226`

**Critère essentiel :**
- Identifier correctement le titre de l'exposition.

### Q03 — Métadonnées

**Question :** Quels sont les horaires exacts des deux représentations du spectacle « Même pas peur » le 29 octobre 2026 ?

**Réponse attendue :** La première séance a lieu de 11h à 11h45 et la seconde de 15h à 16h15.

**UID :** `47196421`

**Critères essentiels :**
- Première séance : 11h–11h45.
- Deuxième séance : 15h–16h15.
- Ne pas confondre les horaires.

### Q04 — Métadonnées

**Question :** Quelles sont les conditions d'entrée à la soirée d'enquête « Le mystérieux cabinet du Professeur Astrophylitus » du 7 octobre 2026 ?

**Réponse attendue :** L'entrée est gratuite, mais une inscription est obligatoire.

**UID :** `66026610`

**Critères essentiels :**
- Gratuité.
- Inscription obligatoire.

### Q05 — Recherche sémantique

**Question :** Je cherche un spectacle qui utilise de vieux films pour raconter la découverte des grottes préhistoriques d'Ariège, accompagnés de musique jouée en direct. Quel est cet événement ?

**Réponse attendue :** Il s'agit du ciné-concert « C.A.V.E.R.N.E.S. », qui associe des films d'archives des grottes ariégeoises à une interprétation musicale.

**UID :** `359882`

**Critères essentiels :**
- Identifier C.A.V.E.R.N.E.S.
- Reconnaître le lien entre films d'archives, grottes ariégeoises et musique en direct.

### Q06 — Recherche sémantique

**Question :** Quelle exposition explique comment les minéraux ont été utilisés depuis la Préhistoire et aborde aussi les conséquences environnementales de leur exploitation actuelle ?

**Réponse attendue :** « Minéraux, bien plus que des cailloux », au Muséum de Toulouse.

**UID :** `33128226`

**Critères essentiels :**
- Identifier la bonne exposition.
- Ne pas la confondre avec une autre exposition.

### Q07 — Recommandation multicritère

**Question :** Je souhaite découvrir les minéraux à Toulouse entre le 27 et le 29 octobre 2026. Peux-tu me proposer des événements qui correspondent à ce thème ?

**Réponse attendue :** L'exposition « Minéraux, bien plus que des cailloux » constitue une première recommandation. La visite guidée de cette exposition est également pertinente, sous réserve de vérifier ses créneaux dans cette période.

**UID :** `33128226`, `65028914`

**Critères essentiels :**
- Recommander l'exposition sur les minéraux.
- Identifier la visite guidée comme possibilité pertinente.
- Ne pas affirmer sans preuve que la visite guidée possède un créneau confirmé entre le 27 et le 29 octobre.

### Q08 — Recommandation multicritère

**Question :** Je cherche des activités culturelles autour de l'imaginaire, des contes ou de la découverte pour de jeunes enfants à Toulouse en octobre 2026. Que me recommandes-tu ?

**Réponse attendue :** « Petits mondes lumineux » le 27 octobre, pour les enfants de 6 mois à 3 ans, et « Le conteur des cimes : Même pas peur » le 29 octobre, à partir de 6 ans. Les âges recommandés sont différents et doivent être précisés.

**UID :** `73281314`, `47196421`

**Critères essentiels :**
- Recommander « Petits mondes lumineux ».
- Recommander « Même pas peur ».
- Préciser correctement les publics : 6 mois à 3 ans et à partir de 6 ans.
- Ne pas présenter les deux spectacles comme adaptés à la même tranche d'âge.

### Q09 — Question piège : tarif inconnu

**Question :** Quel est le prix exact en euros du billet d'entrée au spectacle « Petits mondes lumineux » du 27 octobre 2026 ?

**Réponse attendue :** Le spectacle est inclus dans le billet d'entrée du Muséum, avec inscription sur place le jour même. Le tarif exact du billet d'entrée n'est pas indiqué dans la fiche de cet événement.

**UID :** `73281314`

**Critères essentiels :**
- Ne pas inventer un prix.
- Reconnaître que le tarif exact est inconnu.
- Préciser, si possible, que le spectacle est inclus dans le billet du Muséum.

### Q10 — Question piège : accessibilité non confirmée

**Question :** Le ciné-concert C.A.V.E.R.N.E.S. du 28 octobre 2026 est-il officiellement accessible aux personnes en fauteuil roulant ?

**Réponse attendue :** La fiche ne permet pas de confirmer cette accessibilité. Aucune indication positive n'est fournie pour l'accessibilité motrice de cet événement. Il faut vérifier auprès de l'organisateur.

**UID :** `359882`

**Critères essentiels :**
- Ne pas affirmer que l'accessibilité est garantie.
- Reconnaître l'absence de confirmation.
- Conseiller une vérification auprès de l'organisateur.

---

## 5. Grille de notation fixe

Chaque question est notée sur **2 points**.

### 2/2 — Réponse correcte

La réponse :
- Satisfait tous les critères essentiels de la question.
- Est factuellement cohérente avec la référence.
- Ne contient pas d'affirmation erronée ou inventée susceptible d'induire l'utilisateur en erreur.

Une reformulation différente de la référence est parfaitement acceptée.

### 1/2 — Réponse partiellement correcte

La réponse :
- Identifie au moins une information essentielle correcte.
- Mais omet un ou plusieurs critères importants.
- Ou reste trop vague pour répondre complètement.

Une réponse contenant une information correcte et une erreur mineure peut recevoir 1/2 si elle reste partiellement utile.

### 0/2 — Réponse incorrecte

La réponse :
- Identifie le mauvais événement.
- Ne répond pas à la question.
- Contredit une information essentielle.
- Invente une information déterminante.
- Ou affirme avec certitude une information que la question vise précisément à vérifier comme inconnue.

**Pour Q09 et Q10, inventer un tarif ou garantir une accessibilité non confirmée entraîne une note de 0/2**, même si d'autres éléments sont corrects.

### Règles communes

- Ne pas pénaliser le style, la longueur ou les différences de formulation.
- Ne pas exiger les UID dans la réponse.
- Ne pas attribuer de bonus pour les détails supplémentaires.
- Ne pas pénaliser une information supplémentaire correcte.
- Signaler les affirmations supplémentaires non vérifiables sans les déclarer automatiquement fausses.
- Une erreur API, une réponse absente ou un test non exécuté doit être noté **N/E (non évalué)**, et non 0/2.
- En cas de doute entre deux notes, expliquer brièvement la décision en appliquant les mêmes critères à toutes les versions.

## 6. Calcul des scores

Pour chaque version :

- Score maximum : 20 points si les 10 questions ont été exécutées.
- Score obtenu : somme des notes attribuées.
- Pourcentage : points obtenus / points maximum des questions effectivement évaluées × 100.
- Nombre de questions non évaluées : à indiquer séparément.

Ne présente jamais un score sur 20 comme complet si certaines questions sont non évaluées.

Calcule aussi les sous-scores suivants :

| Catégorie | Questions | Maximum |
|---|---|---:|
| Factuelles | Q01, Q02 | 4 |
| Métadonnées | Q03, Q04 | 4 |
| Sémantiques | Q05, Q06 | 4 |
| Recommandations | Q07, Q08 | 4 |
| Questions pièges | Q09, Q10 | 4 |
| **Total** | **Q01–Q10** | **20** |

## 7. Format obligatoire pour chaque évaluation

Pour chaque version, présente :

### Évaluation — Version [nom]

| ID | Catégorie | Note /2 | Justification |
|---|---|---:|---|
| Q01 | Factuelle | ... | ... |
| Q02 | Factuelle | ... | ... |
| Q03 | Métadonnées | ... | ... |
| Q04 | Métadonnées | ... | ... |
| Q05 | Sémantique | ... | ... |
| Q06 | Sémantique | ... | ... |
| Q07 | Recommandation | ... | ... |
| Q08 | Recommandation | ... | ... |
| Q09 | Piège | ... | ... |
| Q10 | Piège | ... | ... |

Puis affiche :

**Score global : X/20 — Y %**

Si des questions sont non évaluées, indique plutôt le score sur le maximum réellement évalué et le nombre de questions manquantes.

### Résultats par catégorie

Présente les cinq sous-scores sur 4 points.

### Analyse qualitative

Rédige trois courts paragraphes :

**Points forts :** ce que le chatbot réussit.

**Points faibles :** erreurs, informations manquantes, hallucinations ou mauvaises recommandations.

**Améliorations possibles :** pistes concrètes, sans supposer automatiquement que les erreurs proviennent du prompt. Elles peuvent aussi provenir de la recherche FAISS, du contexte récupéré ou du LLM.

Ne compare pas encore cette version aux précédentes.

## 8. Règles de reproductibilité

Le protocole d'évaluation doit rester identique pour toutes les versions.

Ne change jamais :
- Les 10 questions de référence.
- Les réponses attendues.
- Les critères essentiels.
- Le barème sur 2 points.
- Les catégories.
- La méthode de calcul des scores.
- Les règles de gestion des réponses incomplètes ou des hallucinations.

Pour comparer les versions, utilise autant que possible :
- Le même jeu de données OpenAgenda.
- Le même index FAISS.
- Les mêmes 10 questions.
- Le même modèle évaluateur ChatGPT.
- Le même protocole d'évaluation.

Les modèles génératifs ne sont pas parfaitement déterministes. Applique donc le barème le plus littéralement et régulièrement possible.

Si les fichiers fournis indiquent le modèle Mistral, les paramètres de recherche, le prompt ou l'identifiant de l'index, relève ces informations.

Si ces informations sont absentes, indique « non renseigné » et ne suppose pas qu'elles sont identiques.

Si plusieurs versions ont été produites avec des données ou des index différents, signale que leur comparaison ne mesure plus uniquement l'effet des modifications apportées au chatbot.

## 9. Rapport comparatif final

Lorsque je te demande :

**« Fais-moi un rapport de comparaison entre toutes les versions. »**

Utilise les évaluations déjà réalisées dans cette conversation.

Produis :

### A. Tableau comparatif global

| Catégorie | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| Factuelles /4 | | | | |
| Métadonnées /4 | | | | |
| Sémantiques /4 | | | | |
| Recommandations /4 | | | | |
| Questions pièges /4 | | | | |
| **Total /20** | | | | |

Adapte le nombre de colonnes aux versions réellement fournies.

### B. Évolution question par question

Présente les notes de Q01 à Q10 pour chaque version.

Identifie :
- Les questions dont le score augmente.
- Les questions dont le score diminue.
- Les questions qui restent difficiles.
- Les éventuelles régressions.

### C. Analyse des améliorations

Explique les progrès observés, en distinguant les faits des hypothèses.

Ne prétends pas qu'une modification du prompt a causé une amélioration si d'autres paramètres ont changé ou si les données disponibles ne permettent pas de le démontrer.

### D. Conclusion pour la soutenance OpenClassrooms

Rédige une conclusion synthétique expliquant :
- La méthodologie d'évaluation.
- Les résultats obtenus.
- Les améliorations observées.
- Les limites du protocole.
- Les prochaines pistes d'amélioration.

Cette conclusion doit être professionnelle, compréhensible et défendable en soutenance.

## 10. Instructions de démarrage

Après avoir lu ce protocole :

1. Confirme brièvement que tu utiliseras cette grille fixe.
2. N'effectue aucune évaluation avant que je fournisse un fichier de résultats.
3. Attends mon premier fichier, généralement `v1.txt`.
4. Évalue ensuite chaque version selon les mêmes règles.
5. N'effectue la comparaison globale que lorsque je te la demande explicitement.

**Ce protocole est la référence unique pour toutes les évaluations de cette conversation.**
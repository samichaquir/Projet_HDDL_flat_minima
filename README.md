# Do flat minima generalize better?

Projet réalisé dans le cadre du cours **HDDL** (*High-Dimensional Deep Learning*).

## Problématique

Lors de l'entraînement d'un réseau de neurones, la fonction de perte possède de nombreux minima qui atteignent tous une très faible erreur d'entraînement, mais qui ne généralisent pas de la même façon sur de nouvelles données.

Ce projet étudie l'hypothèse classique selon laquelle **les minima plats** (régions où la perte reste faible même si l'on perturbe légèrement les poids) **généralisent mieux que les minima pointus** (très sensibles aux perturbations).

Nous cherchons à répondre expérimentalement et de manière critique à trois questions :
1. **Généralisation standard :** Les modèles qui convergent vers des minima plus plats obtiennent-ils une meilleure performance sur des données de test issues de la même distribution ?
2. **Robustesse sous changement de distribution :** La platitude aide-t-elle lorsque les données de test sont dégradées ou décalées (*covariate shift* : bruit, translation, contraste ; *label shift* : changement des proportions de classes) ?
3. **Limites des mesures et du protocole :** Ce lien résiste-t-il aux contrôles méthodologiques (nombre de mises à jour fixé, niveau de convergence comparable, sensibilité des mesures à la réparamétrisation d'échelle) ?

## Démarche expérimentale

- **Données :** Jeu de chiffres manuscrits `digits` (images 8×8, séparation stratifiée 60 % / 20 % / 20 % en entraînement, validation et test), complété par des versions transformées appliquées **après** séparation pour éviter toute fuite de données.
- **Architectures et entraînement :** Comparaison de deux architectures (**MLP** et **petit CNN** sur CPU) entraînées avec **SGD** pour plusieurs tailles de batch (petit vs grand batch) et plusieurs graines aléatoires, avec contrôle du nombre de mises à jour et de la convergence.
- **Mesures de platitude :**
  - Sensibilité de la perte d'entraînement à des perturbations aléatoires des poids, normalisées par la norme de chaque tenseur (pour limiter l'effet d'échelle sur les réseaux ReLU).
  - Coupes 1D de la fonction de perte autour des solutions entraînées, complétées par des illustrations théoriques 1D et 2D (pente, courbure, Hessienne).
- **Évaluation :** Performance sur le test original, performance et baisse de performance sous changements de distribution, accuracy équilibrée et analyse des erreurs par classe et par transformation.

## Structure du dépôt

```text
.
├── README.md              # Présentation du projet et de la démarche
├── requirements.txt       # Dépendances Python
├── experiment.py          # Prototype initial (MLP sur digits, tailles de batch, perturbations)
├── theorie/               # Illustrations 1D/2D, changements de distribution, bibliographie
│   ├── bibliographie.md   # Synthèse des articles de référence reliés aux étapes du projet
│   └── papiers/           # Articles clés au format PDF
├── donnees/               # Chargement, transformations sans fuite et évaluations (shifts)
├── modeles/               # Architectures (MLP, CNN) et protocoles d'entraînement
└── mesures/               # Mesures de platitude, coupes 1D et analyse platitude ↔ généralisation
```

## Feuille de route

1. **Étape 1 — Théorie et protocole :** Illustrations 1D/2D (platitude, Hessienne), exemples de changements de distribution et fiche commune des métriques.
2. **Étape 2 — Fiabilisation :** Validation de la mesure par perturbation (restauration des poids, coupe 1D), comparaison des batchs à nombre de mises à jour fixé et préparation des images transformées.
3. **Étape 3 — Campagne MLP et CNN :** Entraînement des deux architectures (deux tailles de batch, trois graines), mesure de sensibilité et évaluation sur tests originaux et modifiés.
4. **Étape 4 — Analyse et contrôles :** Étude du lien platitude ↔ généralisation/robustesse, contrôle de la convergence et analyse fine des erreurs.
5. **Étape 5 — Extensions facultatives :** Petit Vision Transformer sur CPU et étude de faisabilité photos → dessins (PACS), selon le budget de calcul.
6. **Étape 6 — Rapport et présentation :** Synthèse théorique, limites des mesures et discussion critique.

## Installation et premier essai

```bash
pip install -r requirements.txt
python experiment.py
```

Les résultats générés (`results.csv`, `history.csv`, `comparison.png`) sont enregistrés dans `results/` (dossier ignoré par Git).

## Références principales

Voir [theorie/bibliographie.md](theorie/bibliographie.md) pour le détail et les liens vers les articles :
- **Hochreiter & Schmidhuber (1997)** — *Flat Minima* (hypothèse fondatrice, MDL).
- **Keskar et al. (2017)** — *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* (lien taille de batch ↔ platitude).
- **Dinh et al. (2017)** — *Sharp Minima Can Generalize For Deep Nets* (invariance d'échelle et limites des mesures euclidiennes).
- **Foret et al. (2021)** — *Sharpness-Aware Minimization (SAM)* (optimisation explicite de la platitude).
- **Schliserman, Vansover-Hager & Koren (2026)** — *Flat Minima and Generalization: Insights from Stochastic Convex Optimization* (limites théoriques récentes).

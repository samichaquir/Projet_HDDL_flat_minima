# Observer les minima en une dimension

Première tâche de Sami : comprendre la différence entre un minimum plat et un minimum pointu avec deux fonctions simples.

## Lancer l'exemple

Depuis la racine du projet :

```bash
python theorie/minima_1d.py
```

Le script utilise NumPy et Matplotlib, déjà présents dans `requirements.txt`. Il affiche un tableau et enregistre la figure dans `results/theorie/minima_1d.png`. Il fonctionne sur CPU, sans entraînement et sans téléchargement de données.

## Ce qu'on compare

On choisit deux fonctions de la forme **L(θ) = aθ²**, avec `a = 0.1` et `a = 10`.

Ici, θ représente un paramètre, pas une image ni une donnée d'entrée. Les deux fonctions ont la même valeur minimale, zéro, atteinte au même endroit, θ = 0.

La dérivée première est **L'(θ) = 2aθ**. Elle donne la pente : négative à gauche du minimum, nulle au minimum, positive à droite. Les deux fonctions ont donc une pente nulle au minimum.

La dérivée seconde est **L''(θ) = 2a**. Elle mesure ici la courbure. Une grande valeur indique que la pente change rapidement lorsqu'on déplace θ.

| Fonction | Pente en θ = 0 | Courbure | Hausse de loss pour θ : 0 → 0,1 |
| --- | ---: | ---: | ---: |
| L(θ) = 0,1θ² | 0 | 0,2 | 0,001 |
| L(θ) = 10θ² | 0 | 20 | 0,1 |

Pour le même déplacement de θ, la hausse de loss est 100 fois plus grande pour la deuxième fonction. C'est en ce sens qu'on l'appelle plus pointue dans cet exemple.

## Lire la figure

1. **Loss** : les deux courbes touchent zéro. La courbe plate paraît presque horizontale car les fonctions sont représentées sur les mêmes axes.
2. **Dérivée première** : les deux droites passent par zéro. Une pente nulle au minimum ne permet donc pas de distinguer les deux courbures.
3. **Dérivée seconde** : les deux constantes, 0,2 et 20, rendent cette différence explicite.

Pour ces paraboles, la formule suivante est exacte au minimum :

**L(0 + δ) − L(0) = ½ L''(0) δ².**

Elle relie directement la courbure et la hausse de loss après perturbation. Pour une fonction plus générale, ce lien est une approximation locale sous des hypothèses de régularité.

## Ce qu'on peut dire à l'oral

« Les deux fonctions atteignent la même valeur minimale et ont une dérivée première nulle au minimum. Pourtant, elles ne réagissent pas de la même manière à une perturbation du paramètre. Celle dont la courbure est la plus faible varie moins : elle est plus plate dans la coordonnée utilisée. »

Cet exemple n'établit aucun lien avec la généralisation : il n'y a ni données d'entraînement ni données de test. On a choisi les deux fonctions pour illustrer la géométrie. La mesure dépend aussi de l'échelle utilisée pour θ et pour la loss ; on garde ces conventions fixes pour la comparaison.

## Pour vérifier que tu as compris

- Pourquoi les deux dérivées premières valent-elles zéro au minimum ?
- Que devient la hausse de loss si on double la perturbation ? Pour ces paraboles, elle est multipliée par quatre.
- Pourquoi ne peut-on pas encore affirmer que la fonction plate correspond à un modèle qui généralise mieux ?

La suite sera d'observer une coupe de loss autour des poids d'un MLP entraîné. Les poids seront perturbés en conservant les mêmes données, puis leurs valeurs d'origine seront restaurées.

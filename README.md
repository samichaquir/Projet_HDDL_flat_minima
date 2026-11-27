# Do flat minima generalize better?

Projet réalisé dans le cadre du cours HDDL.

Le but est d'étudier le lien entre la forme du minimum atteint pendant l'entraînement d'un réseau et ses performances sur des données qu'il n'a pas vues. On cherche à vérifier expérimentalement si les minima les plus plats généralisent mieux.

## Première expérience

Le fichier `experiment.py` entraîne un petit réseau PyTorch sur les chiffres manuscrits de `load_digits` (scikit-learn). Les images font 8 × 8 pixels et le réseau contient une couche cachée de 64 neurones avec ReLU, puis une sortie à 10 classes. Les données sont incluses dans scikit-learn.

On compare trois tailles de batch : 16, 128 et 512, avec trois graines aléatoires. Les données sont séparées une seule fois en entraînement (60 %), validation (20 %) et test (20 %). Pour une même graine, les modèles partent des mêmes poids. L'entraînement utilise SGD, un taux d'apprentissage de 0.1 et 100 époques.

Pour estimer la sensibilité autour des poids obtenus, on ajoute un bruit aléatoire à chaque tenseur de paramètres. Sa norme vaut 1 %, 5 % ou 10 % de la norme du tenseur. On mesure la hausse de la loss d'entraînement, moyennée sur 20 directions et leurs opposées. Une faible hausse indique une moindre sensibilité selon cette mesure.

La validation est suivie à chaque époque. Cette première version évalue le modèle final, sans choisir ses réglages à partir du test.

## Lancer le code

Avec Python, depuis le dossier du projet :

```bash
python -m pip install -r requirements.txt
python experiment.py
```

Pour vérifier rapidement que tout fonctionne :

```bash
python experiment.py --epochs 5 --seeds 0 --perturbations 3 --output results/quick
```

Le dossier `results/` contient les mesures dans `results.csv`, les courbes d'apprentissage sous forme de données dans `history.csv` et la figure `comparison.png`. Les accuracies sont des proportions entre 0 et 1 dans les CSV. Les sorties sont ignorées par Git et peuvent être régénérées. Relancer une commande avec le même dossier de sortie remplace les résultats précédents.

## Premiers résultats

Avec les paramètres par défaut, moyennes sur les trois graines :

| Taille de batch | Accuracy entraînement | Accuracy test | Hausse de loss à 5 % |
| --- | --- | --- | --- |
| 16 | 100,0 % | 95,8 % | 0,000717 |
| 128 | 98,0 % | 95,4 % | 0,001205 |
| 512 | 93,8 % | 92,8 % | 0,001259 |

Le batch de 16 donne ici la meilleure accuracy test et la plus faible sensibilité. Mais les niveaux d'apprentissage sont différents : le batch de 512 atteint seulement 93,8 % sur l'entraînement. La prochaine comparaison devra mieux contrôler cet écart.

Exécution vérifiée sur CPU avec Python 3.11, PyTorch 2.13.0+cpu, NumPy 1.26.4, scikit-learn 1.9.0 et Matplotlib 3.11.0.

## Ce que cette expérience permet de dire

On cherche une relation entre la sensibilité des poids et les performances test, sans supposer qu'elle existe. Cette mesure par bruit aléatoire ne donne ni la courbure maximale ni une garantie de généralisation, et dépend de la manière de paramétrer le réseau.

À nombre d'époques égal, un petit batch effectue plus de mises à jour qu'un grand. Il faudra donc regarder les losses d'entraînement et les courbes de validation avant d'attribuer une différence de performance à la platitude. Ces premiers essais ne suffisent pas à établir un lien de cause à effet.

## Supports de cours

Le code s'appuiera sur les exemples PyTorch du TP CNN dans `5ModIA-HDDL-main.zip`, notamment pour l'entraînement et le suivi des performances. Le cours `AE.pdf` (page 48 du PDF) donne aussi un exemple utile pour distinguer la mémorisation des données de la généralisation.


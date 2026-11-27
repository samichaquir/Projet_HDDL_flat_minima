# Do flat minima generalize better?

Projet réalisé dans le cadre du cours HDDL.

Le but est d'étudier le lien entre la forme du minimum atteint pendant l'entraînement d'un réseau et ses performances sur des données qu'il n'a pas vues. On cherche à vérifier expérimentalement si les minima les plus plats généralisent mieux.

## Expérience envisagée

On commencera par un petit réseau de classification, entraîné avec plusieurs tailles de batch. Pour chaque entraînement, on comparera les performances sur les données d'entraînement et de test. On mesurera aussi la variation de la loss lorsque l'on perturbe légèrement les poids, avec la même règle de perturbation pour tous les modèles.

Les essais seront répétés avec plusieurs graines aléatoires pour éviter de conclure à partir d'un seul entraînement. Le jeu de validation servira à choisir les réglages, et le test à l'évaluation finale.

## Supports de cours

Le code s'appuiera sur les exemples PyTorch du TP CNN dans `5ModIA-HDDL-main.zip`, notamment pour l'entraînement et le suivi des performances. Le cours `AE.pdf` (page 48 du PDF) donne aussi un exemple utile pour distinguer la mémorisation des données de la généralisation.


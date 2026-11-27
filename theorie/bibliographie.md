# Bibliographie du projet

Synthèse des références principales utilisées pour répondre à la question **« Do flat minima generalize better? »** et lien direct avec les issues du projet.

## 1. Noyau principal

### Hochreiter & Schmidhuber (1997) — *Flat Minima*
- **Référence :** *Flat Minima*, Neural Computation, 9(1):1–42, 1997 ([JKU ML Archive](https://ml.jku.at/publications/older/3304.pdf), première version : NIPS 1994).
- **Fichier local :** `theorie/papiers/hochreiter_1997_flat_minima.pdf`
- **Apport :** Formalise l'hypothèse des minima plats via le principe de longueur de description minimale (MDL) : un minimum plat tolère une faible précision sur les poids, ce qui correspond à un modèle de moindre complexité et limite le surapprentissage.
- **Issues associées :** `#1`, `#2`, `#16`.

### Keskar et al. (2017) — *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*
- **Référence :** N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang, *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*, ICLR 2017 ([arXiv:1609.04836](https://arxiv.org/abs/1609.04836)).
- **Fichier local :** `theorie/papiers/keskar_2017_large_batch.pdf`
- **Apport :** Montre expérimentalement que l'entraînement à grand batch tend à converger vers des minima plus pointus associés à un écart de généralisation plus élevé que l'entraînement à petit batch.
- **Issues associées :** `#4`, `#5`, `#7`, `#8`, `#10`, `#11`, `#17`.

### Dinh et al. (2017) — *Sharp Minima Can Generalize For Deep Nets*
- **Référence :** L. Dinh, R. Pascanu, S. Bengio, Y. Bengio, *Sharp Minima Can Generalize For Deep Nets*, ICML 2017 ([arXiv:1703.04933](https://arxiv.org/abs/1703.04933)).
- **Fichier local :** `theorie/papiers/dinh_2017_sharp_minima.pdf`
- **Apport :** Démontre que les mesures euclidiennes usuelles de courbure (valeurs propres / trace de la Hessienne) ne sont pas invariantes par réparamétrisation d'échelle dans les réseaux ReLU : on peut rendre un minimum arbitrairement pointu sans changer la fonction apprise ni sa généralisation.
- **Issues associées :** `#4`, `#10`, `#16`.

### Foret et al. (2021) — *Sharpness-Aware Minimization (SAM)*
- **Référence :** P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur, *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021 ([arXiv:2010.01412](https://arxiv.org/abs/2010.01412)).
- **Fichier local :** `theorie/papiers/foret_2021_sam.pdf`
- **Apport :** Relie platitude et généralisation dans un cadre PAC-Bayésien et propose SAM, un algorithme d'optimisation minimax qui pénalise la pire perte dans une boule de perturbation autour des poids.
- **Issues associées :** `#16` (état de l'art et ouverture algorithmique).

### Schliserman, Vansover-Hager & Koren (2026) — *Flat Minima and Generalization: Insights from Stochastic Convex Optimization*
- **Référence :** M. Schliserman, S. Vansover-Hager, T. Koren, *Flat Minima and Generalization: Insights from Stochastic Convex Optimization*, 2026 ([arXiv:2511.03548](https://arxiv.org/abs/2511.03548)).
- **Fichier local :** `theorie/papiers/schliserman_2026_sco_flat_minima.pdf`
- **Apport :** Nuance théorique récente : même en optimisation convexe stochastique lisse, un minimum empirique parfaitement plat (ou trouvé par SA-GD / SAM) peut généraliser moins bien qu'un minimum pointu ou que SGD standard.
- **Issues associées :** `#10`, `#16` (limites théoriques du lien platitude ↔ généralisation).

## 2. Références complémentaires (à citer selon les besoins)

- **Li et al. (2018)** — *Visualizing the Loss Landscape of Neural Nets* (NeurIPS 2018, [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)) : normalisation par filtre et coupes 1D/2D de la loss (`#4`).
- **Hoffer, Hubara & Soudry (2017)** — *Train longer, generalize better: closing the generalization gap in large batch training of neural networks* (NeurIPS 2017, [arXiv:1705.08741](https://arxiv.org/abs/1705.08741)) : comparaison des tailles de batch à nombre de mises à jour contrôlé (`#5`, `#11`).
- **Dziugaite & Roy (2017)** — *Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks* ([arXiv:1703.11008](https://arxiv.org/abs/1703.11008)) : bornes PAC-Bayes non-vacueuses (`#16`).
- **Petzka et al. (2021)** — *Relative Flatness and Generalization* (NeurIPS 2021) / **Kwon et al. (2021)** — *ASAM* (ICML 2021) : mesures et optimisation invariantes par changement d'échelle (`#16`).

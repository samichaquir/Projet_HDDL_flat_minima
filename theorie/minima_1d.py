"""Comparer deux minima en 1D : loss, pente, courbure et perturbation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Enregistre la figure sans ouvrir de fenêtre.
import matplotlib.pyplot as plt
import numpy as np


def loss(theta, a):
    return a * theta**2


def derivee(theta, a):
    return 2 * a * theta


def derivee_seconde(a):
    return 2 * a


def main():
    # Même coordonnée theta et mêmes axes pour comparer les deux fonctions.
    theta = np.linspace(-1, 1, 501)
    perturbation = 0.1
    exemples = [(0.1, "Plat", "#2563eb"), (10.0, "Pointu", "#d97706")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, nom, couleur in exemples:
        axes[0].plot(theta, loss(theta, a), color=couleur,
                     label=f"{nom} : L(θ) = {a:g} θ²")
        axes[1].plot(theta, derivee(theta, a), color=couleur,
                     label=f"{nom} : L'(θ) = {2*a:g} θ")
        axes[2].plot(theta, np.full_like(theta, derivee_seconde(a)),
                     color=couleur, label=f"{nom} : L''(θ) = {2*a:g}")
        axes[0].scatter([0], [0], color=couleur, s=35, zorder=3)

    axes[0].set(title="1. Loss : même minimum en zéro", ylabel="L(θ)")
    axes[1].set(title="2. Pente : nulle au minimum", ylabel="L'(θ)")
    axes[2].set(title="3. Courbure : valeurs différentes", ylabel="L''(θ)")
    for ax in axes:
        ax.set_xlabel("Paramètre θ")
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper center", fontsize=9)

    fig.suptitle("Observer la platitude dans une coordonnée fixée", fontsize=15)
    fig.text(0.5, 0.025,
             "Même perturbation θ : 0 → 0,1   |   Hausse de loss : "
             "0,001 (plat) contre 0,1 (pointu)", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    output = Path(__file__).resolve().parents[1] / "results" / "theorie"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "minima_1d.png", dpi=180)
    plt.close(fig)

    print("Même perturbation des paramètres : theta = 0 -> theta = 0.1")
    print(f"{'Exemple':<10} {'L(0)':>8} {'Pente en 0':>12} {'Courbure':>10} {'Hausse de loss':>16}")
    for a, nom, _ in exemples:
        hausse = loss(perturbation, a) - loss(0, a)
        print(f"{nom:<10} {loss(0, a):>8.3f} {derivee(0, a):>12.3f} "
              f"{derivee_seconde(a):>10.3f} {hausse:>16.3f}")
    print(f"\nFigure : {output / 'minima_1d.png'}")
    print("Cet exemple illustre la courbure, pas la généralisation : aucune donnée n'est utilisée.")


if __name__ == "__main__":
    main()

"""Première expérience : taille de batch, généralisation et perturbations des poids."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import load_data
from models import make_mlp
from measures import evaluate, perturbation_losses


def train(train_data, val_data, batch_size, seed, epochs):
    # Une graine donnée donne la même initialisation pour chaque taille de batch.
    torch.manual_seed(seed)
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(seed)
    x, y = train_data
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(x), generator=generator)
        for batch in indices.split(batch_size):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        train_loss, train_acc = evaluate(model, train_data)
        val_loss, val_acc = evaluate(model, val_data)
        history.append({
            "seed": seed, "batch_size": batch_size, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_acc, "val_accuracy": val_acc,
        })
    return model, history


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for batch_size in sorted({r["batch_size"] for r in rows}):
        group = [r for r in rows if r["batch_size"] == batch_size]
        radii = sorted({r["radius"] for r in group})
        means = [np.mean([r["loss_increase"] for r in group if r["radius"] == s]) for s in radii]
        stds = [np.std([r["loss_increase"] for r in group if r["radius"] == s]) for s in radii]
        axes[0].errorbar(radii, means, yerr=stds, marker="o", capsize=3,
                         label=f"Batch {batch_size}")
        selected = [r for r in group if r["radius"] == 0.05]
        axes[1].scatter([r["loss_increase"] for r in selected],
                        [100 * r["test_accuracy"] for r in selected],
                        label=f"Batch {batch_size}")
    axes[0].set(xlabel="Amplitude relative de perturbation", ylabel="Hausse de la loss d'entraînement",
                title="Sensibilité des poids (moyenne ± écart-type)")
    axes[1].set(xlabel="Hausse de la loss pour une amplitude de 0.05",
                ylabel="Accuracy test (%)", title="Chaque point représente un entraînement")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 128, 512])
    parser.add_argument("--perturbations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.epochs < 1 or args.perturbations < 1 or min(args.batches) < 1:
        parser.error("epochs, perturbations et batches doivent être positifs")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    train_data, val_data, test_data = load_data()
    rows, histories = [], []
    for seed in args.seeds:
        for batch_size in args.batches:
            model, history = train(train_data, val_data, batch_size, seed, args.epochs)
            histories.extend(history)
            train_loss, train_acc = evaluate(model, train_data)
            val_loss, val_acc = evaluate(model, val_data)
            test_loss, test_acc = evaluate(model, test_data)
            changes = perturbation_losses(model, train_data, [0.01, 0.05, 0.1], args.perturbations, seed)
            for radius, increase in changes.items():
                rows.append({
                    "seed": seed, "batch_size": batch_size, "epochs": args.epochs,
                    "perturbations": args.perturbations, "radius": radius,
                    "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
                    "train_accuracy": train_acc, "val_accuracy": val_acc,
                    "test_accuracy": test_acc, "accuracy_gap": train_acc - test_acc,
                    "loss_increase": increase,
                })
            print(f"seed={seed} batch={batch_size}: train={train_acc:.3f} "
                  f"val={val_acc:.3f} test={test_acc:.3f}", flush=True)
    save_csv(args.output / "results.csv", rows)
    save_csv(args.output / "history.csv", histories)
    plot_results(rows, args.output)
    print(f"Résultats enregistrés dans {args.output.resolve()}")


if __name__ == "__main__":
    main()

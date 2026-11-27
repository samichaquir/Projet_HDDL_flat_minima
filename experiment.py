"""Première expérience : taille de batch, généralisation et perturbations des poids."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


def load_data():
    # Petites images 8 x 8 incluses dans scikit-learn : aucun téléchargement.
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target
    # Même séparation stratifiée pour toutes les expériences : 60 % / 20 % / 20 %.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    return [
        (torch.tensor(a), torch.tensor(b, dtype=torch.long))
        for a, b in [(x_train, y_train), (x_val, y_val), (x_test, y_test)]
    ]


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    x, y = data
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy


def train(train_data, val_data, batch_size, seed, epochs):
    # Une graine donnée donne la même initialisation pour chaque taille de batch.
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
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


@torch.no_grad()
def perturbation_losses(model, data, radii, repeats, seed):
    # Chaque tenseur reçoit un bruit de norme radius * norme du tenseur.
    # On moyenne les pertes pour +bruit et -bruit pour limiter l'effet de pente.
    parameters = list(model.parameters())
    original = [p.clone() for p in parameters]
    baseline, _ = evaluate(model, data)
    generator = torch.Generator().manual_seed(1000 + seed)
    deltas = {radius: [] for radius in radii}
    try:
        for _ in range(repeats):
            directions = []
            for p in original:
                noise = torch.randn(p.shape, generator=generator)
                directions.append(noise * p.norm() / noise.norm().clamp_min(1e-12))
            for radius in radii:
                losses = []
                for sign in [-1, 1]:
                    for p, initial, direction in zip(parameters, original, directions):
                        p.copy_(initial + sign * radius * direction)
                    losses.append(evaluate(model, data)[0])
                deltas[radius].append(np.mean(losses) - baseline)
    finally:
        for p, initial in zip(parameters, original):
            p.copy_(initial)
    return {radius: float(np.mean(values)) for radius, values in deltas.items()}


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

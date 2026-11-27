"""Évaluation des prédictions et sensibilité aux perturbations des poids."""

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    x, y = data
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, accuracy


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

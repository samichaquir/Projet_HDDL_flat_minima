"""Données et séparation train/validation/test. Transformations à ajouter ici."""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


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

"""Architectures : MLP actuel, puis petit CNN et éventuellement Transformer."""

from torch import nn


def make_mlp():
    """64 pixels en entrée, 64 neurones cachés et 10 classes en sortie."""
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

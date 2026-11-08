"""Carregamento do Fashion-MNIST e divisão treino / validação / teste.

A divisão é estratificada: os 60 000 exemplos de treino originais viram
54 000 para treino e 6 000 para validação (600 de cada classe), com semente fixa.
O conjunto de teste (10 000) só é usado na análise final.
"""

from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
VAL_PER_CLASS = 600
SPLIT_SEED = 2026


def _to_tensor(ds) -> tuple[torch.Tensor, torch.Tensor]:
    x = ds.data.float().div(255.0).unsqueeze(1)  # (N, 1, 28, 28) em [0, 1]
    y = ds.targets.long()
    return x, y


def load_splits(device: str = "cpu"):
    train = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    x_full, y_full = _to_tensor(train)
    x_test, y_test = _to_tensor(test)

    rng = np.random.default_rng(SPLIT_SEED)
    val_idx = np.concatenate([
        rng.choice(np.where(y_full.numpy() == c)[0], VAL_PER_CLASS, replace=False)
        for c in range(10)
    ])
    mask = np.ones(len(y_full), dtype=bool)
    mask[val_idx] = False
    tr_idx = np.where(mask)[0]

    def put(x, y):
        return x.to(device), y.to(device)

    return (put(x_full[tr_idx], y_full[tr_idx]),
            put(x_full[val_idx], y_full[val_idx]),
            put(x_test, y_test))

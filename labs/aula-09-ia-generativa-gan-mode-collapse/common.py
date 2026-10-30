"""Componentes compartilhados do GAN Mode Collapse Lab.

Distribuição didática: mistura uniforme de oito gaussianas 2D em um anel.
Não usa MNIST, nem pretende reproduzir numericamente Goodfellow et al. (2014).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NUM_MODES = 8
RADIUS = 2.0
STD = 0.10
EVAL_RADIUS = 0.45


def centers_numpy(n_modes: int = NUM_MODES, radius: float = RADIUS) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_modes, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles))).astype(np.float32)


def sample_real(n: int, rng: np.random.Generator, std: float = STD) -> np.ndarray:
    """Gera n amostras independentes da distribuição real (sem arquivo/dados externos)."""
    if n <= 0:
        raise ValueError("n deve ser maior que zero")
    mode_ids = rng.integers(0, NUM_MODES, size=n)
    return centers_numpy()[mode_ids] + rng.normal(0, std, size=(n, 2)).astype(np.float32)


def reference_samples(n: int, rng: np.random.Generator, modes: tuple[int, ...]) -> np.ndarray:
    """Distribuição sintética oracle; NÃO é saída de GAN treinada."""
    if not modes or any(k < 0 or k >= NUM_MODES for k in modes):
        raise ValueError("modes deve conter índices válidos dos clusters")
    chosen = rng.choice(modes, size=n)
    return centers_numpy()[chosen] + rng.normal(0, STD, (n, 2)).astype(np.float32)


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Em Vanilla GAN, logits; em WGAN-GP, escore do crítico sem sigmoid."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()


@dataclass(frozen=True)
class ModeMetrics:
    coverage: int
    coverage_fraction: float
    valid_fraction: float
    js_divergence_bits: float
    occupancy: list[float]
    counts: list[int]
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "coverage_fraction": self.coverage_fraction,
            "valid_fraction": self.valid_fraction,
            "js_divergence_bits": self.js_divergence_bits,
            "occupancy": self.occupancy,
            "counts": self.counts,
            "n_samples": self.n_samples,
        }


def evaluate_modes(points: np.ndarray, threshold: float = EVAL_RADIUS,
                   min_mode_fraction: float = 0.01) -> ModeMetrics:
    """Métricas didáticas para a distribuição de 8 modos conhecida.

    Só atribui um modo às amostras a distância <= threshold do centro mais próximo.
    JS usa frequências APENAS das amostras válidas; ler junto com valid_fraction.
    Coverage é a quantidade de modos com >=1% do TOTAL de pontos gerados.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points deve ser uma matriz não vazia [n,2]")
    if threshold <= 0 or not (0 < min_mode_fraction <= 1):
        raise ValueError("limiares inválidos")
    dist = np.linalg.norm(points[:, None, :] - centers_numpy()[None, :, :], axis=2)
    nearest = dist.argmin(axis=1)
    valid = dist[np.arange(len(points)), nearest] <= threshold
    counts = np.bincount(nearest[valid], minlength=NUM_MODES)
    coverage = int(np.count_nonzero(counts / len(points) >= min_mode_fraction))
    nvalid = int(valid.sum())
    p = counts.astype(np.float64) / nvalid if nvalid else np.zeros(NUM_MODES)
    q = np.full(NUM_MODES, 1 / NUM_MODES)
    if nvalid:
        m = (p + q) / 2
        kl_p = np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
        kl_q = np.sum(q * np.log2(q / m))
        js = float((kl_p + kl_q) / 2)
    else:
        js = 1.0  # sem amostras válidas = pior valor convencional para o indicador
    return ModeMetrics(
        coverage=coverage,
        coverage_fraction=coverage / NUM_MODES,
        valid_fraction=nvalid / len(points),
        js_divergence_bits=js,
        occupancy=p.tolist(),
        counts=counts.astype(int).tolist(),
        n_samples=int(len(points)),
    )

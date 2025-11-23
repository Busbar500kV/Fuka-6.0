"""
Fuka-6.0 analysis: plots
=======================

Standard plotting suite used across experiments.

Includes:
- fitness / turbulence time series
- cluster id per slot
- token chain comparison (true vs decoded)
- PCA projection of attractor samples
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

from .cluster import pca_project

Array = np.ndarray


def plot_fitness(
    fitness_hist: Array,
    title: str = "Fitness F(t)",
    x_max: Optional[int] = None
) -> None:
    f = np.asarray(fitness_hist)
    if x_max is None:
        x_max = len(f)
    plt.figure(figsize=(9, 4))
    plt.plot(f[:x_max])
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("F")
    plt.tight_layout()
    plt.show()


def plot_turbulence(
    turbulence_hist: Array,
    title: str = "Turbulence",
    x_max: Optional[int] = None
) -> None:
    a = np.asarray(turbulence_hist)
    if x_max is None:
        x_max = len(a)
    plt.figure(figsize=(9, 4))
    plt.plot(a[:x_max])
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("mean(dV^2)")
    plt.tight_layout()
    plt.show()


def plot_cluster_ids_per_slot(
    slot_ids: Array,
    cluster_ids: Array,
    title: str = "Emergent attractor per slot"
) -> None:
    slot_ids = np.asarray(slot_ids)
    cluster_ids = np.asarray(cluster_ids)
    plt.figure(figsize=(9, 3.8))
    plt.plot(slot_ids, cluster_ids, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("slot index")
    plt.ylabel("cluster id")
    plt.tight_layout()
    plt.show()


def plot_token_chain(
    true_tokens: List[str],
    decoded_tokens: List[str],
    title: str = "Token chain: true vs decoded",
    max_len: int = 60
) -> None:
    K = min(max_len, len(true_tokens), len(decoded_tokens))
    enc = {t: i for i, t in enumerate(sorted(set(true_tokens + decoded_tokens)))}

    true_line = np.array([enc[t] for t in true_tokens[:K]])
    dec_line = np.array([enc[t] for t in decoded_tokens[:K]])

    x = np.arange(K)

    plt.figure(figsize=(9, 3.5))
    plt.step(x, true_line, where="mid", label="true")
    plt.step(x, dec_line, where="mid", linestyle="--", label="decoded")
    plt.title(title)
    plt.xlabel("token position")
    plt.ylabel("token")

    # y ticks using inverse map
    inv = {v: k for k, v in enc.items()}
    yticks = sorted(inv.keys())
    plt.yticks(yticks, [inv[y] for y in yticks])

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pca_samples(
    samples: Array,
    sample_labels: Optional[Array] = None,
    title: str = "Attractor samples (PCA)"
) -> None:
    samples = np.asarray(samples)
    proj, comps, mean = pca_project(samples, n_components=2)

    plt.figure(figsize=(6, 5))
    if sample_labels is None:
        plt.scatter(proj[:, 0], proj[:, 1])
    else:
        labels = np.asarray(sample_labels)
        for lab in np.unique(labels):
            pts = proj[labels == lab]
            plt.scatter(pts[:, 0], pts[:, 1], label=str(lab))
        plt.legend()

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
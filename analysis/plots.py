"""
Fuka-6.0
========

Common plotting utilities used by experiments.

Notes
-----
- Only uses matplotlib (no seaborn).
- Keep dependencies light: numpy + matplotlib + sklearn.decomposition.PCA.
- These helpers are for *analysis only* and are not part of the core model.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Helper: generic time-series plotting
# ---------------------------------------------------------------------


def _maybe_limit_x(x: np.ndarray, y: np.ndarray, x_max: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    If x_max is given, truncate x and y so that x <= x_max.
    """
    if x_max is None:
        return x, y
    mask = x <= x_max
    return x[mask], y[mask]


# ---------------------------------------------------------------------
# Fitness plotting
# ---------------------------------------------------------------------


def plot_fitness(fitness_hist: np.ndarray, title: str = "Fitness F(t)", x_max: Optional[int] = None) -> None:
    """
    Plot scalar fitness vs time.

    Parameters
    ----------
    fitness_hist : np.ndarray, shape (T,)
        Fitness at each time step.
    title : str
        Plot title.
    x_max : Optional[int]
        If provided, restrict x-axis to [0, x_max].
    """
    T = len(fitness_hist)
    t = np.arange(T, dtype=np.int32)

    t, f = _maybe_limit_x(t, fitness_hist, x_max)

    plt.figure(figsize=(9, 3.5))
    plt.plot(t, f)
    plt.xlabel("time step")
    plt.ylabel("fitness F(t)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Cluster IDs per slot
# ---------------------------------------------------------------------


def plot_cluster_ids_per_slot(
    slot_ids: Sequence[int],
    cluster_ids: Sequence[int],
    title: str = "Attractor clusters per slot",
) -> None:
    """
    Visualize which cluster (token ID) appears at each slot.

    slot_ids    : sequence of slot indices (integers)
    cluster_ids : sequence of cluster indices, same length as slot_ids
    """
    slot_ids = np.asarray(slot_ids, dtype=np.int32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)

    plt.figure(figsize=(10, 4))
    plt.scatter(slot_ids, cluster_ids, s=10)
    plt.xlabel("slot index")
    plt.ylabel("cluster ID")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# PCA visualization of attractor samples
# ---------------------------------------------------------------------


def plot_pca_samples(
    samples: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
    title: str = "Attractor samples (PCA projection)",
    annotate: bool = False,
) -> None:
    """
    Project high-dimensional attractor samples to 2D with PCA and scatter-plot.

    Parameters
    ----------
    samples : np.ndarray, shape (M, N)
        M samples in N-dimensional voltage space.
    sample_labels : Optional[np.ndarray], shape (M,)
        Optional integer labels (e.g., hidden regimes or cluster IDs).
    title : str
        Plot title.
    annotate : bool
        If True and labels are given, draw small text labels next to points.
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2D array of shape (M, N)")

    M = samples.shape[0]
    if M == 0:
        print("[plot_pca_samples] No samples to plot.")
        return

    pca = PCA(n_components=2)
    X = pca.fit_transform(samples)

    plt.figure(figsize=(6, 6))

    if sample_labels is None:
        plt.scatter(X[:, 0], X[:, 1], s=10)
    else:
        sample_labels = np.asarray(sample_labels)
        # To keep it simple, use labels as-is; matplotlib will choose colors.
        scatter = plt.scatter(X[:, 0], X[:, 1], s=10, c=sample_labels)
        cbar = plt.colorbar(scatter)
        cbar.set_label("label")

    if annotate and sample_labels is not None:
        for i in range(M):
            plt.text(X[i, 0], X[i, 1], str(sample_labels[i]), fontsize=6)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Environment & readout plotting
# ---------------------------------------------------------------------


def plot_environment(E_hist: np.ndarray, title: str = "Environment E(t)") -> None:
    """
    Plot scalar environment signal E(t).

    Parameters
    ----------
    E_hist : np.ndarray, shape (T,)
        Environment scalar state at each time step.
    title : str
        Plot title.
    """
    E_hist = np.asarray(E_hist, dtype=np.float64)
    T = len(E_hist)
    t = np.arange(T, dtype=np.int32)

    plt.figure(figsize=(9, 3.5))
    plt.plot(t, E_hist)
    plt.xlabel("time step")
    plt.ylabel("E(t)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_readout(readout_hist: np.ndarray, title: str = "Substrate readout") -> None:
    """
    Plot substrate->environment readout over time.

    Parameters
    ----------
    readout_hist : np.ndarray, shape (T,)
        Scalar readout from substrate (e.g., ||V||^2 or mean(V)).
    title : str
        Plot title.
    """
    readout_hist = np.asarray(readout_hist, dtype=np.float64)
    T = len(readout_hist)
    t = np.arange(T, dtype=np.int32)

    plt.figure(figsize=(9, 3.5))
    plt.plot(t, readout_hist)
    plt.xlabel("time step")
    plt.ylabel("readout")
    plt.title(title)
    plt.tight_layout()
    plt.show()
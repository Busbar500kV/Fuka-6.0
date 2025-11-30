"""
Fuka-6.0
plot_phenotype_grammar.py

Utility for Phase-6 phenotype runs.

Loads the latest:
    runs/exp_phenotype_fixed_*.npz

and produces two figures:

  1) images/phenotype_grammar_latest.png
       - directed graph over "core" clusters (frequent attractors)
       - edge thickness encodes P(j | i)
       - node size encodes cluster frequency

  2) images/phenotype_fingerprints_latest.png
       - PCA of mean voltage "fingerprints" for each core cluster

Core clusters are defined by a minimum cluster size (default 10).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def find_latest_npz(pattern: str) -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No NPZ files matching pattern: {pattern}")
    return paths[-1]


def load_phenotype_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def compute_core_clusters(
    cluster_sizes: np.ndarray, min_size: int
) -> np.ndarray:
    """Return array of cluster ids considered 'core'."""
    core = np.where(cluster_sizes >= min_size)[0]
    return core.astype(np.int32)


def transition_counts(ids: np.ndarray) -> np.ndarray:
    """
    Build full count matrix C[i,j] of transitions i->j
    between consecutive elements of ids.
    """
    k = int(ids.max()) + 1
    C = np.zeros((k, k), dtype=np.int32)
    for a, b in zip(ids[:-1], ids[1:]):
        C[a, b] += 1
    return C


def filter_core_edges(
    C: np.ndarray,
    core_ids: np.ndarray,
    prob_threshold: float = 0.10,
    min_count: int = 3,
) -> List[Tuple[int, int, float, int]]:
    """
    Return list of (i, j, P(j|i), count) for core->core edges
    that satisfy probability and count thresholds.
    """
    edges: List[Tuple[int, int, float, int]] = []

    core_set = set(int(c) for c in core_ids)
    total_out = C.sum(axis=1)

    for i in core_ids:
        out = int(total_out[i])
        if out == 0:
            continue
        for j in core_ids:
            count = int(C[i, j])
            if count < min_count:
                continue
            p = count / out
            if p >= prob_threshold:
                edges.append((int(i), int(j), float(p), count))

    # sort by probability *and* count (roughly importance)
    edges.sort(key=lambda e: (e[2], e[3]), reverse=True)
    return edges


# ---------------------------------------------------------------------
# Plot: grammar graph
# ---------------------------------------------------------------------


def plot_grammar_graph(
    core_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    edges: List[Tuple[int, int, float, int]],
    out_path: str,
) -> None:
    """
    Draw a simple circular layout for core clusters and
    directed edges with thickness proportional to P(j|i).
    """
    if len(core_ids) == 0:
        print("[plot_grammar_graph] No core clusters to plot.")
        return

    n_core = len(core_ids)
    angles = np.linspace(0.0, 2.0 * np.pi, n_core, endpoint=False)
    radius = 1.0

    pos_x = {}
    pos_y = {}
    for idx, cid in enumerate(core_ids):
        pos_x[int(cid)] = radius * np.cos(angles[idx])
        pos_y[int(cid)] = radius * np.sin(angles[idx])

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    if edges:
        max_p = max(e[2] for e in edges)
    else:
        max_p = 1.0

    for i, j, p, count in edges:
        x0, y0 = pos_x[i], pos_y[i]
        x1, y1 = pos_x[j], pos_y[j]

        # small offset for self-loops
        if i == j:
            # draw a little circle around the node
            loop_radius = 0.12
            theta = np.linspace(0, 2 * np.pi, 80)
            lx = x0 + loop_radius * np.cos(theta)
            ly = y0 + loop_radius * np.sin(theta)
            plt.plot(lx, ly, linewidth=1.0 + 3.0 * (p / max_p), alpha=0.7)
            continue

        width = 0.5 + 3.5 * (p / max_p)
        alpha = 0.35 + 0.55 * (p / max_p)

        # shorten arrows so they don't hit node centers
        shrink = 0.2
        dx = x1 - x0
        dy = y1 - y0
        dist = np.hypot(dx, dy) + 1e-6
        ux = dx / dist
        uy = dy / dist
        start_x = x0 + shrink * ux
        start_y = y0 + shrink * uy
        end_x = x1 - shrink * ux
        end_y = y1 - shrink * uy

        plt.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=width,
                alpha=alpha,
            ),
        )

    # Draw nodes
    core_sizes = cluster_sizes[core_ids]
    max_size = float(core_sizes.max()) if len(core_sizes) else 1.0
    # scale marker sizes for visibility
    marker_sizes = 400 * (0.3 + 0.7 * (core_sizes / max_size))

    for cid, ms in zip(core_ids, marker_sizes):
        x, y = pos_x[int(cid)], pos_y[int(cid)]
        plt.scatter([x], [y], s=ms, color="white", edgecolors="black", zorder=3)
        plt.text(
            x,
            y,
            f"T{cid}",
            ha="center",
            va="center",
            fontsize=9,
            zorder=4,
        )

    plt.title("Phenotype grammar over core attractor tokens")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[plot_grammar_graph] Saved: {out_path}")


# ---------------------------------------------------------------------
# Plot: attractor fingerprints (PCA)
# ---------------------------------------------------------------------


def pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Simple PCA to 2D using SVD.
    X: (M, D) matrix (M samples, D dimensions)

    Returns:
        Y: (M, 2) projected coordinates
    """
    X = np.asarray(X, dtype=np.float64)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # project onto first 2 principal components
    V2 = Vt[:2].T  # (D,2)
    Y = Xc @ V2    # (M,2)
    return Y.astype(np.float32)


def plot_attractor_fingerprints(
    attractor_samples: np.ndarray,
    attractor_id: np.ndarray,
    core_ids: np.ndarray,
    out_path: str,
) -> None:
    """
    For each core cluster:
        - compute mean voltage vector ("fingerprint")
        - run PCA on fingerprints
        - scatter plot with labels T<i>
    """
    if len(core_ids) == 0:
        print("[plot_attractor_fingerprints] No core clusters to plot.")
        return

    fingerprints = []
    used_ids = []
    for cid in core_ids:
        idx = np.where(attractor_id == cid)[0]
        if idx.size == 0:
            continue
        mean_vec = attractor_samples[idx].mean(axis=0)
        fingerprints.append(mean_vec)
        used_ids.append(int(cid))

    if not fingerprints:
        print("[plot_attractor_fingerprints] No samples found for core clusters.")
        return

    F = np.stack(fingerprints, axis=0)  # (M_core, N)
    Y = pca_2d(F)                       # (M_core, 2)

    plt.figure(figsize=(7, 6))
    plt.scatter(Y[:, 0], Y[:, 1], s=40, color="white", edgecolors="black")
    for (x, y, cid) in zip(Y[:, 0], Y[:, 1], used_ids):
        plt.text(x, y, f"T{cid}", ha="center", va="center", fontsize=9)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Core attractor fingerprints (PCA over mean voltage patterns)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[plot_attractor_fingerprints] Saved: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    pattern = "runs/exp_phenotype_fixed_*.npz"
    npz_path = find_latest_npz(pattern)
    print(f"Loading NPZ: {npz_path}")

    data = load_phenotype_npz(npz_path)

    attractor_samples = data["attractor_samples"]  # (S, N)
    attractor_id = data["attractor_id"]            # (S,)
    cluster_sizes = data["cluster_sizes"]          # (K,)

    core_threshold = 10
    core_ids = compute_core_clusters(cluster_sizes, core_threshold)
    print(f"Core threshold: {core_threshold}")
    print(f"Core clusters: {list(core_ids)}")

    # Grammar graph
    C = transition_counts(attractor_id)
    edges = filter_core_edges(
        C,
        core_ids,
        prob_threshold=0.10,
        min_count=3,
    )

    print("\nTop core->core edges (i->j, P, count):")
    for i, j, p, count in edges[:15]:
        print(f"  {i:2d} -> {j:2d}   P = {p:6.3f}   count = {count:4d}")

    grammar_path = "images/phenotype_grammar_latest.png"
    plot_grammar_graph(core_ids, cluster_sizes, edges, grammar_path)

    # Attractor fingerprints PCA
    fp_path = "images/phenotype_fingerprints_latest.png"
    plot_attractor_fingerprints(attractor_samples, attractor_id, core_ids, fp_path)


if __name__ == "__main__":
    main()
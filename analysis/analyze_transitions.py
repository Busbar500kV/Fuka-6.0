"""
Fuka-6.0 | Phenotype transition analysis

This script analyzes attractor transitions in a Phase-6 / Phase-6.1 phenotype run.

Features:
    - Loads the latest exp_phenotype_fixed_*.npz by default (or a user-specified file)
    - Identifies "core" clusters (frequent attractors) by a count threshold
    - Builds a transition matrix P(i -> j) over attractor_ids
    - Computes per-cluster transition entropy
    - Prints top core->core transitions (most "grammatical" edges)
    - Compares transitions in low-E vs high-E environment bands

Run (from repo root):

    venv/bin/python3 -m analysis.analyze_transitions

Optional args:

    venv/bin/python3 -m analysis.analyze_transitions --npz runs/exp_phenotype_fixed_20251129_230800.npz
    venv/bin/python3 -m analysis.analyze_transitions --core-threshold 15
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def find_latest_npz(pattern: str = "runs/exp_phenotype_fixed_*.npz") -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found matching pattern: {pattern}")
    return paths[-1]


def load_npz(path: str) -> np.lib.npyio.NpzFile:
    print(f"Loading NPZ: {path}")
    return np.load(path, allow_pickle=False)


def compute_core_clusters(cluster_sizes: np.ndarray, threshold: int) -> np.ndarray:
    """Return indices of clusters with count >= threshold."""
    return np.where(cluster_sizes >= threshold)[0]


def build_transition_matrix(attractor_ids: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Build count matrix T where T[i,j] = number of times we observed i -> j
    between consecutive attractor_ids.
    """
    T = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    src = attractor_ids[:-1]
    dst = attractor_ids[1:]
    for i, j in zip(src, dst):
        if 0 <= i < num_clusters and 0 <= j < num_clusters:
            T[i, j] += 1
    return T


def transition_probabilities(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert count matrix T into row-normalized probabilities P, and
    return (P, row_counts).
    """
    row_counts = T.sum(axis=1)
    P = np.zeros_like(T, dtype=np.float64)
    mask = row_counts > 0
    P[mask] = T[mask] / row_counts[mask, None]
    return P, row_counts


def transition_entropy(P: np.ndarray, row_counts: np.ndarray) -> np.ndarray:
    """
    Compute entropy H_i for each row i:
        H_i = -sum_j P_ij * log2(P_ij)
    Undefined rows (row_counts == 0) -> NaN.
    """
    H = np.full(P.shape[0], np.nan, dtype=np.float64)
    for i in range(P.shape[0]):
        if row_counts[i] == 0:
            continue
        p = P[i]
        m = p > 0
        H[i] = -np.sum(p[m] * np.log2(p[m]))
    return H


def top_core_transitions(
    T: np.ndarray,
    P: np.ndarray,
    core_ids: np.ndarray,
    top_k: int = 15,
) -> List[Tuple[float, int, int, int]]:
    """
    Return list of (prob, count, i, j) for top core->core transitions,
    sorted by decreasing prob (then count).
    """
    records: List[Tuple[float, int, int, int]] = []
    core_set = set(int(c) for c in core_ids)

    for i in core_ids:
        i = int(i)
        row_sum = T[i].sum()
        if row_sum == 0:
            continue
        for j in core_ids:
            j = int(j)
            c = int(T[i, j])
            if c <= 0:
                continue
            records.append((float(P[i, j]), c, i, j))

    records.sort(key=lambda x: (-x[0], -x[1]))
    return records[:top_k]


def build_env_conditioned_matrices(
    attractor_ids: np.ndarray,
    sample_times: np.ndarray,
    E_hist: np.ndarray,
    num_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two transition matrices:
        T_low  : transitions where both endpoints are in low-E half
        T_high : transitions where both endpoints are in high-E half
    Using median E(sample_times) as the split.

    Returns (T_low, T_high).
    """
    # E per sample
    E_samples = E_hist[sample_times]
    median_E = float(np.median(E_samples))
    print(f"\nEnvironment split:")
    print(f"  median E(sample) = {median_E:.3f}")

    low_mask = E_samples <= median_E
    high_mask = E_samples > median_E

    T_low = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    T_high = np.zeros((num_clusters, num_clusters), dtype=np.int64)

    # transitions between consecutive samples
    for idx in range(len(attractor_ids) - 1):
        src = int(attractor_ids[idx])
        dst = int(attractor_ids[idx + 1])

        src_low = low_mask[idx]
        dst_low = low_mask[idx + 1]
        src_high = high_mask[idx]
        dst_high = high_mask[idx + 1]

        if src_low and dst_low:
            if 0 <= src < num_clusters and 0 <= dst < num_clusters:
                T_low[src, dst] += 1
        if src_high and dst_high:
            if 0 <= src < num_clusters and 0 <= dst < num_clusters:
                T_high[src, dst] += 1

    print(f"  low-E transitions   : {int(T_low.sum())}")
    print(f"  high-E transitions  : {int(T_high.sum())}")

    return T_low, T_high


def print_top_conditional_transitions(
    name: str,
    T: np.ndarray,
    core_ids: np.ndarray,
    top_k: int = 10,
) -> None:
    """
    Print top core->core transitions for a conditional matrix T (e.g., low-E or high-E).
    """
    if T.sum() == 0:
        print(f"\n{name}: no transitions recorded.")
        return

    P, row_counts = transition_probabilities(T)
    records = top_core_transitions(T, P, core_ids, top_k=top_k)

    print(f"\n{name}: top core->core transitions (prob, count, i->j):")
    if not records:
        print("  (none)")
        return

    for prob, count, i, j in records:
        print(f"  P({i:3d}->{j:3d}) = {prob:6.3f}   count = {count:4d}")


# ---------------------------------------------------------------------
# Plotting helpers (optional visualization)
# ---------------------------------------------------------------------


def plot_core_transition_heatmap(
    P: np.ndarray,
    core_ids: np.ndarray,
    title: str = "Core-core transition probabilities",
) -> None:
    """
    Plot a small heatmap for transitions restricted to the core cluster IDs.
    """
    core_ids = np.array(core_ids, dtype=int)
    if core_ids.size == 0:
        print("No core clusters to plot.")
        return

    subP = P[np.ix_(core_ids, core_ids)]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(subP, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="P(i -> j)")
    plt.xticks(
        ticks=np.arange(len(core_ids)),
        labels=[str(c) for c in core_ids],
        rotation=45,
    )
    plt.yticks(
        ticks=np.arange(len(core_ids)),
        labels=[str(c) for c in core_ids],
    )
    plt.title(title)
    plt.xlabel("target cluster j")
    plt.ylabel("source cluster i")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attractor transitions in a Phase-6 phenotype run."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Path to exp_phenotype_fixed_*.npz (default: latest matching file).",
    )
    parser.add_argument(
        "--core-threshold",
        type=int,
        default=10,
        help="Minimum cluster size to be considered a core cluster.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots (text summary only).",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # Load NPZ
    # --------------------------------------------------
    if args.npz is None:
        npz_path = find_latest_npz()
    else:
        npz_path = args.npz

    data = load_npz(npz_path)

    # Basic required fields
    attractor_ids = data["attractor_id"].astype(np.int32)
    cluster_sizes = data["cluster_sizes"].astype(np.int32)
    E_hist = data["E_hist"].astype(np.float32)
    sample_times = data["sample_times"].astype(np.int32)

    num_samples = attractor_ids.shape[0]
    num_clusters = int(cluster_sizes.shape[0])

    print("\n--- Transition analysis summary ---")
    print(f"file            : {npz_path}")
    print(f"samples         : {num_samples}")
    print(f"num_clusters    : {num_clusters}")

    # --------------------------------------------------
    # Core clusters (frequency-based alphabet)
    # --------------------------------------------------
    core_ids = compute_core_clusters(cluster_sizes, threshold=args.core_threshold)
    core_counts = cluster_sizes[core_ids]
    core_order = np.argsort(core_counts)[::-1]
    core_ids = core_ids[core_order]
    core_counts = core_counts[core_order]

    core_coverage = core_counts.sum() / float(num_samples) if num_samples > 0 else 0.0

    print("\n--- Core clusters ---")
    print(f"core_threshold  : {args.core_threshold}")
    print(f"core_clusters   : {len(core_ids)}")
    print(f"core_coverage   : {core_coverage:.3f}")
    print("core_ids (sorted by frequency):")
    print("  ", core_ids.tolist())
    print("core_counts:")
    print("  ", core_counts.tolist())

    # --------------------------------------------------
    # Global transition matrix & entropy
    # --------------------------------------------------
    T = build_transition_matrix(attractor_ids, num_clusters=num_clusters)
    P, row_counts = transition_probabilities(T)
    H = transition_entropy(P, row_counts)

    print("\n--- Transition statistics (global) ---")
    print(f"total transitions  : {int(T.sum())}")
    print(f"states with outgoing transitions: {int(np.sum(row_counts > 0))}")

    # basic entropy stats over clusters with transitions
    valid_H = H[np.isfinite(H)]
    if valid_H.size > 0:
        print(f"entropy H(i->.) mean / std / min / max (bits): "
              f"{valid_H.mean():.3f} / {valid_H.std():.3f} / "
              f"{valid_H.min():.3f} / {valid_H.max():.3f}")
    else:
        print("entropy H(i->.) : no valid rows")

    # print entropies of core states
    print("\nCore cluster entropies (bits):")
    for cid, cnt in zip(core_ids, core_counts):
        h = H[int(cid)]
        h_str = f"{h:.3f}" if np.isfinite(h) else "NaN"
        print(f"  cluster {int(cid):3d}  count = {int(cnt):4d}   H = {h_str}")

    # --------------------------------------------------
    # Top core->core transitions (global)
    # --------------------------------------------------
    top_trans = top_core_transitions(T, P, core_ids, top_k=15)
    print("\nTop core->core transitions (global):")
    if not top_trans:
        print("  (none)")
    else:
        for prob, count, i, j in top_trans:
            print(f"  P({i:3d}->{j:3d}) = {prob:6.3f}   count = {count:4d}")

    # --------------------------------------------------
    # Environment-conditioned transitions (low-E vs high-E)
    # --------------------------------------------------
    T_low, T_high = build_env_conditioned_matrices(
        attractor_ids=attractor_ids,
        sample_times=sample_times,
        E_hist=E_hist,
        num_clusters=num_clusters,
    )

    print_top_conditional_transitions("Low-E band", T_low, core_ids, top_k=10)
    print_top_conditional_transitions("High-E band", T_high, core_ids, top_k=10)

    # --------------------------------------------------
    # Optional plots
    # --------------------------------------------------
    if not args.no_plots and len(core_ids) > 0:
        # Limit to at most first 12 core states for readability
        core_subset = core_ids[:12]
        plot_core_transition_heatmap(
            P,
            core_subset,
            title="Core-core transition probabilities (top 12 cores)",
        )


if __name__ == "__main__":
    main()
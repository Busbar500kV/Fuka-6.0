#!/usr/bin/env python3
"""
Fuka-6.0
Analysis helper for Phase-6 phenotype runs.

Usage examples (from repo root):

    # Analyze latest exp_phenotype_fixed run
    venv/bin/python3 -m analysis.analyze_phenotype_run

    # Or analyze a specific file
    venv/bin/python3 -m analysis.analyze_phenotype_run \
        --npz runs/exp_phenotype_fixed_20251129_230800.npz

This script:
    - Verifies NPZ safety (no pickles)
    - Summarizes token usage distribution
    - Identifies a "core alphabet" of frequent clusters
    - Prints simple metrics
    - Optionally shows a few basic plots
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Utility: find latest NPZ
# ---------------------------------------------------------

def find_latest_npz(pattern: str) -> Optional[str]:
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


# ---------------------------------------------------------
# Data model
# ---------------------------------------------------------

@dataclass
class PhenotypeRun:
    path: str
    attractor_ids: np.ndarray        # (S,)
    cluster_sizes: np.ndarray        # (C,)
    token_samples: np.ndarray        # (S,) unicode array
    E_hist: np.ndarray               # (T,)
    fitness_hist: Optional[np.ndarray]  # (T,) or None


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------

def load_phenotype_run(path: str) -> PhenotypeRun:
    """
    Load a phenotype NPZ saved by exp_phenotype (Phase-6.x).

    Assumes NPZ was created with safe, non-pickled arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}")

    print(f"Loading NPZ: {path}")
    d = np.load(path, allow_pickle=False)

    required_keys = [
        "attractor_id",
        "cluster_sizes",
        "unsupervised_token_samples",
        "E_hist",
    ]
    for k in required_keys:
        if k not in d:
            raise KeyError(f"Required key '{k}' missing in NPZ")

    attractor_ids = d["attractor_id"].astype(np.int32)
    cluster_sizes = d["cluster_sizes"].astype(np.int32)
    token_samples = d["unsupervised_token_samples"].astype("U8")
    E_hist = d["E_hist"].astype(np.float32)

    fitness_hist = d["fitness_hist"].astype(np.float32) if "fitness_hist" in d else None

    return PhenotypeRun(
        path=path,
        attractor_ids=attractor_ids,
        cluster_sizes=cluster_sizes,
        token_samples=token_samples,
        E_hist=E_hist,
        fitness_hist=fitness_hist,
    )


# ---------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------

@dataclass
class AlphabetStats:
    total_samples: int
    total_clusters: int
    core_threshold: int
    core_clusters: int
    core_coverage: float
    sizes_sorted: np.ndarray


def compute_alphabet_stats(run: PhenotypeRun, core_threshold: int = 10) -> AlphabetStats:
    sizes = np.asarray(run.cluster_sizes, dtype=np.int32)
    total_samples = int(sizes.sum())
    total_clusters = int(len(sizes))

    core_mask = sizes >= core_threshold
    core_clusters = int(core_mask.sum())
    core_coverage = float(sizes[core_mask].sum()) / float(total_samples) if total_samples > 0 else 0.0

    sizes_sorted = np.sort(sizes)[::-1]

    return AlphabetStats(
        total_samples=total_samples,
        total_clusters=total_clusters,
        core_threshold=core_threshold,
        core_clusters=core_clusters,
        core_coverage=core_coverage,
        sizes_sorted=sizes_sorted,
    )


def cumulative_coverage(sizes_sorted: np.ndarray, k: int) -> float:
    """
    Fraction of all samples accounted for by top-k clusters.
    """
    total = float(sizes_sorted.sum())
    if total <= 0.0:
        return 0.0
    k = min(k, len(sizes_sorted))
    return float(sizes_sorted[:k].sum()) / total


# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------

def plot_cluster_size_histogram(stats: AlphabetStats, max_bins: int = 50) -> None:
    sizes = stats.sizes_sorted
    plt.figure(figsize=(7, 4))
    plt.hist(sizes, bins=min(max_bins, len(sizes)))
    plt.title("Cluster size distribution (phenotype run)")
    plt.xlabel("cluster size")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_cumulative_coverage(stats: AlphabetStats, max_k: int = 50) -> None:
    sizes = stats.sizes_sorted
    max_k = min(max_k, len(sizes))
    ks = np.arange(1, max_k + 1)
    cov = [cumulative_coverage(sizes, int(k)) for k in ks]

    plt.figure(figsize=(7, 4))
    plt.plot(ks, cov)
    plt.title("Cumulative coverage of top-k clusters")
    plt.xlabel("k (number of largest clusters)")
    plt.ylabel("fraction of samples")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_environment(E_hist: np.ndarray) -> None:
    plt.figure(figsize=(9, 3.5))
    plt.plot(E_hist)
    plt.title("Environment scalar E(t)")
    plt.xlabel("t")
    plt.ylabel("E")
    plt.tight_layout()
    plt.show()


def plot_fitness(fitness_hist: Optional[np.ndarray]) -> None:
    if fitness_hist is None:
        print("No 'fitness_hist' found in NPZ, skipping fitness plot.")
        return
    plt.figure(figsize=(9, 3.5))
    plt.plot(fitness_hist)
    plt.title("Fitness F(t)")
    plt.xlabel("t")
    plt.ylabel("F")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase-6 phenotype NPZ run (Fuka-6.0)."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Path to NPZ file. If not provided, use latest 'runs/exp_phenotype_fixed_*.npz'.",
    )
    parser.add_argument(
        "--core-threshold",
        type=int,
        default=10,
        help="Minimum cluster size to be counted as part of the 'core alphabet'.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, only print text summary (no matplotlib windows).",
    )

    args = parser.parse_args()

    if args.npz is None:
        pattern = "runs/exp_phenotype_fixed_*.npz"
        latest = find_latest_npz(pattern)
        if latest is None:
            raise SystemExit(f"No NPZ found matching pattern: {pattern}")
        npz_path = latest
    else:
        npz_path = args.npz

    run = load_phenotype_run(npz_path)

    # Basic consistency check
    S = len(run.attractor_ids)
    if run.token_samples.shape[0] != S:
        print(f"[WARN] token_samples length {len(run.token_samples)} != attractor_ids length {S}")
    print(f"\n--- Phenotype run summary ---")
    print(f"file           : {npz_path}")
    print(f"samples        : {S}")
    print(f"clusters_found : {len(run.cluster_sizes)}")

    # Alphabet stats
    stats = compute_alphabet_stats(run, core_threshold=args.core_threshold)

    print(f"\n--- Alphabet statistics ---")
    print(f"total_samples       : {stats.total_samples}")
    print(f"total_clusters      : {stats.total_clusters}")
    print(f"core_threshold      : {stats.core_threshold}")
    print(f"core_clusters       : {stats.core_clusters}")
    print(f"core_coverage       : {stats.core_coverage:.3f}")

    # Coverage for some k values
    for k in [5, 10, 20, 50]:
        if k <= stats.total_clusters:
            cov = cumulative_coverage(stats.sizes_sorted, k)
            print(f"coverage top-{k:<3}: {cov:.3f}")

    # Top sizes preview
    top_show = min(15, len(stats.sizes_sorted))
    print(f"\nlargest cluster sizes (top {top_show}):")
    print(stats.sizes_sorted[:top_show])

    # Environment summary
    print(f"\n--- Environment E(t) ---")
    print(f"E_hist length   : {len(run.E_hist)}")
    print(f"E_min / E_max   : {run.E_hist.min():.3f} / {run.E_hist.max():.3f}")
    print(f"E_final         : {run.E_hist[-1]:.3f}")

    if not args.no_plots:
        plot_cluster_size_histogram(stats)
        plot_cumulative_coverage(stats)
        plot_environment(run.E_hist)
        plot_fitness(run.fitness_hist)


if __name__ == "__main__":
    main()
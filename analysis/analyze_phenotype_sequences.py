"""
Fuka-6.0
analysis/analyze_phenotype_sequences.py

Sequence-level analysis for Phase-6 (Phenotype loop).

What this script does
---------------------
1. Loads a phenotype NPZ (default: latest runs/exp_phenotype_fixed_*.npz)
2. Identifies "core" clusters (high-frequency attractors)
3. Builds compressed token sequences:
       full sequence of cluster_ids
       compressed sequence (collapse consecutive repeats)
4. Extracts n-gram statistics:
       - frequent core bigrams (i -> j)
       - frequent core trigrams (i -> j -> k)
   globally, and (if E_hist + sample_times available) separately
   for low-E and high-E bands.
5. Produces a few diagnostic plots.

Usage
-----
From repo root:

    python -m analysis.analyze_phenotype_sequences \
        [--npz runs/exp_phenotype_fixed_YYYYMMDD_HHMMSS.npz] \
        [--core-threshold 10] \
        [--top-k 20]

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_latest_npz(pattern: str = "runs/exp_phenotype_fixed_*.npz") -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No NPZ files matching pattern: {pattern}")
    return paths[-1]


def load_npz(path: str) -> Dict[str, np.ndarray]:
    print(f"Loading NPZ: {path}")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def get_core_clusters(cluster_sizes: np.ndarray,
                      core_threshold: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        core_ids: indices of clusters with size >= core_threshold
        order_by_size: cluster ids sorted by descending size
    """
    order = np.argsort(cluster_sizes)[::-1]
    core_ids = np.where(cluster_sizes >= core_threshold)[0]
    return core_ids, order


def compress_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Remove consecutive duplicates:
        [5,5,5,2,2,3] -> [5,2,3]
    """
    if len(seq) == 0:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return np.array(out, dtype=seq.dtype)


def compute_ngrams(seq: np.ndarray,
                   n: int,
                   core_set: Optional[set] = None) -> Counter:
    """
    Compute n-gram counts over integer sequence.

    If core_set is provided, only n-grams where all tokens in core_set
    are retained.
    """
    c = Counter()
    if len(seq) < n:
        return c
    for i in range(len(seq) - n + 1):
        ngram = tuple(seq[i:i + n])
        if core_set is not None:
            if not all(tok in core_set for tok in ngram):
                continue
        c[ngram] += 1
    return c


def describe_ngram(ngram: Tuple[int, ...],
                   cluster_sizes: np.ndarray,
                   label_prefix: str = "T") -> str:
    """
    Turn something like (12, 24, 3) into:
        "(12:T12, 24:T24, 3:T3)"
    """
    parts = []
    for idx in ngram:
        parts.append(f"{idx}:{label_prefix}{idx}")
    return "(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_cluster_sequence(cluster_ids: np.ndarray,
                          core_ids: np.ndarray,
                          title: str = "Cluster IDs over slots") -> None:
    """
    Simple line plot of cluster IDs, with core vs non-core separated.
    """
    x = np.arange(len(cluster_ids))
    core_mask = np.isin(cluster_ids, core_ids)
    noncore_mask = ~core_mask

    plt.figure(figsize=(10, 4))
    if noncore_mask.any():
        plt.scatter(x[noncore_mask], cluster_ids[noncore_mask],
                    s=6, alpha=0.5, label="non-core")
    if core_mask.any():
        plt.scatter(x[core_mask], cluster_ids[core_mask],
                    s=10, alpha=0.8, label="core")
    plt.xlabel("sample index (slot)")
    plt.ylabel("cluster_id")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_vs_core(E_samples: np.ndarray,
                        cluster_ids: np.ndarray,
                        core_ids: np.ndarray,
                        title: str = "Energy vs core tokens") -> None:
    """
    Scatter plot of energy vs sample index, with core vs non-core.
    """
    x = np.arange(len(cluster_ids))
    core_mask = np.isin(cluster_ids, core_ids)

    plt.figure(figsize=(10, 4))
    plt.plot(E_samples, label="E(sample)", linewidth=1.0)
    plt.scatter(x[core_mask], E_samples[core_mask],
                s=10, alpha=0.7, label="core token samples")
    plt.xlabel("sample index (slot)")
    plt.ylabel("E(sample)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze_sequences(npz_path: str,
                      core_threshold: int = 10,
                      top_k: int = 20) -> None:
    data = load_npz(npz_path)

    # Required fields
    if "attractor_id" not in data or "cluster_sizes" not in data:
        raise KeyError(
            "NPZ is missing 'attractor_id' or 'cluster_sizes'. "
            "Make sure you're using Phase-6 NPZ saved with the new schema."
        )

    cluster_ids = data["attractor_id"].astype(np.int32)        # shape [samples]
    cluster_sizes = data["cluster_sizes"].astype(np.int32)     # shape [num_clusters]

    # Optional fields
    sample_times = data.get("sample_times", None)
    E_hist = data.get("E_hist", None)

    print("\n=== Sequence-level analysis ===")
    print(f"file            : {npz_path}")
    print(f"samples         : {cluster_ids.shape[0]}")
    print(f"num_clusters    : {cluster_sizes.shape[0]}")
    print(f"core_threshold  : {core_threshold}")

    core_ids, order = get_core_clusters(cluster_sizes, core_threshold=core_threshold)
    core_set = set(core_ids.tolist())

    print(f"core_clusters   : {len(core_ids)}")
    print("largest clusters (top 12) [cluster_id:size]:")
    for cid in order[:12]:
        print(f"  {cid:4d} : {cluster_sizes[cid]}")

    # -------------------------------------------------
    # Build compressed sequence
    # -------------------------------------------------
    full_seq = cluster_ids
    comp_seq = compress_sequence(full_seq)

    print(f"\nSequence lengths:")
    print(f"  full      : {len(full_seq)}")
    print(f"  compressed: {len(comp_seq)}")

    # -------------------------------------------------
    # Global n-gram stats (core-only)
    # -------------------------------------------------
    for n in (2, 3):
        ngr = compute_ngrams(comp_seq, n=n, core_set=core_set)
        if not ngr:
            print(f"\nNo core-only {n}-grams found.")
            continue

        total = sum(ngr.values())
        print(f"\nGlobal core {n}-grams:")
        print(f"  distinct {n}-grams: {len(ngr)}")
        print(f"  total    {n}-gram count: {total}")

        for (ng, count) in ngr.most_common(top_k):
            prob = count / total
            desc = describe_ngram(ng, cluster_sizes)
            print(f"  {desc:40s}  count = {count:4d}   P = {prob:6.3f}")

    # -------------------------------------------------
    # Environment-dependent n-grams (if E_hist + sample_times)
    # -------------------------------------------------
    if E_hist is not None and sample_times is not None:
        E_hist = np.asarray(E_hist, dtype=np.float32)
        sample_times = np.asarray(sample_times, dtype=np.int64)

        if sample_times.max() >= len(E_hist):
            print("\nWarning: sample_times extends beyond E_hist; "
                  "skipping environment split.")
        else:
            E_samples = E_hist[sample_times]
            median_E = float(np.median(E_samples))
            low_mask = E_samples < median_E
            high_mask = ~low_mask

            print("\nEnvironment bands:")
            print(f"  median E(sample)   : {median_E:.3f}")
            print(f"  low-E samples      : {int(low_mask.sum())}")
            print(f"  high-E samples     : {int(high_mask.sum())}")

            # compressed sequences within each band (preserving order)
            low_seq = compress_sequence(full_seq[low_mask])
            high_seq = compress_sequence(full_seq[high_mask])

            for band_name, seq_band in (("low-E", low_seq), ("high-E", high_seq)):
                print(f"\n{band_name} core bigrams:")
                ngr2 = compute_ngrams(seq_band, n=2, core_set=core_set)
                if not ngr2:
                    print("  (none)")
                else:
                    total2 = sum(ngr2.values())
                    for (ng, count) in ngr2.most_common(top_k):
                        prob = count / total2
                        desc = describe_ngram(ng, cluster_sizes)
                        print(f"  {desc:40s}  count = {count:4d}   P = {prob:6.3f}")

    else:
        print("\nNo E_hist or sample_times found; skipping environment split.")

    # -------------------------------------------------
    # Plots (only if E_hist + sample_times are present)
    # -------------------------------------------------
    try:
        plot_cluster_sequence(cluster_ids, core_ids,
                              title="Cluster IDs over slots (core vs non-core)")

        if E_hist is not None and sample_times is not None:
            if sample_times.max() < len(E_hist):
                E_samples = E_hist[sample_times]
                plot_energy_vs_core(
                    E_samples,
                    cluster_ids,
                    core_ids,
                    title="Environment E(sample) and core token occurrences"
                )
    except Exception as e:
        print(f"\nPlotting failed (this won't affect analysis): {e}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sequence-level analysis for Fuka-6.0 Phase-6 phenotype runs."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Path to NPZ file (default: latest runs/exp_phenotype_fixed_*.npz).",
    )
    parser.add_argument(
        "--core-threshold",
        type=int,
        default=10,
        help="Minimum cluster size to be considered a 'core' token.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top n-grams to print for each category.",
    )

    args = parser.parse_args()

    if args.npz is None:
        npz_path = find_latest_npz()
    else:
        npz_path = args.npz

    analyze_sequences(
        npz_path=npz_path,
        core_threshold=args.core_threshold,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
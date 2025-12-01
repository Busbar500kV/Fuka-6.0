"""
Fuka-6.0: Long-run phenotype analysis

Usage (from repo root):

    venv/bin/python3 -m analysis.analyze_longrun_phenotype
    venv/bin/python3 -m analysis.analyze_longrun_phenotype runs/exp_longrun_phenotype_20251201_005507.npz

This script is deliberately defensive:
it inspects whatever is in the NPZ and produces:

  - Environment E(t) stats (length, min/max, time at saturation)
  - Fitness stats if present
  - Attractor / alphabet stats if long-run sampling was enabled
"""

from __future__ import annotations

import sys
import glob
import os
from typing import Optional, Tuple, Dict

import numpy as np


def _pick_latest(pattern: str) -> Optional[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    return paths[-1]


def load_npz(path: Optional[str]) -> Tuple[str, np.lib.npyio.NpzFile]:
    if path is None:
        # Default to latest longrun phenotype NPZ
        path = _pick_latest("runs/exp_longrun_phenotype_*.npz")
        if path is None:
            raise SystemExit("No files matching runs/exp_longrun_phenotype_*.npz")

    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    data = np.load(path, allow_pickle=False)
    return path, data


def summarize_environment(data: Dict[str, np.ndarray]) -> None:
    if "E_hist" not in data:
        print("\n[Environment] E_hist not found in NPZ.")
        return

    E = np.asarray(data["E_hist"], dtype=np.float64)
    n = len(E)
    print("\n--- Environment E(t) ---")
    print(f"E_hist length   : {n}")
    print(f"E_min / E_max   : {E.min():.3f} / {E.max():.3f}")
    print(f"E_mean / E_std  : {E.mean():.3f} / {E.std():.3f}")
    print("E_first 5       :", np.round(E[:5], 3))
    print("E_last  5       :", np.round(E[-5:], 3))

    # Heuristic: treat max(E) as saturation if it looks like a “hard” value
    e_max = E.max()
    if e_max > 0:
        near_max = np.mean(E > 0.95 * e_max)
        print(f"Fraction of time with E > 0.95*max (≈saturation): {near_max:.3f}")

    # Optional split: early vs late
    mid = n // 2
    if mid > 0:
        early = E[:mid]
        late = E[mid:]
        print(f"Early half:  mean={early.mean():.3f}, std={early.std():.3f}, min={early.min():.3f}, max={early.max():.3f}")
        print(f"Late  half:  mean={late.mean():.3f}, std={late.std():.3f}, min={late.min():.3f}, max={late.max():.3f}")


def summarize_fitness(data: Dict[str, np.ndarray]) -> None:
    if "fitness_hist" not in data:
        print("\n[Fitness] fitness_hist not found in NPZ.")
        return

    F = np.asarray(data["fitness_hist"], dtype=np.float64)
    if F.size == 0:
        print("\n[Fitness] fitness_hist present but empty.")
        return

    print("\n--- Fitness F(t) ---")
    print(f"length          : {len(F)}")
    print(f"F_min / F_max   : {F.min():.4f} / {F.max():.4f}")
    print(f"F_mean / F_std  : {F.mean():.4f} / {F.std():.4f}")
    print("F_first 5       :", np.round(F[:5], 4))
    print("F_last  5       :", np.round(F[-5:], 4))


def summarize_alphabet(data: Dict[str, np.ndarray]) -> None:
    """
    Long-run might (or might not) have attractor samples encoded like Phase-6:

        - attractor_id         (int32 [num_samples])
        - cluster_sizes        (int32 [num_clusters])
        - unsupervised_token_samples (str [num_samples])
        - sample_times         (int32 [num_samples])

    This function is tolerant: it prints what it can, or bails gracefully.
    """
    have_ids = "attractor_id" in data
    have_sizes = "cluster_sizes" in data

    if not have_ids and not have_sizes:
        print("\n[Alphabet] No attractor_id / cluster_sizes in NPZ (probably env-only long run).")
        return

    print("\n--- Attractor / alphabet stats (if present) ---")

    if have_ids:
        ids = np.asarray(data["attractor_id"], dtype=np.int64)
        print(f"num_samples      : {len(ids)}")
        unique, counts = np.unique(ids, return_counts=True)
        num_clusters = len(unique)
        print(f"clusters_seen    : {num_clusters}")

        # Sort cluster sizes descending
        order = np.argsort(counts)[::-1]
        unique_sorted = unique[order]
        counts_sorted = counts[order]

        top_k = min(15, num_clusters)
        print("largest cluster sizes (cluster_id : count) [top 15]:")
        for cid, c in zip(unique_sorted[:top_k], counts_sorted[:top_k]):
            print(f"  {int(cid):4d} : {int(c)}")

        # Core coverage (clusters with >=10 samples)
        core_mask = counts_sorted >= 10
        core_clusters = unique_sorted[core_mask]
        core_counts = counts_sorted[core_mask]
        if core_clusters.size > 0:
            coverage = core_counts.sum() / len(ids)
            print(f"core_threshold   : 10")
            print(f"core_clusters    : {len(core_clusters)}")
            print(f"core_coverage    : {coverage:.3f}")
        else:
            print("core_threshold   : 10 (no clusters above this)")

    if have_sizes:
        sizes = np.asarray(data["cluster_sizes"], dtype=np.int64)
        print(f"\ncluster_sizes array length: {len(sizes)}")
        if sizes.size > 0:
            top_sizes = np.sort(sizes)[::-1][:15]
            print("cluster_sizes (sorted, top 15):", top_sizes)


def main(argv: list[str]) -> None:
    path_arg = argv[1] if len(argv) > 1 else None
    path, npz = load_npz(path_arg)

    print(f"Loading NPZ: {path}")
    print("Available keys:", list(npz.keys()))

    data = {k: npz[k] for k in npz.files}

    summarize_environment(data)
    summarize_fitness(data)
    summarize_alphabet(data)

    print("\n[Done] Long-run phenotype summary complete.\n")


if __name__ == "__main__":
    main(sys.argv)
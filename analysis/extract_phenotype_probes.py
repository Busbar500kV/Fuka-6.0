"""
Fuka-6.0: Phenotype probe extractor

Given a phenotype NPZ (Phase-6 or long-run Phase 6-L), this script:

  1. Loads the attractor sequence and environment trajectory.
  2. Detects "avalanches" relative to a baseline cluster (the most
     frequent attractor ID).
  3. For each avalanche above a configurable size, extracts a high-
     resolution time window from V_hist and E_hist around the start
     of the avalanche.
  4. Stores each window as a separate NPZ under runs/probes/.

This gives you an offline "probe" mechanism: run experiments for days
or weeks, then zoom in on interesting episodes later.

Usage (from repo root):

    venv/bin/python3 -m analysis.extract_phenotype_probes \
        --npz runs/exp_longrun_phenotype_YYYYMMDD_HHMMSS.npz \
        --min_size 8 \
        --pre 800 \
        --post 2000

All arguments are optional; defaults target the latest NPZ.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np


@dataclass
class Avalanche:
    start_idx: int      # index in attractor sequence
    end_idx: int        # inclusive index in attractor sequence
    size: int           # number of non-baseline tokens
    baseline_id: int    # the baseline cluster ID


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def find_latest_npz(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files match pattern: {pattern}")
    return files[-1]


def detect_avalanches(
    seq: np.ndarray,
    baseline_id: int,
) -> List[Avalanche]:
    """
    Simple avalanche detection:

    - We assume seq is the sequence of attractor_id (cluster_ids) per slot.
    - baseline_id is the "rest" state (typically most frequent cluster).
    - An avalanche is a maximal contiguous block that:
        * starts at a non-baseline token,
        * ends when we first return to baseline,
        * size is the number of non-baseline tokens inside.
    """
    avalanches: List[Avalanche] = []
    n = len(seq)
    i = 0

    while i < n:
        if seq[i] == baseline_id:
            i += 1
            continue

        # Avalanche starts at i
        start = i
        size = 0
        while i < n and seq[i] != baseline_id:
            size += 1
            i += 1
        end = i - 1  # inclusive
        avalanches.append(Avalanche(start_idx=start, end_idx=end, size=size, baseline_id=baseline_id))

        # Next iteration continues from i (which is baseline or end)
    return avalanches


def choose_baseline_id(cluster_ids: np.ndarray) -> int:
    """
    Choose the baseline cluster as the ID with the highest frequency.
    This generalizes the 'token 5' heuristic used earlier.
    """
    ids, counts = np.unique(cluster_ids, return_counts=True)
    baseline = int(ids[np.argmax(counts)])
    return baseline


# ---------------------------------------------------------------------
# Main probe extractor
# ---------------------------------------------------------------------


def extract_probes(
    npz_path: str,
    min_size: int = 8,
    pre: int = 800,
    post: int = 2000,
) -> None:
    """
    Extract high-resolution probe windows around large avalanches.

    Args:
        npz_path: path to a phenotype or long-run NPZ.
        min_size: minimum avalanche size (in non-baseline tokens) to keep.
        pre:  number of time steps before avalanche start to include.
        post: number of time steps after avalanche start to include.
    """
    print(f"Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)

    # Required arrays
    V_hist = data["V_hist"]                 # (T, N)
    E_hist = data["E_hist"]                 # (T,)
    sample_times = data["sample_times"]     # (S,)
    attractor_ids = data["attractor_id"]    # (S,)

    T, N = V_hist.shape
    S = len(attractor_ids)
    print(f"V_hist shape   : {V_hist.shape}")
    print(f"E_hist length  : {len(E_hist)}")
    print(f"samples (slots): {S}")

    baseline_id = choose_baseline_id(attractor_ids)
    print(f"Baseline cluster ID (most frequent): {baseline_id}")

    avalanches = detect_avalanches(attractor_ids, baseline_id)
    print(f"Total avalanches detected: {len(avalanches)}")

    # Filter by size
    big_avalanches = [a for a in avalanches if a.size >= min_size]
    print(f"Avalanches with size >= {min_size}: {len(big_avalanches)}")

    if not big_avalanches:
        print("No avalanches above threshold; nothing to probe.")
        return

    # Sort by size descending for consistent naming
    big_avalanches.sort(key=lambda a: a.size, reverse=True)

    # Prepare output directory
    base_dir = os.path.dirname(npz_path)
    probes_dir = os.path.join(base_dir, "probes")
    os.makedirs(probes_dir, exist_ok=True)

    # For each avalanche, extract a window in raw time coordinates
    for idx, av in enumerate(big_avalanches):
        # Slot index of avalanche start -> raw time index
        first_slot = av.start_idx
        t0 = int(sample_times[first_slot])

        t_start = max(0, t0 - pre)
        t_end = min(T, t0 + post)

        # Window data
        V_window = V_hist[t_start:t_end]
        E_window = E_hist[t_start:t_end]
        time_indices = np.arange(t_start, t_end, dtype=np.int32)

        # Which sample indices fall in this window?
        within = np.where((sample_times >= t_start) & (sample_times < t_end))[0]
        sample_indices_in_window = within.astype(np.int32)
        attractor_window = attractor_ids[within].astype(np.int32)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        probe_name = f"probe_{idx:03d}_size{av.size}_from_slot{first_slot}_{stamp}.npz"
        probe_path = os.path.join(probes_dir, probe_name)

        np.savez_compressed(
            probe_path,
            V_window=V_window.astype(np.float32),
            E_window=E_window.astype(np.float32),
            time_indices=time_indices,
            sample_indices_in_window=sample_indices_in_window,
            attractor_id_window=attractor_window,
            baseline_id=np.array([baseline_id], dtype=np.int32),
            avalanche_start_slot=np.array([av.start_idx], dtype=np.int32),
            avalanche_end_slot=np.array([av.end_idx], dtype=np.int32),
            avalanche_size=np.array([av.size], dtype=np.int32),
        )

        print(
            f"  Saved probe {idx:03d}: size={av.size}, "
            f"slots [{av.start_idx}, {av.end_idx}], "
            f"time [{t_start}, {t_end}) -> {probe_path}"
        )

    print("\nProbe extraction complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract probe windows from a phenotype NPZ.")
    p.add_argument(
        "--npz",
        type=str,
        default="",
        help="Path to NPZ file. If empty, use latest exp_longrun_phenotype_*.npz or exp_phenotype_fixed_*.npz.",
    )
    p.add_argument(
        "--min_size",
        type=int,
        default=8,
        help="Minimum avalanche size (non-baseline tokens) to extract a probe.",
    )
    p.add_argument(
        "--pre",
        type=int,
        default=800,
        help="Number of time steps before avalanche start to include in the probe window.",
    )
    p.add_argument(
        "--post",
        type=int,
        default=2000,
        help="Number of time steps after avalanche start to include in the probe window.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.npz:
        npz_path = args.npz
    else:
        # Prefer long-run phenotype files; otherwise fall back to fixed Phase-6 runs
        try:
            npz_path = find_latest_npz("runs/exp_longrun_phenotype_*.npz")
        except FileNotFoundError:
            npz_path = find_latest_npz("runs/exp_phenotype_fixed_*.npz")

    extract_probes(
        npz_path=npz_path,
        min_size=args.min_size,
        pre=args.pre,
        post=args.post,
    )


if __name__ == "__main__":
    main()
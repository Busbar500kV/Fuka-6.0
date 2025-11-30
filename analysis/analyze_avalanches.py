# analysis/analyze_avalanches.py
"""
Fuka-6.0: Avalanche size analysis for phenotype runs.

We treat the most frequent attractor ID as a "baseline" state.
An avalanche is a maximal contiguous block of time steps where
attractor_id != baseline, bounded (when possible) by baseline on both sides.

We then:
  - compute the distribution of avalanche sizes
  - plot counts vs size on linear and log-log axes
  - fit a power-law-like line to the tail (sizes >= min_size_for_fit)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AvalancheConfig:
    # Minimum avalanche size to consider for fitting power law
    min_size_for_fit: int = 2
    # Optional max size for fitting (0 = no upper cap)
    max_size_for_fit: int = 0
    # Use latest NPZ automatically if path not provided
    npz_path: str | None = None


def find_latest_phenotype_npz() -> str:
    """Find latest exp_phenotype_fixed_*.npz in runs/."""
    paths = sorted(glob.glob("runs/exp_phenotype_fixed_*.npz"))
    if not paths:
        raise FileNotFoundError("No runs/exp_phenotype_fixed_*.npz found.")
    return paths[-1]


def extract_attractor_ids(npz_path: str) -> np.ndarray:
    """Load attractor_id from NPZ (no pickles)."""
    data = np.load(npz_path, allow_pickle=False)
    if "attractor_id" not in data:
        raise KeyError(f"'attractor_id' not in {npz_path} keys={list(data.keys())}")
    ids = data["attractor_id"]
    if ids.ndim != 1:
        raise ValueError(f"Expected 1D attractor_id, got shape {ids.shape}")
    return ids.astype(np.int32)


def detect_avalanches(attractor_id: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Detect avalanches relative to the most frequent baseline state.

    Returns:
        baseline_id, avalanche_sizes (1D int array)
    """
    # Most frequent ID as baseline
    unique, counts = np.unique(attractor_id, return_counts=True)
    baseline_id = unique[np.argmax(counts)]

    sizes: List[int] = []
    N = len(attractor_id)

    in_avalanche = False
    current_size = 0

    for t in range(N):
        if attractor_id[t] == baseline_id:
            # If we were in an avalanche, this closes it
            if in_avalanche and current_size > 0:
                sizes.append(current_size)
                current_size = 0
                in_avalanche = False
        else:
            # Not baseline -> part of avalanche
            in_avalanche = True
            current_size += 1

    # If sequence ends in the middle of an avalanche, we can:
    #  - either include it as a partial avalanche
    #  - or discard it; here we include it.
    if in_avalanche and current_size > 0:
        sizes.append(current_size)

    return int(baseline_id), np.array(sizes, dtype=np.int32)


def size_histogram(sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (unique_sizes, counts).
    """
    uniq, counts = np.unique(sizes, return_counts=True)
    return uniq, counts


def fit_power_law(
    uniq_sizes: np.ndarray,
    counts: np.ndarray,
    cfg: AvalancheConfig
) -> Tuple[float, float, int]:
    """
    Fit a line to log10(size) vs log10(count) for sizes >= cfg.min_size_for_fit
    and (optionally) <= cfg.max_size_for_fit.

    Returns (slope, intercept, num_points_used).
    If less than 2 points, returns (nan, nan, 0).
    """
    mask = uniq_sizes >= cfg.min_size_for_fit
    if cfg.max_size_for_fit > 0:
        mask &= uniq_sizes <= cfg.max_size_for_fit

    x = uniq_sizes[mask].astype(np.float64)
    y = counts[mask].astype(np.float64)

    if len(x) < 2:
        return np.nan, np.nan, 0

    log_x = np.log10(x)
    log_y = np.log10(y)

    # Fit log_y = a * log_x + b
    a, b = np.polyfit(log_x, log_y, 1)
    return float(a), float(b), len(x)


def plot_avalanches(
    uniq_sizes: np.ndarray,
    counts: np.ndarray,
    slope: float | None = None,
    intercept: float | None = None,
    npz_path: str | None = None,
    min_size_for_fit: int = 2,
) -> None:
    """
    Plot:
      1) linear scale: size vs count
      2) log-log scale: log10(size) vs log10(count) with optional fit line
    """
    title_suffix = ""
    if npz_path is not None:
        base = os.path.basename(npz_path)
        title_suffix = f"\n{base}"

    # Linear scale
    plt.figure(figsize=(7, 4))
    plt.bar(uniq_sizes, counts, width=0.8, align="center")
    plt.xlabel("Avalanche size (timesteps)")
    plt.ylabel("Count")
    plt.title("Avalanche size distribution" + title_suffix)
    plt.tight_layout()
    plt.show()

    # Log-log
    plt.figure(figsize=(7, 4))
    # Only plot points with count > 0 (they all should be)
    mask = counts > 0
    x = uniq_sizes[mask]
    y = counts[mask]

    plt.scatter(np.log10(x), np.log10(y), label="data", s=20)

    if slope is not None and intercept is not None and not np.isnan(slope):
        # Draw fit line over the min_size_for_fit .. max(x)
        x_fit = np.linspace(np.log10(min_size_for_fit), np.log10(x.max()), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, label=f"fit: slope={slope:.2f}", linewidth=2)

    plt.xlabel("log10(size)")
    plt.ylabel("log10(count)")
    plt.title("Avalanche size distribution (log-log)" + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    cfg = AvalancheConfig()

    if cfg.npz_path is None:
        npz_path = find_latest_phenotype_npz()
    else:
        npz_path = cfg.npz_path

    print(f"Loading NPZ: {npz_path}")
    ids = extract_attractor_ids(npz_path)
    baseline_id, sizes = detect_avalanches(ids)

    print("\n--- Avalanche summary ---")
    print(f"sequence length       : {len(ids)}")
    print(f"baseline attractor ID : {baseline_id}")
    print(f"total avalanches      : {len(sizes)}")
    if len(sizes) > 0:
        print(f"min / max size        : {sizes.min()} / {sizes.max()}")
        print(f"mean size             : {sizes.mean():.3f}")

    uniq_sizes, counts = size_histogram(sizes)

    print("\nAvalanche size histogram (size: count):")
    for s, c in zip(uniq_sizes, counts):
        print(f"  {int(s):4d} : {int(c)}")

    slope, intercept, npts = fit_power_law(uniq_sizes, counts, cfg)
    if npts >= 2:
        print("\n--- Power-law-like fit (log10 space) ---")
        print(f"points used   : {npts}")
        print(f"slope (alpha) : {slope:.3f}")
        print(f"intercept     : {intercept:.3f}")
        print("Note: For a perfect power-law, points would lie close to this line.")
    else:
        print("\nNot enough points to fit a power law tail.")

    # Plot with fit
    plot_avalanches(
        uniq_sizes,
        counts,
        slope=slope,
        intercept=intercept,
        npz_path=npz_path,
        min_size_for_fit=cfg.min_size_for_fit,
    )


if __name__ == "__main__":
    main()
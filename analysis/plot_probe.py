"""
Fuka-6.0: Probe visualizer

This script visualizes a single "probe" NPZ produced by
analysis.extract_phenotype_probes.

A probe file contains a high-resolution time window around a
large avalanche in the attractor sequence:

    V_window                : (T_window, N)
    E_window                : (T_window,)
    time_indices            : (T_window,)
    sample_indices_in_window: (K,)
    attractor_id_window     : (K,)
    baseline_id             : (1,)
    avalanche_start_slot    : (1,)
    avalanche_end_slot      : (1,)
    avalanche_size          : (1,)

The script plots:

  1. Environment E(t) over the window, with attractor samples shown
     as colored markers (one color per attractor ID in this probe).

  2. Global substrate activity: ||V(t)||_2 over the same window.

This gives an intuitive "microscope view" for each interesting
episode extracted by the probe tool.

Usage (from repo root):

    # Using the latest probe:
    venv/bin/python3 -m analysis.plot_probe

    # Or specify a particular probe:
    venv/bin/python3 -m analysis.plot_probe \
        --probe runs/probes/probe_000_size17_from_slot123_2025....npz
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def find_latest_probe(pattern: str = "runs/probes/probe_*.npz") -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No probe files match pattern: {pattern}\n"
            "Make sure you've run analysis.extract_phenotype_probes first."
        )
    return files[-1]


def load_probe(path: str) -> Tuple[dict, str]:
    data = np.load(path, allow_pickle=False)
    return data, path


def summarize_probe(data: dict, path: str) -> None:
    V_window = data["V_window"]              # (T_window, N)
    E_window = data["E_window"]              # (T_window,)
    time_indices = data["time_indices"]      # (T_window,)
    sample_indices = data["sample_indices_in_window"]  # (K,)
    attractor_ids = data["attractor_id_window"]        # (K,)

    baseline_id = int(data["baseline_id"][0])
    av_start_slot = int(data["avalanche_start_slot"][0])
    av_end_slot = int(data["avalanche_end_slot"][0])
    av_size = int(data["avalanche_size"][0])

    T_window, N = V_window.shape
    K = len(sample_indices)

    print(f"Probe file: {path}")
    print("------------------------------")
    print(f"V_window shape     : {V_window.shape}  (T_window, N)")
    print(f"E_window length    : {len(E_window)}")
    print(f"time_indices range : [{time_indices[0]}, {time_indices[-1]}]")
    print(f"samples in window  : {K}")
    print(f"baseline_id        : {baseline_id}")
    print(f"avalanche slots    : [{av_start_slot}, {av_end_slot}]")
    print(f"avalanche size     : {av_size}")
    print(f"unique attractors  : {np.unique(attractor_ids)}")
    print()


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_probe(data: dict, title_suffix: str = "") -> None:
    V_window = data["V_window"]              # (T_window, N)
    E_window = data["E_window"]              # (T_window,)
    time_indices = data["time_indices"]      # (T_window,)
    sample_indices = data["sample_indices_in_window"]  # (K,)
    attractor_ids = data["attractor_id_window"]        # (K,)

    baseline_id = int(data["baseline_id"][0])
    av_size = int(data["avalanche_size"][0])

    # Time axis (we'll just use raw step indices)
    t = time_indices.astype(np.int64)

    # Global activity norm ||V(t)||_2
    V_norm = np.linalg.norm(V_window.astype(np.float64), axis=1)

    # Prepare mapping from attractor IDs to colors
    unique_ids = np.unique(attractor_ids)
    cmap = plt.get_cmap("tab20")
    color_map = {
        int(aid): cmap(i % cmap.N) for i, aid in enumerate(unique_ids)
    }

    # Map sample_indices (indices in window) to t values
    sample_t = t[sample_indices]
    sample_colors = [color_map[int(a)] for a in attractor_ids]

    # -----------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"Probe window (avalanche size = {av_size}, baseline = {baseline_id})"
        + (f" — {title_suffix}" if title_suffix else "")
    )

    ax1, ax2 = axes

    # --- Top: E(t) and attractor markers ---
    ax1.plot(t, E_window, label="E(t)", linewidth=1.3)
    ax1.scatter(
        sample_t,
        E_window[sample_indices],
        c=sample_colors,
        s=24,
        edgecolors="none",
        alpha=0.9,
        label="attractor samples",
    )
    ax1.set_ylabel("E(t)")
    ax1.grid(True, alpha=0.3)

    # Build a small legend showing which color corresponds to which attractor
    # (only for core IDs in this probe)
    handles = []
    labels = []
    for aid in unique_ids:
        color = color_map[int(aid)]
        h = ax1.scatter([], [], c=[color], s=24, edgecolors="none")
        handles.append(h)
        labels.append(f"T{aid}")
    ax1.legend(handles, labels, title="Attractors in probe", fontsize=8, ncol=4)

    # --- Bottom: ||V(t)||_2 ---
    ax2.plot(t, V_norm, label="||V(t)||₂", linewidth=1.3)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("||V||₂")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a single probe NPZ.")
    p.add_argument(
        "--probe",
        type=str,
        default="",
        help="Path to a probe NPZ. If empty, uses latest runs/probes/probe_*.npz.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.probe:
        probe_path = args.probe
    else:
        probe_path = find_latest_probe()

    data, path = load_probe(probe_path)
    summarize_probe(data, path)
    plot_probe(data, title_suffix=os.path.basename(path))


if __name__ == "__main__":
    main()
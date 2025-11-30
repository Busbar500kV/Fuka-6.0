"""
Fuka-6.0 Experiment: Long-run Phenotype (Phase 6-L)

Goal:
    Run a larger capacitor substrate in the closed-loop phenotype
    environment for MUCH longer than Phase-6.1, while keeping the
    NPZ layout compatible with existing analysis tools.

    This is essentially Phase-6.1 with:
      - larger N
      - much larger total_steps
      - same environment dynamics
      - same attractor sampling & clustering pipeline

Run:
    ./tools/run_local.sh exp_longrun_phenotype

Output:
    runs/exp_longrun_phenotype_<timestamp>.npz

The NPZ is pickle-free (only plain numpy arrays), so it is safe to
load with allow_pickle=False.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation

# Reuse the closed-loop environment from Phase-6
from experiments.exp_phenotype import ClosedLoopEnvConfig, ClosedLoopEnvironment

from analysis.cluster import CosineClusterConfig, cluster_cosine_incremental
from analysis.decode import build_unsupervised_labels, decode_sequence
from analysis.plots import (
    plot_fitness,
    plot_cluster_ids_per_slot,
    plot_pca_samples,
    plot_environment,
    plot_readout,
)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:
    """
    Long-run phenotype experiment.

    Defaults are chosen to be "large but realistic". You can later
    crank N and total_steps up further if the machine can handle it.
    """

    # -------------------------
    # Configs
    # -------------------------

    # Slightly larger substrate than Phase-6.1
    substrate_cfg = SubstrateConfig(
        N=320,              # Phase-6 was 200; this is moderately larger
        C=1.0,
        lam=0.03,
        dt=0.05,
        init_v_std=0.01,
        init_g_scale=1e-3,
        symmetric_g=False,
        seed=0,
    )

    plasticity_cfg = PlasticityConfig(
        eta=0.002,
        alpha=0.02,
        dt=substrate_cfg.dt,
        use_fitness_gate=True,
        symmetric_g=substrate_cfg.symmetric_g,
        enable_create_prune=False,  # keep topology fixed for first long-run
        seed=0,
    )

    # Same environment configuration as Phase-6.1 (closed-loop)
    env_cfg = ClosedLoopEnvConfig(
        period=260,
        pulse_len=35,
        relax_len=110,
        regimes=3,
        nodes_per_regime=9,
        base_amp=0.8,
        noise_std=0.18,
        E_init=0.0,
        E_leak=0.008,       # Phase-6.1: slightly stronger pull to baseline
        E_drive=0.004,
        E_clip=2.0,
        feedback_gain=0.03,
        feedback_mode="energy",
        E_scale_factor=0.35,
    )

    slot_cfg = SlotConfig(pulse_len=env_cfg.pulse_len, relax_len=env_cfg.relax_len)
    env = ClosedLoopEnvironment(substrate_cfg.N, env_cfg, seed=5)

    # Long-run setting: more steps than Phase-6.1
    # Phase-6.1 used total_steps ~180k; here we go to 1e6 by default.
    # You can increase to 2e6 or 3e6 later if stable.
    run_cfg = RunConfig(
        total_steps=1_000_000,
        slot_period=env_cfg.period,
        record_V=True,        # so we can sample attractors
        record_dV=False,
        record_metrics=True,  # fitness / etc.
    )

    # -------------------------
    # Run simulation
    # -------------------------
    out: Dict[str, Any] = run_simulation(
        substrate_cfg=substrate_cfg,
        plasticity_cfg=plasticity_cfg,
        run_cfg=run_cfg,
        env_fn=env.env_fn,
        slot_cfg=slot_cfg,
        true_token_fn=None,  # unsupervised
        seed=0,
    )

    V_hist: np.ndarray = out["V_hist"]          # shape (T, N)
    fitness_hist: np.ndarray = out["fitness_hist"]

    E_hist = np.array(env.E_hist, dtype=np.float32)
    regime_hist = np.array(env.regime_hist, dtype=np.int32)
    readout_hist = np.array(env.readout_hist, dtype=np.float32)

    T, N = V_hist.shape
    period = env_cfg.period
    slots_total = T // period

    # -------------------------
    # Sample attractors per slot
    # -------------------------
    sample_times: List[int] = []
    samples: List[np.ndarray] = []
    regime_labels: List[int] = []

    # As in Phase-6.1: skip early transient slots
    skip_slots = 12
    for slot in range(skip_slots, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts < T:
            sample_times.append(ts)
            samples.append(V_hist[ts])
            regime_labels.append(env.regime_at_slot(slot))

    sample_times_arr = np.array(sample_times, dtype=np.int32)
    if samples:
        samples_arr = np.stack(samples, axis=0)
        regime_labels_arr = np.array(regime_labels, dtype=np.int32)
    else:
        # Extremely defensive, but keeps shapes defined even in degenerate case
        samples_arr = np.empty((0, N), dtype=np.float32)
        regime_labels_arr = np.empty((0,), dtype=np.int32)

    # -------------------------
    # Cluster emergent alphabet (unsupervised)
    # -------------------------
    cl_cfg = CosineClusterConfig(threshold=0.995)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples_arr, cl_cfg)

    labels_map = build_unsupervised_labels(cluster_ids, prefix="T")
    decoded = decode_sequence(cluster_ids, labels_map)

    # -------------------------
    # Print summary
    # -------------------------
    print("\nLong-run Phenotype results (Phase 6-L)")
    print("--------------------------------------")
    print(f"total_steps: {run_cfg.total_steps}")
    print(f"V_hist shape: {V_hist.shape}")
    print(f"samples: {len(samples_arr)}")
    print(f"clusters_found: {len(reps)}")
    print(f"cluster_sizes (top 15): {sizes[:15]}")
    print(f"Environment final E = {E_hist[-1]:.3f}")

    # -------------------------
    # Plots (optional, use locally with GUI)
    # -------------------------
    try:
        from matplotlib import pyplot as plt  # noqa: F401

        plot_fitness(fitness_hist, title="Fitness F(t) — long-run closed-loop", x_max=run_cfg.total_steps)
        plot_environment(E_hist, title="Environment scalar E(t) — long-run")
        plot_readout(readout_hist, title="Substrate readout driving E(t) — long-run")

        plot_cluster_ids_per_slot(
            slot_ids=(sample_times_arr // period),
            cluster_ids=cluster_ids,
            title="Emergent attractor clusters per slot (long-run closed-loop)",
        )

        plot_pca_samples(
            samples=samples_arr,
            sample_labels=regime_labels_arr,
            title="Attractor samples by hidden regime (PCA, long-run analysis)",
        )
    except Exception as exc:  # noqa: BLE001
        # On headless servers, plotting may fail; it's fine, NPZ is still saved.
        print(f"[WARN] Plotting failed: {exc!r}")

    # -------------------------
    # Save NPZ (pickle-free)
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_longrun_phenotype_{stamp}.npz"

    # Everything here is plain numpy arrays; no object dtype is used.
    np.savez_compressed(
        path,
        **out,
        sample_times=sample_times_arr,
        attractor_samples=samples_arr.astype(np.float32),
        attractor_id=cluster_ids.astype(np.int32),
        cluster_reps=reps.astype(np.float32),
        cluster_sizes=np.array(sizes, dtype=np.int32),
        hidden_regime_labels=regime_labels_arr.astype(np.int32),
        unsupervised_token_samples=np.array(decoded, dtype="U8"),
        E_hist=E_hist,
        regime_hist=regime_hist,
        substrate_readout_hist=readout_hist,
    )

    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
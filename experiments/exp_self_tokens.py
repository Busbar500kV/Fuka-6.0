"""
Fuka-6.0 Experiment: Phase 4
Self-generated attractor alphabet from noisy environment

Goal:
    Remove external labeled tokens (A/B/C).
    Drive the substrate with physical, continuous input waves.
    Show that it still self-organizes a finite attractor alphabet.

Environment:
    A small number of hidden regimes (not labeled to the substrate)
    that switch slowly. Each regime generates noisy wave input.

Outputs:
    - fitness plot
    - emergent cluster IDs per slot
    - PCA projection of attractor samples colored by regime (only for us)
    - unsupervised token chain (T0, T1, ...)

Run:
    python -m experiments.exp_self_tokens

NPZ:
    Saves to ./runs/exp_self_tokens_<timestamp>.npz
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation

from analysis.cluster import CosineClusterConfig, cluster_cosine_incremental
from analysis.decode import build_unsupervised_labels, decode_sequence
from analysis.plots import (
    plot_fitness,
    plot_cluster_ids_per_slot,
    plot_pca_samples
)


# ---------------------------------------------------------------------
# Noisy regime environment
# ---------------------------------------------------------------------

class RegimeNoiseEnvironment:
    """
    Environment with hidden regimes.

    Each regime defines:
      - a sparse spatial pattern of driven nodes
      - a mean amplitude
      - additive Gaussian noise

    The regime switches every 'regime_period' slots.
    """

    def __init__(
        self,
        N: int,
        regimes: int = 4,
        nodes_per_regime: int = 8,
        period: int = 260,
        pulse_len: int = 35,
        relax_len: int = 110,
        regime_period_slots: int = 8,
        base_amp: float = 0.7,
        noise_std: float = 0.25,
        seed: int = 3,
    ):
        self.N = N
        self.R = regimes
        self.period = period
        self.pulse_len = pulse_len
        self.relax_len = relax_len
        self.regime_period_slots = regime_period_slots
        self.base_amp = base_amp
        self.noise_std = noise_std

        self.rng = np.random.default_rng(seed)
        self.regime_nodes = [
            self.rng.choice(N, nodes_per_regime, replace=False)
            for _ in range(self.R)
        ]
        # slightly different mean amps per regime
        self.regime_amp = base_amp * (1.0 + 0.25 * self.rng.standard_normal(self.R))

        # also add a weak global sinusoid to make waves continuous
        self.global_freq = 2 * np.pi / (self.period * 6.0)

    def regime_at_slot(self, slot: int) -> int:
        return (slot // self.regime_period_slots) % self.R

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        slot = t // self.period
        r = self.regime_at_slot(slot)

        within = t % self.period
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        if within < self.pulse_len:
            I[self.regime_nodes[r]] = self.regime_amp[r]

        # Add continuous global wave + local noise always
        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15
        I += self.noise_std * self.rng.standard_normal(self.N)

        return I.astype(substrate.cfg.dtype)

    # Only for our analysis/plots (substrate never sees it)
    def regime_label_fn(self, t: int) -> int:
        slot = t // self.period
        return self.regime_at_slot(slot)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    # -------------------------
    # Configs
    # -------------------------
    substrate_cfg = SubstrateConfig(
        N=180,
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
        seed=0,
    )

    period = 260
    pulse_len = 35
    relax_len = 110

    slot_cfg = SlotConfig(pulse_len=pulse_len, relax_len=relax_len)

    env = RegimeNoiseEnvironment(
        N=substrate_cfg.N,
        regimes=4,
        nodes_per_regime=8,
        period=period,
        pulse_len=pulse_len,
        relax_len=relax_len,
        regime_period_slots=8,
        base_amp=0.7,
        noise_std=0.25,
        seed=3,
    )

    run_cfg = RunConfig(
        total_steps=42000,
        slot_period=period,
        record_V=True,
        record_dV=False,
        record_metrics=True,
    )

    # -------------------------
    # Run simulation
    # -------------------------
    out = run_simulation(
        substrate_cfg=substrate_cfg,
        plasticity_cfg=plasticity_cfg,
        run_cfg=run_cfg,
        env_fn=env.env_fn,
        slot_cfg=slot_cfg,
        true_token_fn=None,   # unsupervised
        seed=0
    )

    V_hist = out["V_hist"]
    fitness_hist = out["fitness_hist"]

    # -------------------------
    # Sample attractors per slot
    # -------------------------
    slots_total = run_cfg.total_steps // period
    sample_times = []
    samples = []
    regime_labels = []

    skip_slots = 12
    for slot in range(skip_slots, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts < run_cfg.total_steps:
            sample_times.append(ts)
            samples.append(V_hist[ts])
            regime_labels.append(env.regime_at_slot(slot))

    sample_times = np.array(sample_times, dtype=np.int32)
    samples = np.array(samples)
    regime_labels = np.array(regime_labels, dtype=np.int32)

    # -------------------------
    # Cluster emergent alphabet
    # -------------------------
    cl_cfg = CosineClusterConfig(threshold=0.995)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples, cl_cfg)

    labels_map = build_unsupervised_labels(cluster_ids, prefix="T")
    decoded = decode_sequence(cluster_ids, labels_map)

    print("\nPhase 4 results")
    print("----------------")
    print(f"samples: {len(samples)}")
    print(f"clusters_found: {len(reps)}")
    print(f"cluster_sizes: {sizes}")
    print(f"unsupervised_labels: {labels_map}")

    # -------------------------
    # Plots
    # -------------------------
    plot_fitness(fitness_hist, title="Fitness F(t) — noisy environment", x_max=8000)

    plot_cluster_ids_per_slot(
        slot_ids=(sample_times // period),
        cluster_ids=cluster_ids,
        title="Emergent attractor clusters per slot (unsupervised alphabet)"
    )

    plot_pca_samples(
        samples=samples,
        sample_labels=regime_labels,
        title="Attractor samples by hidden regime (PCA, for analysis only)"
    )

    # -------------------------
    # Save NPZ
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_self_tokens_{stamp}.npz"

    np.savez_compressed(
        path,
        **out,
        sample_times=sample_times,
        attractor_samples=samples.astype(np.float32),
        attractor_id=cluster_ids.astype(np.int32),
        cluster_reps=reps.astype(np.float32),
        cluster_sizes=np.array(sizes, dtype=np.int32),
        hidden_regime_labels=regime_labels.astype(np.int32),
        unsupervised_token_samples=np.array(decoded, dtype="U8"),
    )

    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
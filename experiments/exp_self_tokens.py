"""
Fuka-6.0 Experiment: Phase 4 (FIXED)
Self tokens (unsupervised alphabet emergence)

Problem in prior Phase-4:
    - Environment forcing was too noisy and too fast.
    - Relaxation window was too short.
    - Cosine threshold was too strict.
Result: every sample looked unique -> 1 cluster per sample.

Fix:
    - Use smooth, slowly-switching regimes.
    - Reduce noise strongly.
    - Increase relax_len and slot_period.
    - Use a more tolerant cosine clustering threshold.

Run:
    python -m experiments.exp_self_tokens

NPZ:
    Saves to ./runs/exp_self_tokens_fixed_<timestamp>.npz
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

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

from tools.safe_npz import savez_safe

# ---------------------------------------------------------------------
# Smooth unsupervised environment
# ---------------------------------------------------------------------

@dataclass
class SmoothRegimeEnvConfig:
    period: int = 320
    pulse_len: int = 28
    relax_len: int = 220

    regimes: int = 4
    nodes_per_regime: int = 9
    regime_period_slots: int = 10   # slow switching

    base_amp: float = 0.55
    noise_std: float = 0.05         # << much lower than before

    # slow, continuous background wave
    wave_amp: float = 0.12
    wave_period_slots: float = 24.0

    seed: int = 7


class SmoothRegimeEnvironment:
    """
    Unlabeled environment with hidden regimes that change slowly.
    Pulses are soft and noise is low so attractors can recur.
    """

    def __init__(self, N: int, cfg: SmoothRegimeEnvConfig):
        self.N = N
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.R = cfg.regimes
        self.regime_nodes = [
            self.rng.choice(N, cfg.nodes_per_regime, replace=False)
            for _ in range(self.R)
        ]

        # regime amplitudes slightly different but stable
        self.regime_amp = cfg.base_amp * (1.0 + 0.15 * self.rng.standard_normal(self.R))

        # slow wave in continuous time
        self.global_freq = 2 * np.pi / (cfg.period * cfg.wave_period_slots)

    def regime_at_slot(self, slot: int) -> int:
        return (slot // self.cfg.regime_period_slots) % self.R

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        slot = t // self.cfg.period
        r = self.regime_at_slot(slot)

        within = t % self.cfg.period
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        # soft pulse: ramp up / down to avoid sharp driving
        if within < self.cfg.pulse_len:
            # cosine ramp over the pulse window
            phase = within / max(1, self.cfg.pulse_len - 1)
            ramp = 0.5 - 0.5 * np.cos(np.pi * phase)
            I[self.regime_nodes[r]] = self.regime_amp[r] * ramp

        # slow background wave
        I += self.cfg.wave_amp * np.sin(self.global_freq * t)

        # small noise only
        I += self.cfg.noise_std * self.rng.standard_normal(self.N)

        return I.astype(substrate.cfg.dtype)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # -------------------------
    # Configs
    # -------------------------
    env_cfg = SmoothRegimeEnvConfig()

    substrate_cfg = SubstrateConfig(
        N=220,
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
        enable_create_prune=False,
        seed=0,
    )

    slot_cfg = SlotConfig(pulse_len=env_cfg.pulse_len, relax_len=env_cfg.relax_len)

    env = SmoothRegimeEnvironment(substrate_cfg.N, env_cfg)

    run_cfg = RunConfig(
        total_steps=68000,
        slot_period=env_cfg.period,
        record_V=True,
        record_dV=False,
        record_metrics=True,
    )

    # -------------------------
    # Run
    # -------------------------
    out = run_simulation(
        substrate_cfg=substrate_cfg,
        plasticity_cfg=plasticity_cfg,
        run_cfg=run_cfg,
        env_fn=env.env_fn,
        slot_cfg=slot_cfg,
        true_token_fn=None,   # fully unsupervised
        seed=0,
    )

    V_hist = out["V_hist"]
    fitness_hist = out["fitness_hist"]

    # -------------------------
    # Sample attractors per slot
    # -------------------------
    period = env_cfg.period
    slots_total = run_cfg.total_steps // period

    sample_times = []
    samples = []
    hidden_regimes = []

    skip_slots = 18  # allow hardware to settle first
    for slot in range(skip_slots, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts < run_cfg.total_steps:
            sample_times.append(ts)
            samples.append(V_hist[ts])
            hidden_regimes.append(env.regime_at_slot(slot))

    sample_times = np.array(sample_times, dtype=np.int32)
    samples = np.array(samples)
    hidden_regimes = np.array(hidden_regimes, dtype=np.int32)

    # -------------------------
    # Cluster emergent alphabet
    # -------------------------
    # more tolerant threshold than Phase 1–3
    cl_cfg = CosineClusterConfig(threshold=0.985)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples, cl_cfg)

    labels_map = build_unsupervised_labels(cluster_ids, prefix="T")
    decoded = decode_sequence(cluster_ids, labels_map)

    print("\nPhase 4 results (FIXED)")
    print("------------------------")
    print(f"samples: {len(samples)}")
    print(f"clusters_found: {len(reps)}")
    print(f"cluster_sizes: {sizes}")
    print(f"unsupervised_labels: {labels_map}")

    # -------------------------
    # Plots
    # -------------------------
    plot_fitness(fitness_hist, title="Fitness F(t) — self tokens (fixed)", x_max=12000)

    plot_cluster_ids_per_slot(
        slot_ids=(sample_times // period),
        cluster_ids=cluster_ids,
        title="Emergent self-tokens per slot (unsupervised)"
    )

    plot_pca_samples(
        samples=samples,
        sample_labels=hidden_regimes,
        title="Self-token attractors vs hidden regimes (PCA)"
    )

    # -------------------------
    # Save NPZ
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_self_tokens_fixed_{stamp}.npz"

    payload = dict(out)
    payload.update(
        sample_times=sample_times,
        attractor_samples=samples.astype(np.float32),
        attractor_id=cluster_ids.astype(np.int32),
        cluster_reps=reps.astype(np.float32),
        cluster_sizes=np.array(sizes, dtype=np.int32),
        hidden_regime_labels=regime_labels.astype(np.int32),
        unsupervised_token_samples=np.array(decoded, dtype="U8"),
        E_hist=E_hist,
        regime_hist=regime_hist,
        substrate_readout_hist=readout_hist,
    )

    savez_safe(path, payload)
    
    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
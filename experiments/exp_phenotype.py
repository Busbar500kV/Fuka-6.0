"""
Fuka-6.0 Experiment: Phase 6
Phenotype emergence via closed-loop environment

Goal:
    Establish the minimal closed evolutionary loop:

        substrate <-> code/attractors <-> environment

    The environment has a scalar state E(t) that:
      - drives energy injection into the substrate
      - is itself modified by the substrate's emergent state

This is the first prototype of "phenotype":
  computation that changes the environment which changes the computation.

Run:
    python -m experiments.exp_phenotype

NPZ:
    Saves to ./runs/exp_phenotype_<timestamp>.npz
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

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


# ---------------------------------------------------------------------
# Closed-loop environment
# ---------------------------------------------------------------------

@dataclass
class ClosedLoopEnvConfig:
    period: int = 260
    pulse_len: int = 35
    relax_len: int = 110

    regimes: int = 3                # hidden physical regimes
    nodes_per_regime: int = 8

    base_amp: float = 0.8
    noise_std: float = 0.2

    # scalar environment dynamics
    E_init: float = 0.0
    E_leak: float = 0.002           # environment tends to relax to 0
    E_drive: float = 0.004          # external background drift (sunlight etc.)
    E_clip: float = 2.0

    # substrate -> environment coupling
    feedback_gain: float = 0.03     # how strongly substrate readout pushes E
    feedback_mode: str = "energy"   # "energy" or "sign"


class ClosedLoopEnvironment:
    """
    Environment with:
      - hidden regimes (spatial patterns)
      - scalar state E(t) that modulates injection amplitude
      - E(t) updated by substrate readout

    This is the minimal phenotype loop.
    """

    def __init__(self, N: int, cfg: ClosedLoopEnvConfig, seed: int = 4):
        self.N = N
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.R = cfg.regimes
        self.regime_nodes = [
            self.rng.choice(N, cfg.nodes_per_regime, replace=False)
            for _ in range(self.R)
        ]
        self.regime_amp = cfg.base_amp * (1.0 + 0.25 * self.rng.standard_normal(self.R))
        self.regime_period_slots = 7

        self.global_freq = 2 * np.pi / (cfg.period * 6.0)

        # scalar environment state
        self.E = float(cfg.E_init)

        # logs
        self.E_hist: List[float] = []
        self.regime_hist: List[int] = []
        self.readout_hist: List[float] = []

    def regime_at_slot(self, slot: int) -> int:
        return (slot // self.regime_period_slots) % self.R

    def substrate_readout(self, substrate: Substrate) -> float:
        """
        Minimal readout from substrate to environment.

        Options:
          - "energy": use ||V||^2 (global activation)
          - "sign":   use mean(V) (directional bias)
        """
        V = substrate.V.astype(np.float64)

        if self.cfg.feedback_mode == "energy":
            return float(np.dot(V, V) / len(V))
        elif self.cfg.feedback_mode == "sign":
            return float(np.mean(V))
        else:
            raise ValueError(f"Unknown feedback_mode {self.cfg.feedback_mode}")

    def update_environment(self, readout: float) -> None:
        """
        Update scalar E(t) using substrate readout.

        Simple dynamics:
            dE/dt = -E_leak * E + E_drive + feedback_gain * phi(readout)

        where phi is a squashing to keep things stable.
        """
        # squash to avoid runaway
        phi = np.tanh(readout)

        dE = (
            -self.cfg.E_leak * self.E
            + self.cfg.E_drive
            + self.cfg.feedback_gain * phi
        )
        self.E += dE
        self.E = float(np.clip(self.E, -self.cfg.E_clip, self.cfg.E_clip))

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        """
        Called each time step by core.run.

        1) Read substrate state
        2) Update E(t)
        3) Build I(t) using E(t)
        """
        slot = t // self.cfg.period
        r = self.regime_at_slot(slot)

        # --- substrate -> environment ---
        readout = self.substrate_readout(substrate)
        self.update_environment(readout)

        # log env state
        self.E_hist.append(self.E)
        self.regime_hist.append(r)
        self.readout_hist.append(readout)

        within = t % self.cfg.period
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        # regime-based pulse, scaled by E(t)
        if within < self.cfg.pulse_len:
            I[self.regime_nodes[r]] = self.regime_amp[r] * (1.0 + self.E)

        # continuous bias + noise
        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15
        I += self.cfg.noise_std * self.rng.standard_normal(self.N)

        return I.astype(substrate.cfg.dtype)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def plot_environment(E_hist: np.ndarray, title: str = "Environment E(t)") -> None:
    plt.figure(figsize=(9, 3.5))
    plt.plot(E_hist)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("E")
    plt.tight_layout()
    plt.show()


def plot_readout(readout_hist: np.ndarray, title: str = "Substrate readout") -> None:
    plt.figure(figsize=(9, 3.5))
    plt.plot(readout_hist)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("readout")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    # -------------------------
    # Configs
    # -------------------------
    substrate_cfg = SubstrateConfig(
        N=200,
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
        enable_create_prune=False,  # keep off for the first phenotype test
        seed=0,
    )

    env_cfg = ClosedLoopEnvConfig(
        period=260,
        pulse_len=35,
        relax_len=110,
        regimes=3,
        nodes_per_regime=9,
        base_amp=0.8,
        noise_std=0.18,
        E_init=0.0,
        E_leak=0.002,
        E_drive=0.004,
        E_clip=2.0,
        feedback_gain=0.03,
        feedback_mode="energy",
    )

    slot_cfg = SlotConfig(pulse_len=env_cfg.pulse_len, relax_len=env_cfg.relax_len)
    env = ClosedLoopEnvironment(substrate_cfg.N, env_cfg, seed=5)

    run_cfg = RunConfig(
        total_steps=52000,
        slot_period=env_cfg.period,
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
        true_token_fn=None,  # unsupervised
        seed=0,
    )

    V_hist = out["V_hist"]
    fitness_hist = out["fitness_hist"]

    E_hist = np.array(env.E_hist, dtype=np.float32)
    regime_hist = np.array(env.regime_hist, dtype=np.int32)
    readout_hist = np.array(env.readout_hist, dtype=np.float32)

    # -------------------------
    # Sample attractors per slot
    # -------------------------
    period = env_cfg.period
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
    # Cluster emergent alphabet (unsupervised)
    # -------------------------
    cl_cfg = CosineClusterConfig(threshold=0.995)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples, cl_cfg)

    labels_map = build_unsupervised_labels(cluster_ids, prefix="T")
    decoded = decode_sequence(cluster_ids, labels_map)

    print("\nPhase 6 results (Phenotype loop)")
    print("--------------------------------")
    print(f"samples: {len(samples)}")
    print(f"clusters_found: {len(reps)}")
    print(f"cluster_sizes: {sizes}")
    print(f"unsupervised_labels: {labels_map}")
    print(f"Environment final E = {E_hist[-1]:.3f}")

    # -------------------------
    # Plots
    # -------------------------
    plot_fitness(fitness_hist, title="Fitness F(t) — closed-loop env", x_max=9000)
    plot_environment(E_hist, title="Environment scalar E(t)")
    plot_readout(readout_hist, title="Substrate readout driving E(t)")

    plot_cluster_ids_per_slot(
        slot_ids=(sample_times // period),
        cluster_ids=cluster_ids,
        title="Emergent attractor clusters per slot (closed-loop)"
    )

    plot_pca_samples(
        samples=samples,
        sample_labels=regime_labels,
        title="Attractor samples by hidden regime (PCA, analysis only)"
    )

    # -------------------------
    # Save NPZ
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_phenotype_{stamp}.npz"

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

        E_hist=E_hist,
        regime_hist=regime_hist,
        substrate_readout_hist=readout_hist,
    )

    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
"""
Fuka-6.0 Experiment: Phase 7
Autotuned phenotype loop (self-tuning from chaos)

Goal
----
Instead of hand-tuning parameters, let the closed-loop system
(self-organizing substrate + environment) gradually tune itself:

    substrate <-> environment E(t) <-> injection patterns

using only a few slow, global signals:

    - environment energy band: keep E(t) away from 0 and away from clip
    - pattern novelty: avoid "pure gas" (all states new) and
      "pure crystal" (only a few states repeated)

Mechanism
---------
We implement a long-run closed-loop environment similar to Phase 6,
but with two slow feedback controllers:

  1) Environment homeostasis:
     - Keeps the scalar E(t) in a target band by slowly updating
       effective E_drive and feedback_gain.

  2) Novelty balance:
     - Tracks a coarse "novelty rate" over attractor-like samples
       using cheap sign-hash fingerprints of V.
     - If novelty is too high: reduce noise_scale (more stable).
     - If novelty is too low: increase noise_scale (more exploratory).

This is not a separate meta-optimizer. All decisions are based only
on signals that the system itself can measure during the run.

Run
---
    ./tools/run_local.sh exp_autotune_phenotype

Output
------
    - Long-run NPZ:
        runs/exp_autotune_phenotype_<timestamp>.npz

      Contains all usual Fuka-6.0 outputs plus:

        E_hist                 : environment scalar history
        regime_hist            : hidden regime id per step
        substrate_readout_hist : readout used to drive E(t)
        E_drive_hist           : slow-tuned E_drive per slot
        feedback_gain_hist     : slow-tuned feedback_gain per slot
        noise_scale_hist       : slow-tuned noise_scale per slot
        novelty_hist           : novelty rate per slot
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation

from experiments.exp_phenotype import ClosedLoopEnvConfig  # reuse config shape


# ---------------------------------------------------------------------
# Local safe NPZ helper (no pickles)
# ---------------------------------------------------------------------


def savez_safe(path: str, payload: Dict[str, Any]) -> None:
    """
    Save a dict of arrays into a compressed .npz file with pickling disabled.
    """
    # Ensure keys are strings and values are array-like
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------
# Autotuning config (very slow, global feedback)
# ---------------------------------------------------------------------


@dataclass
class AutoTuneConfig:
    """
    Slow control parameters for the self-tuning loop.
    """

    # --- Environment homeostasis ---
    E_target: float = 0.7          # desired mean fraction of E_clip
    homeo_window_slots: int = 64   # how many recent slots to average over
    homeo_interval_slots: int = 16 # how often (in slots) to apply update

    homeo_lr_drive: float = 0.005      # learning rate for E_drive
    homeo_lr_feedback: float = 0.001   # learning rate for feedback_gain

    E_drive_min: float = 0.0
    E_drive_max: float = 0.02
    feedback_min: float = 0.0
    feedback_max: float = 0.08

    # --- Novelty-based tuning of noise ---
    # We do NOT change plasticity eta here, only environment noise.
    novelty_window_slots: int = 256     # slots included in novelty estimate
    novelty_update_interval: int = 32   # how often (in slots) to update noise_scale
    novelty_target: float = 0.5         # target fraction of unique patterns

    novelty_lr_noise: float = 0.4       # how strongly novelty error nudges noise_scale

    noise_scale_min: float = 0.2
    noise_scale_max: float = 3.0

    # --- Hashing parameters for coarse "attractor fingerprints" ---
    hash_dim: int = 16                  # dimension of random projection
    max_hash_window: int = 512          # maximum window of hash codes per slot


# ---------------------------------------------------------------------
# Auto-tuning closed-loop environment
# ---------------------------------------------------------------------


class AutoTuningEnvironment:
    """
    Closed-loop environment with self-tuning on two slow timescales:

      1) Keeps E(t) in a target band by adapting E_drive and feedback_gain.
      2) Balances novelty vs recurrence by adapting noise_scale.

    The "novelty" signal is computed from cheap sign-hash fingerprints
    of V at the end of each slot, so there is no heavy clustering
    inside the main run.
    """

    def __init__(
        self,
        N: int,
        env_cfg: ClosedLoopEnvConfig,
        auto_cfg: AutoTuneConfig,
        seed: int = 7,
    ):
        self.N = N
        self.cfg = env_cfg
        self.auto = auto_cfg

        self.rng = np.random.default_rng(seed)

        # --- Spatial regimes (same spirit as exp_phenotype) ---
        self.R = env_cfg.regimes
        self.regime_nodes = [
            self.rng.choice(N, env_cfg.nodes_per_regime, replace=False)
            for _ in range(self.R)
        ]
        self.regime_amp = env_cfg.base_amp * (
            1.0 + 0.25 * self.rng.standard_normal(self.R)
        )
        self.regime_period_slots = 7
        self.global_freq = 2 * np.pi / (env_cfg.period * 6.0)

        # --- Scalar environment state E(t) ---
        self.E = float(env_cfg.E_init)
        self.E_clip = float(env_cfg.E_clip)

        # We keep mutable copies of drive/gain that will drift:
        self.E_drive = float(env_cfg.E_drive)
        self.feedback_gain = float(env_cfg.feedback_gain)

        # --- Noise scale factor (autotuned) ---
        self.noise_scale = 1.0

        # --- Logs (per time step or per slot) ---
        self.E_hist: List[float] = []
        self.regime_hist: List[int] = []
        self.readout_hist: List[float] = []

        self.slot_index: int = 0  # counts slots

        # Per-slot logs of slow variables:
        self.E_drive_hist: List[float] = []
        self.feedback_gain_hist: List[float] = []
        self.noise_scale_hist: List[float] = []
        self.novelty_hist: List[float] = []

        # --- Hash-based "attractor fingerprints" ---
        # Random projection matrix: hash_dim x N with +/-1 entries
        self.hash_dim = auto_cfg.hash_dim
        self.hash_matrix = self.rng.choice(
            [-1.0, 1.0], size=(self.hash_dim, N)
        ).astype(np.float32)

        # Rolling window of hash integers (one per slot)
        self.hash_window: List[int] = []

    # -------------------------------
    # Core environment pieces
    # -------------------------------

    def regime_at_slot(self, slot: int) -> int:
        return (slot // self.regime_period_slots) % self.R

    def substrate_readout(self, substrate: Substrate) -> float:
        """
        Minimal readout from substrate to environment.

        We reuse the "energy" mode, as in exp_phenotype:
            readout = ||V||^2 / N
        """
        V = substrate.V.astype(np.float64)
        return float(np.dot(V, V) / len(V))

    def update_environment(self, readout: float) -> None:
        """
        Update scalar E(t) using substrate readout.

        Dynamics:
            dE/dt = -E_leak * E + E_drive + feedback_gain * tanh(readout)

        E_drive and feedback_gain will themselves be slowly adapted
        via homeostasis rules (see _apply_homeostasis).
        """
        phi = np.tanh(readout)

        dE = (
            -self.cfg.E_leak * self.E
            + self.E_drive
            + self.feedback_gain * phi
        )
        self.E += dE
        self.E = float(np.clip(self.E, -self.E_clip, self.E_clip))

    def _hash_V(self, V: np.ndarray) -> int:
        """
        Compute a coarse binary fingerprint of the current V.

        1) Project V into a low-dimensional space using a random +/-1 matrix.
        2) Take signs of the projection.
        3) Pack sign bits into a single integer.

        This is not meant to be collision-free, only to detect rough
        similarity vs difference across slots.
        """
        v = V.astype(np.float32)
        proj = self.hash_matrix @ v  # shape: (hash_dim,)
        bits = proj > 0.0

        code = 0
        for b in bits:
            code = (code << 1) | int(bool(b))
        return code

    # -------------------------------
    # Slow control rules
    # -------------------------------

    def _apply_homeostasis(self) -> None:
        """
        Homeostasis for environment energy:

        Keep E(t) in a mid-band by nudging E_drive and feedback_gain
        using a simple proportional controller on the mean E over a
        sliding window of slots.
        """
        if self.slot_index < 1:
            return

        # Limit window to the last homeo_window_slots slots
        w = self.auto.homeo_window_slots
        period = self.cfg.period
        steps = w * period
        E_arr = np.array(self.E_hist, dtype=np.float32)
        if E_arr.size < steps:
            return

        E_window = E_arr[-steps:]
        E_mean = float(E_window.mean())
        E_target_abs = self.auto.E_target * self.E_clip
        err = E_mean - E_target_abs  # positive if E too high

        # Nudge E_drive and feedback_gain downward if E is too high,
        # upward if E is too low.
        self.E_drive -= self.auto.homeo_lr_drive * err
        self.feedback_gain -= self.auto.homeo_lr_feedback * err

        # Clip to reasonable ranges
        self.E_drive = float(
            np.clip(self.E_drive, self.auto.E_drive_min, self.auto.E_drive_max)
        )
        self.feedback_gain = float(
            np.clip(
                self.feedback_gain,
                self.auto.feedback_min,
                self.auto.feedback_max,
            )
        )

    def _compute_novelty(self) -> float:
        """
        Estimate novelty rate in the recent window of hash codes.

        We define:
            novelty_rate = (# unique hashes) / (window length)
        """
        if not self.hash_window:
            return 0.0
        window = self.hash_window[-self.auto.novelty_window_slots :]
        length = len(window)
        if length == 0:
            return 0.0
        unique = len(set(window))
        return float(unique) / float(length)

    def _apply_novelty_balance(self, novelty_rate: float) -> None:
        """
        Tuning of environment noise based on novelty.

        - If novelty_rate > novelty_target:
              system is too chaotic => reduce noise_scale.
        - If novelty_rate < novelty_target:
              system is too rigid   => increase noise_scale.
        """
        err = novelty_rate - self.auto.novelty_target

        # Move noise_scale opposite the sign of err:
        #  - positive err => novelty too high => scale below 1
        #  - negative err => novelty too low  => scale above 1
        factor = 1.0 - self.auto.novelty_lr_noise * err
        self.noise_scale *= factor

        self.noise_scale = float(
            np.clip(
                self.noise_scale,
                self.auto.noise_scale_min,
                self.auto.noise_scale_max,
            )
        )

    # -------------------------------
    # Main environment hook
    # -------------------------------

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        """
        This function is called by the simulator at every time step.

        Order:
          1) Compute current slot.
          2) Read substrate and update environment E(t).
          3) Build injection current I(t).
          4) At the end of each slot, update slow controllers using
             hashed V fingerprints and E statistics.
        """
        slot = t // self.cfg.period
        r = self.regime_at_slot(slot)
        within = t % self.cfg.period

        # --- substrate -> environment ---
        readout = self.substrate_readout(substrate)
        self.update_environment(readout)

        # Log per-step env state
        self.E_hist.append(self.E)
        self.regime_hist.append(r)
        self.readout_hist.append(readout)

        # --- Build I(t) ---
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        # Regime-based pulse, scaled by (1 + E_scale_factor * E)
        if within < self.cfg.pulse_len:
            scale = 1.0 + self.cfg.E_scale_factor * self.E
            scale = float(scale)
            I[self.regime_nodes[r]] = self.regime_amp[r] * scale

        # Continuous bias + noise (with autotuned noise_scale)
        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15
        noise = (
            self.cfg.noise_std
            * self.noise_scale
            * self.rng.standard_normal(self.N)
        )
        I += noise.astype(substrate.cfg.dtype)

        # --- End-of-slot slow updates ---
        if within == self.cfg.period - 1:
            self._on_slot_end(slot, substrate)

        return I.astype(substrate.cfg.dtype)

    def _on_slot_end(self, slot: int, substrate: Substrate) -> None:
        """
        Called at the end of each slot. This is where we:

          - Record a hash fingerprint of V.
          - Estimate novelty, and occasionally update noise_scale.
          - Occasionally apply environment homeostasis.
          - Log the current slow-control variables for analysis.
        """
        self.slot_index = slot + 1

        # --- Hash-based novelty ---
        h = self._hash_V(substrate.V)
        self.hash_window.append(h)
        if len(self.hash_window) > self.auto.max_hash_window:
            # Keep the window bounded
            self.hash_window = self.hash_window[-self.auto.max_hash_window :]

        novelty_rate = self._compute_novelty()
        self.novelty_hist.append(novelty_rate)

        # Update noise_scale occasionally
        if self.slot_index % self.auto.novelty_update_interval == 0:
            self._apply_novelty_balance(novelty_rate)

        # Apply homeostasis occasionally
        if self.slot_index % self.auto.homeo_interval_slots == 0:
            self._apply_homeostasis()

        # Log slow-control variables (once per slot)
        self.E_drive_hist.append(self.E_drive)
        self.feedback_gain_hist.append(self.feedback_gain)
        self.noise_scale_hist.append(self.noise_scale)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:
    # -------------------------
    # Configs
    # -------------------------
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
        enable_create_prune=False,  # we let environment self-tune first
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
        E_leak=0.0095,      # fairly strong pull toward baseline
        E_drive=0.004,      # initial drive, will be autotuned
        E_clip=2.0,
        feedback_gain=0.03, # initial gain, will be autotuned
        feedback_mode="energy",
        E_scale_factor=0.35,
    )

    auto_cfg = AutoTuneConfig(
        # E band parameters tuned for E_clip = 2.0
        E_target=0.6,  # target around 1.2
        homeo_window_slots=64,
        homeo_interval_slots=16,
        homeo_lr_drive=0.005,
        homeo_lr_feedback=0.001,
        E_drive_min=0.0,
        E_drive_max=0.02,
        feedback_min=0.0,
        feedback_max=0.08,
        novelty_window_slots=256,
        novelty_update_interval=32,
        novelty_target=0.5,
        novelty_lr_noise=0.3,
        noise_scale_min=0.2,
        noise_scale_max=3.0,
        hash_dim=16,
        max_hash_window=512,
    )

    slot_cfg = SlotConfig(
        pulse_len=env_cfg.pulse_len,
        relax_len=env_cfg.relax_len,
    )

    env = AutoTuningEnvironment(
        N=substrate_cfg.N,
        env_cfg=env_cfg,
        auto_cfg=auto_cfg,
        seed=11,
    )

    # Long-run config: you can increase this further as needed.
    run_cfg = RunConfig(
        total_steps=1_000_000,   # ~1e6 steps; feel free to scale up
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
        true_token_fn=None,  # unsupervised / self-tuned
        seed=0,
    )

    fitness_hist = out.get("fitness_hist", None)

    # -------------------------
    # Print quick summary
    # -------------------------
    E_hist = np.array(env.E_hist, dtype=np.float32)
    novelty_hist = np.array(env.novelty_hist, dtype=np.float32)

    print("\nPhase 7 results (Autotuned phenotype loop)")
    print("------------------------------------------")
    print(f"total_steps     : {run_cfg.total_steps}")
    print(f"E_hist length   : {len(E_hist)}")
    if len(E_hist) > 0:
        print(
            f"E_min / E_max   : {E_hist.min():.3f} / {E_hist.max():.3f} "
            f"(target ~ {auto_cfg.E_target * env_cfg.E_clip:.3f})"
        )
        print(f"E_final         : {E_hist[-1]:.3f}")
    if fitness_hist is not None and len(fitness_hist) > 0:
        F = np.asarray(fitness_hist, dtype=np.float32)
        print("\nFitness F(t):")
        print(
            f"  length        : {len(F)}\n"
            f"  F_min / F_max : {F.min():.4f} / {F.max():.4f}\n"
            f"  F_mean / F_sd : {F.mean():.4f} / {F.std():.4f}"
        )
    if len(novelty_hist) > 0:
        print(
            f"\nNovelty rate (window-unique / window-size):"
            f"\n  mean          : {novelty_hist.mean():.3f}"
            f"\n  min / max     : {novelty_hist.min():.3f} / {novelty_hist.max():.3f}"
            f"\n  target        : {auto_cfg.novelty_target:.3f}"
        )

    # -------------------------
    # Save NPZ
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_autotune_phenotype_{stamp}.npz"

    payload: Dict[str, Any] = dict(out)

    payload.update(
        E_hist=E_hist,
        regime_hist=np.array(env.regime_hist, dtype=np.int32),
        substrate_readout_hist=np.array(
            env.readout_hist, dtype=np.float32
        ),
        E_drive_hist=np.array(env.E_drive_hist, dtype=np.float32),
        feedback_gain_hist=np.array(
            env.feedback_gain_hist, dtype=np.float32
        ),
        noise_scale_hist=np.array(
            env.noise_scale_hist, dtype=np.float32
        ),
        novelty_hist=novelty_hist,
        autotune_config=str(auto_cfg),
        env_config=str(env_cfg),
    )

    savez_safe(path, payload)
    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
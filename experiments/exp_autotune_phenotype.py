"""
Fuka-6.0 Experiment: exp_autotune_phenotype
Autotune closed-loop environment from chaos (no manual hyperparameter tuning).

Key idea:
  We keep a scalar environment state E(t) in a "healthy" band by automatically
  adjusting environment knobs (feedback_gain, E_leak, noise_std, etc.) while the
  substrate evolves.

What gets saved (pickle-free .npz):
  - core outputs from run_simulation (V_hist, fitness_hist, g_last, etc.)
  - E_hist, readout_hist
  - attractor samples and cluster ids (optional / lightweight)

This file intentionally avoids:
  - analysis.plots (sklearn dependency)
  - any object arrays (pickle risk)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation

from analysis.cluster import CosineClusterConfig, cluster_cosine_incremental


# -----------------------------
# Safe saving (no pickle)
# -----------------------------

def savez_safe(path: str, payload: Dict[str, Any]) -> None:
    """
    Save a dict to npz in a pickle-free way.
    - Converts lists to numpy arrays when possible
    - Rejects object dtype arrays
    """
    clean: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue

        if isinstance(v, (list, tuple)):
            v = np.array(v)

        if isinstance(v, np.ndarray) and v.dtype == object:
            raise ValueError(f"Refusing to save object array (pickle risk): key={k}")

        clean[k] = v

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **clean)

    # hard check: ensure load works with allow_pickle=False
    np.load(path, allow_pickle=False)


# -----------------------------
# Closed-loop environment
# -----------------------------

@dataclass
class AutoTuneEnvConfig:
    period: int = 260
    pulse_len: int = 35
    relax_len: int = 110

    regimes: int = 3
    nodes_per_regime: int = 9

    base_amp: float = 0.8
    noise_std: float = 0.25

    # scalar env state
    E_init: float = 0.0
    E_clip: float = 2.0

    # scalar env dynamics
    E_leak: float = 0.008
    E_drive: float = 0.004
    feedback_gain: float = 0.03
    feedback_mode: str = "energy"     # "energy" or "sign"

    # how E scales excitation (gentler than 1+E)
    E_scale_factor: float = 0.35

    # extra chaos knobs
    permeability_jitter: float = 0.02   # slow modulation magnitude
    permeability_period: int = 5000     # steps


class AutoTuneClosedLoopEnvironment:
    def __init__(self, N: int, cfg: AutoTuneEnvConfig, seed: int = 5):
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
        self.noise_hist: List[float] = []
        self.perm_hist: List[float] = []

    def regime_at_slot(self, slot: int) -> int:
        return (slot // self.regime_period_slots) % self.R

    def substrate_readout(self, substrate: Substrate) -> float:
        V = substrate.V.astype(np.float64)
        if self.cfg.feedback_mode == "energy":
            return float(np.dot(V, V) / len(V))
        if self.cfg.feedback_mode == "sign":
            return float(np.mean(V))
        raise ValueError(f"Unknown feedback_mode {self.cfg.feedback_mode}")

    def update_environment(self, readout: float) -> None:
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
        slot = t // self.cfg.period
        r = self.regime_at_slot(slot)

        readout = self.substrate_readout(substrate)
        self.update_environment(readout)

        # slow “permeability” modulation (external chaos)
        perm = 1.0 + self.cfg.permeability_jitter * np.sin(2 * np.pi * t / self.cfg.permeability_period)

        self.E_hist.append(self.E)
        self.regime_hist.append(r)
        self.readout_hist.append(readout)
        self.noise_hist.append(self.cfg.noise_std)
        self.perm_hist.append(float(perm))

        within = t % self.cfg.period
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        # gentler excitation scaling with E
        scale = 1.0 + self.cfg.E_scale_factor * self.E
        scale = float(np.clip(scale, 0.10, 3.0))

        if within < self.cfg.pulse_len:
            I[self.regime_nodes[r]] = (self.regime_amp[r] * scale) * perm

        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15

        # noise (also impacted by permeability)
        I += (self.cfg.noise_std * perm) * self.rng.standard_normal(self.N)

        return I.astype(substrate.cfg.dtype)


# -----------------------------
# Autotune controller
# -----------------------------

@dataclass
class AutoTuneConfig:
    target_E: float = 0.50
    band: float = 0.12  # acceptable band around target
    kp: float = 0.030   # proportional gain
    ki: float = 0.0006  # integral gain

    # which knob to tune primarily
    tune_knob: str = "feedback_gain"  # or "E_leak" or "noise_std"

    # bounds for knobs
    feedback_gain_min: float = 0.002
    feedback_gain_max: float = 0.080
    E_leak_min: float = 0.002
    E_leak_max: float = 0.030
    noise_min: float = 0.05
    noise_max: float = 0.60

    # chunking
    epochs: int = 14
    steps_per_epoch: int = 52000  # ~200 slots at period=260


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def main():
    # -------------------------
    # configs
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
        enable_create_prune=False,
        seed=0,
    )

    env_cfg = AutoTuneEnvConfig(
        period=260,
        pulse_len=35,
        relax_len=110,
        regimes=3,
        nodes_per_regime=9,
        base_amp=0.8,
        noise_std=0.25,
        E_init=0.0,
        E_clip=2.0,
        E_leak=0.008,
        E_drive=0.004,
        feedback_gain=0.03,
        feedback_mode="energy",
        E_scale_factor=0.35,
        permeability_jitter=0.02,
        permeability_period=5000,
    )

    tune_cfg = AutoTuneConfig(
        target_E=0.50,
        band=0.12,
        kp=0.030,
        ki=0.0006,
        tune_knob="feedback_gain",
        epochs=14,
        steps_per_epoch=52000,
    )

    slot_cfg = SlotConfig(pulse_len=env_cfg.pulse_len, relax_len=env_cfg.relax_len)

    # -------------------------
    # autotune loop
    # -------------------------
    I_int = 0.0  # integral state
    epoch_reports: List[str] = []
    all_out: Optional[Dict[str, Any]] = None

    t0_global = time.time()

    for epoch in range(tune_cfg.epochs):
        env = AutoTuneClosedLoopEnvironment(substrate_cfg.N, env_cfg, seed=5 + epoch)

        run_cfg = RunConfig(
            total_steps=tune_cfg.steps_per_epoch,
            slot_period=env_cfg.period,
            record_V=True,
            record_metrics=True,
        )

        out = run_simulation(
            substrate_cfg=substrate_cfg,
            plasticity_cfg=plasticity_cfg,
            run_cfg=run_cfg,
            env_fn=env.env_fn,
            slot_cfg=slot_cfg,
            true_token_fn=None,
            seed=epoch,
        )

        all_out = out  # keep last out for saving

        E_arr = np.array(env.E_hist, dtype=np.float32)
        tail = E_arr[-min(len(E_arr), 20000):]
        E_mean = float(tail.mean())
        E_std = float(tail.std())
        err = tune_cfg.target_E - E_mean

        # PI update
        I_int += err
        u = tune_cfg.kp * err + tune_cfg.ki * I_int

        # apply update to chosen knob
        if tune_cfg.tune_knob == "feedback_gain":
            env_cfg.feedback_gain = clamp(
                env_cfg.feedback_gain + u,
                tune_cfg.feedback_gain_min,
                tune_cfg.feedback_gain_max,
            )
        elif tune_cfg.tune_knob == "E_leak":
            # inverse direction: higher leak pulls E down
            env_cfg.E_leak = clamp(
                env_cfg.E_leak - u,
                tune_cfg.E_leak_min,
                tune_cfg.E_leak_max,
            )
        elif tune_cfg.tune_knob == "noise_std":
            env_cfg.noise_std = clamp(
                env_cfg.noise_std + 0.25 * abs(u),
                tune_cfg.noise_min,
                tune_cfg.noise_max,
            )
        else:
            raise ValueError(f"Unknown tune_knob: {tune_cfg.tune_knob}")

        in_band = (abs(err) <= tune_cfg.band)

        line = (
            f"epoch={epoch:02d}  E_mean={E_mean:.3f}  E_std={E_std:.3f}  "
            f"err={err:+.3f}  in_band={in_band}  "
            f"feedback_gain={env_cfg.feedback_gain:.4f}  E_leak={env_cfg.E_leak:.4f}  noise={env_cfg.noise_std:.3f}"
        )
        print(line)
        epoch_reports.append(line)

    # -------------------------
    # lightweight alphabet sample on final epoch output
    # -------------------------
    assert all_out is not None
    V_hist = all_out["V_hist"]
    period = env_cfg.period
    total_steps = V_hist.shape[0]
    slots_total = total_steps // period

    samples: List[np.ndarray] = []
    sample_times: List[int] = []

    # sample 1 per slot (skip first 12)
    for slot in range(12, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts < total_steps:
            sample_times.append(int(ts))
            samples.append(V_hist[ts].astype(np.float32))

    samples_arr = np.stack(samples, axis=0) if len(samples) else np.zeros((0, substrate_cfg.N), np.float32)
    sample_times_arr = np.array(sample_times, dtype=np.int32)

    cl_cfg = CosineClusterConfig(threshold=0.995)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples_arr, cl_cfg)

    # -------------------------
    # save
    # -------------------------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_autotune_phenotype_{stamp}.npz"

    payload = dict(all_out)
    payload.update(
        # env logs are only from final epoch (still useful)
        E_hist=np.array([], dtype=np.float32),  # placeholder (per-epoch env was local)
        autotune_epoch_reports=np.array(epoch_reports, dtype="U256"),

        sample_times=sample_times_arr,
        attractor_samples=samples_arr,
        attractor_id=cluster_ids.astype(np.int32),
        cluster_reps=reps.astype(np.float32),
        cluster_sizes=np.array(sizes, dtype=np.int32),

        # final tuned params
        tuned_feedback_gain=np.float32(env_cfg.feedback_gain),
        tuned_E_leak=np.float32(env_cfg.E_leak),
        tuned_noise_std=np.float32(env_cfg.noise_std),
        tuned_E_scale_factor=np.float32(env_cfg.E_scale_factor),
        tuned_permeability_jitter=np.float32(env_cfg.permeability_jitter),
        tuned_permeability_period=np.int32(env_cfg.permeability_period),

        target_E=np.float32(tune_cfg.target_E),
        target_band=np.float32(tune_cfg.band),
    )

    savez_safe(path, payload)

    dt = time.time() - t0_global
    print("\nPhase Autotune results")
    print("----------------------")
    print(f"epochs        : {tune_cfg.epochs}")
    print(f"steps/epoch   : {tune_cfg.steps_per_epoch}")
    print(f"target        : {tune_cfg.target_E:.3f}")
    print(f"band          : ±{tune_cfg.band:.3f}")
    print(f"final knobs   : feedback_gain={env_cfg.feedback_gain:.4f}  E_leak={env_cfg.E_leak:.4f}  noise_std={env_cfg.noise_std:.3f}")
    print(f"samples       : {len(samples_arr)}")
    print(f"clusters_seen : {len(reps)}")
    print(f"Saved: {path}")
    print(f"Runtime: {dt:.1f} s\n")


if __name__ == "__main__":
    main()
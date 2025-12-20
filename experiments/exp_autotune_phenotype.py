# experiments/exp_autotune_phenotype.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation

from analysis.cluster import CosineClusterConfig, cluster_cosine_incremental
from analysis.decode import build_unsupervised_labels, decode_sequence


# -----------------------------
# Safe NPZ saving (no pickles)
# -----------------------------
def _to_safe_np(value: Any) -> Any:
    """Convert common Python containers to safe numpy arrays (no object dtype)."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("Refusing to save object-dtype arrays (pickle risk).")
        return value

    if isinstance(value, (float, int, bool, np.floating, np.integer, np.bool_)):
        return np.array(value)

    if isinstance(value, str):
        return np.array(value, dtype="U")

    if isinstance(value, (list, tuple)):
        # Try numeric
        try:
            arr = np.array(value)
            if arr.dtype == object:
                # maybe list of arrays -> stack
                if len(value) > 0 and all(isinstance(x, np.ndarray) for x in value):
                    arr2 = np.stack([x.astype(np.float32) for x in value], axis=0)
                    return arr2
                raise TypeError("List produced object dtype; convert explicitly.")
            return arr
        except Exception as e:
            raise TypeError(f"Cannot safely convert list/tuple for NPZ: {e}")

    if isinstance(value, dict):
        raise TypeError("Nested dicts not allowed in NPZ payload (flatten first).")

    raise TypeError(f"Unsupported type for safe NPZ saving: {type(value)}")


def savez_safe(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    safe: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        safe[k] = _to_safe_np(v)
    np.savez_compressed(path, **safe)


# ---------------------------------------------------------------------
# Closed-loop environment (self-contained; no sklearn dependencies)
# ---------------------------------------------------------------------
@dataclass
class ClosedLoopEnvConfig:
    period: int = 260
    pulse_len: int = 35
    relax_len: int = 110

    regimes: int = 3
    nodes_per_regime: int = 9

    base_amp: float = 0.8
    noise_std: float = 0.18

    # scalar environment dynamics
    E_init: float = 0.0
    E_leak: float = 0.008
    E_drive: float = 0.004
    E_clip: float = 2.0

    # substrate -> environment coupling
    feedback_gain: float = 0.0020
    feedback_mode: str = "energy"  # "energy" or "sign"

    # injection scaling vs E
    E_scale_factor: float = 0.35   # scale = 1 + E_scale_factor * E


class ClosedLoopEnvironment:
    def __init__(self, N: int, cfg: ClosedLoopEnvConfig, seed: int = 5):
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

        self.E = float(cfg.E_init)

        # logs
        self.E_hist: List[float] = []
        self.regime_hist: List[int] = []
        self.readout_hist: List[float] = []

    def reset_logs(self) -> None:
        self.E_hist = []
        self.regime_hist = []
        self.readout_hist = []

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
        phi = np.tanh(readout)
        dE = (-self.cfg.E_leak * self.E) + self.cfg.E_drive + (self.cfg.feedback_gain * phi)
        self.E = float(np.clip(self.E + dE, -self.cfg.E_clip, self.cfg.E_clip))

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        slot = t // self.cfg.period
        r = self.regime_at_slot(slot)

        readout = self.substrate_readout(substrate)
        self.update_environment(readout)

        self.E_hist.append(self.E)
        self.regime_hist.append(r)
        self.readout_hist.append(readout)

        within = t % self.cfg.period
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        scale = 1.0 + self.cfg.E_scale_factor * self.E

        if within < self.cfg.pulse_len:
            I[self.regime_nodes[r]] = (self.regime_amp[r] * scale)

        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15
        I += self.cfg.noise_std * self.rng.standard_normal(self.N)

        return I.astype(substrate.cfg.dtype)


# ---------------------------------------------------------------------
# Autotune loop
# ---------------------------------------------------------------------
@dataclass
class AutotuneConfig:
    epochs: int = 14
    steps_per_epoch: int = 52000

    # Target for mean(E) over an epoch (steady-ish)
    target_E_mean: float = 0.50
    band: float = 0.12

    # Control step sizes
    gain_step: float = 1.25      # multiplicative for feedback_gain
    leak_step: float = 1.15      # multiplicative for E_leak
    noise_step: float = 1.10     # multiplicative for noise_std

    # Bounds
    feedback_gain_min: float = 0.0003
    feedback_gain_max: float = 0.05
    E_leak_min: float = 0.001
    E_leak_max: float = 0.05
    noise_min: float = 0.05
    noise_max: float = 0.60

    # Sampling / clustering
    cosine_threshold: float = 0.995
    skip_slots: int = 12


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def main():
    # -------------------------
    # Base configs
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
        enable_create_prune=False,
        seed=0,
    )

    env_cfg = ClosedLoopEnvConfig()
    slot_cfg = SlotConfig(pulse_len=env_cfg.pulse_len, relax_len=env_cfg.relax_len)

    tune = AutotuneConfig()

    # -------------------------
    # Autotune epochs
    # -------------------------
    start = time.time()

    final_out = None
    final_env = None

    for epoch in range(tune.epochs):
        env = ClosedLoopEnvironment(substrate_cfg.N, env_cfg, seed=5)
        env.reset_logs()

        run_cfg = RunConfig(
            total_steps=tune.steps_per_epoch,
            slot_period=env_cfg.period,
            record_V=True,
            record_dV=False,
            record_metrics=True,
        )

        out = run_simulation(
            substrate_cfg=substrate_cfg,
            plasticity_cfg=plasticity_cfg,
            run_cfg=run_cfg,
            env_fn=env.env_fn,
            slot_cfg=slot_cfg,
            true_token_fn=None,
            seed=epoch,  # vary a bit across epochs
        )

        E_arr = np.array(env.E_hist, dtype=np.float32)
        E_mean = float(E_arr.mean()) if E_arr.size else float("nan")
        E_std = float(E_arr.std()) if E_arr.size else float("nan")

        err = E_mean - tune.target_E_mean
        in_band = (abs(err) <= tune.band)

        print(
            f"epoch={epoch:02d}  E_mean={E_mean:.3f}  E_std={E_std:.3f}  "
            f"err={err:+.3f}  in_band={in_band}  "
            f"feedback_gain={env_cfg.feedback_gain:.4f}  E_leak={env_cfg.E_leak:.4f}  noise={env_cfg.noise_std:.3f}"
        )

        # Simple self-tuning rules:
        # - If E too high => reduce feedback_gain, increase leak, optionally increase noise
        # - If E too low  => increase feedback_gain, decrease leak, optionally decrease noise
        if not in_band:
            if err > 0:
                env_cfg.feedback_gain = clamp(env_cfg.feedback_gain / tune.gain_step,
                                              tune.feedback_gain_min, tune.feedback_gain_max)
                env_cfg.E_leak = clamp(env_cfg.E_leak * tune.leak_step,
                                       tune.E_leak_min, tune.E_leak_max)
                env_cfg.noise_std = clamp(env_cfg.noise_std * tune.noise_step,
                                          tune.noise_min, tune.noise_max)
            else:
                env_cfg.feedback_gain = clamp(env_cfg.feedback_gain * tune.gain_step,
                                              tune.feedback_gain_min, tune.feedback_gain_max)
                env_cfg.E_leak = clamp(env_cfg.E_leak / tune.leak_step,
                                       tune.E_leak_min, tune.E_leak_max)
                env_cfg.noise_std = clamp(env_cfg.noise_std / tune.noise_step,
                                          tune.noise_min, tune.noise_max)

        final_out = out
        final_env = env

    assert final_out is not None and final_env is not None

    # -------------------------
    # Build samples for clustering (like Phase-6)
    # -------------------------
    V_hist = final_out["V_hist"]
    period = env_cfg.period
    slots_total = V_hist.shape[0] // period

    sample_times: List[int] = []
    samples: List[np.ndarray] = []
    regime_labels: List[int] = []

    for slot in range(tune.skip_slots, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts < V_hist.shape[0]:
            sample_times.append(ts)
            samples.append(V_hist[ts])
            regime_labels.append(final_env.regime_at_slot(slot))

    sample_times_np = np.array(sample_times, dtype=np.int32)
    samples_np = np.array(samples)
    regime_labels_np = np.array(regime_labels, dtype=np.int32)

    cl_cfg = CosineClusterConfig(threshold=tune.cosine_threshold)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples_np, cl_cfg)

    labels_map = build_unsupervised_labels(cluster_ids, prefix="T")
    decoded = decode_sequence(cluster_ids, labels_map)

    print("\nPhase Autotune results")
    print("----------------------")
    print(f"epochs        : {tune.epochs}")
    print(f"steps/epoch   : {tune.steps_per_epoch}")
    print(f"target        : {tune.target_E_mean:.3f}")
    print(f"band          : ±{tune.band:.3f}")
    print(f"final knobs   : feedback_gain={env_cfg.feedback_gain:.4f}  E_leak={env_cfg.E_leak:.4f}  noise_std={env_cfg.noise_std:.3f}")
    print(f"samples       : {len(samples_np)}")
    print(f"clusters_seen : {len(np.unique(cluster_ids))}")

    # -------------------------
    # Save NPZ (consistent keys)
    # -------------------------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_autotune_phenotype_{stamp}.npz"

    E_hist = np.array(final_env.E_hist, dtype=np.float32)
    regime_hist = np.array(final_env.regime_hist, dtype=np.int32)
    readout_hist = np.array(final_env.readout_hist, dtype=np.float32)

    payload = dict(final_out)
    payload.update(
        sample_times=sample_times_np,
        attractor_samples=samples_np.astype(np.float32),
        attractor_id=np.array(cluster_ids, dtype=np.int32),
        cluster_reps=np.array(reps, dtype=np.float32),
        cluster_sizes=np.array(sizes, dtype=np.int32),
        hidden_regime_labels=regime_labels_np.astype(np.int32),
        unsupervised_token_samples=np.array(decoded, dtype="U16"),

        E_hist=E_hist,
        regime_hist=regime_hist,
        substrate_readout_hist=readout_hist,

        # metadata (handy for docs)
        autotune_epochs=np.array(tune.epochs, dtype=np.int32),
        autotune_steps_per_epoch=np.array(tune.steps_per_epoch, dtype=np.int32),
        autotune_target_E_mean=np.array(tune.target_E_mean, dtype=np.float32),
        autotune_band=np.array(tune.band, dtype=np.float32),
        final_feedback_gain=np.array(env_cfg.feedback_gain, dtype=np.float32),
        final_E_leak=np.array(env_cfg.E_leak, dtype=np.float32),
        final_noise_std=np.array(env_cfg.noise_std, dtype=np.float32),
        cosine_threshold=np.array(tune.cosine_threshold, dtype=np.float32),
    )

    savez_safe(path, payload)

    runtime = time.time() - start
    print(f"\nSaved: {path}")
    print(f"Runtime: {runtime:.1f} s\n")


if __name__ == "__main__":
    main()
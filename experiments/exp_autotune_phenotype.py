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
"""
Fuka-6.0 Experiment: exp_autotune_phenotype
Self-tuning phenotype loop (no manual tuning)

Core idea:
  - Closed-loop environment with scalar E(t)
  - Add a thermostat T(t) that automatically adjusts exploration (noise + excitation)
  - Use a lightweight online "alphabet probe" from attractor samples
  - Tune T using core_coverage in a sliding window:
        T <- T * exp(k*(target_coverage - observed_coverage))

What it produces:
  - NPZ: runs/exp_autotune_phenotype_fixed_<stamp>.npz
  - Report (Markdown): runs/reports/exp_autotune_phenotype_<stamp>.md
  - Optional triggered snapshots: runs/exp_autotune_phenotype_snap_<stamp>_kXXXXXX.npz

Run:
  python -m experiments.exp_autotune_phenotype
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation


# -----------------------------
# Safe NPZ saving (no pickles)
# -----------------------------

def _is_pickle_risky_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == object

def savez_safe(path: str, payload: Dict[str, Any]) -> None:
    """
    Save with allow_pickle=False compatibility.
    Reject object arrays (ragged lists etc.).
    """
    clean: Dict[str, Any] = {}
    for k, v in payload.items():
        if _is_pickle_risky_array(v):
            raise ValueError(f"savez_safe: key '{k}' is object array (pickle risk). Refuse to save.")
        clean[k] = v
    np.savez_compressed(path, **clean)


# -----------------------------
# Online cosine clustering probe
# -----------------------------

@dataclass
class OnlineClusterConfig:
    threshold: float = 0.995  # cosine similarity threshold
    max_reps: int = 20000     # safety cap (avoid RAM blowups)

class OnlineCosineCluster:
    """
    Incremental clustering of attractor samples by cosine similarity.
    Maintains:
      - reps: (K,N) float32
      - counts: (K,) int32
      - assigns: list[int] for each sample
    """
    def __init__(self, cfg: OnlineClusterConfig, dim: int, dtype=np.float32):
        self.cfg = cfg
        self.dim = dim
        self.dtype = dtype
        self.reps = np.zeros((0, dim), dtype=dtype)
        self.counts = np.zeros((0,), dtype=np.int32)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        # a,b are 1D
        na = float(np.linalg.norm(a) + 1e-12)
        nb = float(np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b) / (na * nb))

    def assign(self, x: np.ndarray) -> int:
        x = x.astype(self.dtype, copy=False)

        if self.reps.shape[0] == 0:
            self.reps = x[None, :].copy()
            self.counts = np.array([1], dtype=np.int32)
            return 0

        # brute-force cosine against reps (K up to a few thousand typically OK)
        best_i = -1
        best_s = -1.0
        for i in range(self.reps.shape[0]):
            s = self._cosine_sim(x, self.reps[i])
            if s > best_s:
                best_s = s
                best_i = i

        if best_s >= self.cfg.threshold:
            self.counts[best_i] += 1
            return int(best_i)

        # new rep
        if self.reps.shape[0] >= self.cfg.max_reps:
            # if we hit cap, force-assign to best anyway
            self.counts[best_i] += 1
            return int(best_i)

        self.reps = np.vstack([self.reps, x[None, :]])
        self.counts = np.concatenate([self.counts, np.array([1], dtype=np.int32)])
        return int(self.reps.shape[0] - 1)


# -----------------------------
# Sliding window coverage probe
# -----------------------------

class SlidingCoverageProbe:
    """
    Tracks core_coverage in a sliding window of cluster ids.

    core_coverage(window) = fraction of samples in window belonging to clusters
    with count >= core_threshold (computed within the window itself).
    """
    def __init__(self, window: int = 500, core_threshold: int = 10):
        self.window = int(window)
        self.core_threshold = int(core_threshold)
        self.buf = np.full((self.window,), -1, dtype=np.int32)
        self.ptr = 0
        self.filled = 0

    def update(self, cid: int) -> None:
        self.buf[self.ptr] = int(cid)
        self.ptr = (self.ptr + 1) % self.window
        self.filled = min(self.window, self.filled + 1)

    def coverage(self) -> float:
        if self.filled < max(20, self.core_threshold * 2):
            return 0.0
        b = self.buf[:self.filled]
        # counts within window
        uniq, cnt = np.unique(b, return_counts=True)
        core = set(int(u) for u, c in zip(uniq, cnt) if int(c) >= self.core_threshold)
        if not core:
            return 0.0
        hits = sum(1 for x in b if int(x) in core)
        return float(hits / len(b))

    def novelty_rate(self) -> float:
        if self.filled < 50:
            return 1.0
        b = self.buf[:self.filled]
        uniq = len(set(int(x) for x in b))
        return float(uniq / len(b))


# -----------------------------
# Closed-loop + thermostat
# -----------------------------

@dataclass
class AutoTuneEnvConfig:
    period: int = 260
    pulse_len: int = 35
    relax_len: int = 110

    regimes: int = 3
    nodes_per_regime: int = 9

    base_amp: float = 0.8
    noise_std: float = 0.18

    # scalar E(t)
    E_init: float = 0.0
    E_leak: float = 0.008
    E_drive: float = 0.004
    E_clip: float = 2.0
    feedback_gain: float = 0.03
    feedback_mode: str = "energy"  # "energy" or "sign"

    # excitation scaling (gentler than 1+E)
    E_scale_factor: float = 0.35

    # thermostat parameters
    T_init: float = 1.0
    T_min: float = 0.25
    T_max: float = 3.00
    T_k: float = 0.8                 # learning rate in exp update
    target_coverage: float = 0.40     # desired core_coverage
    tune_every_slots: int = 20        # update T every N slots

    # how thermostat affects environment
    T_affects_noise: bool = True
    T_affects_amp: bool = True

    # soft saturation (avoid hard clipping flatline)
    soft_saturate_E: bool = True


class AutoTuneEnvironment:
    def __init__(self, N: int, cfg: AutoTuneEnvConfig, seed: int = 5):
        self.N = int(N)
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
        self.T = float(cfg.T_init)

        # logs
        self.E_hist: List[float] = []
        self.T_hist: List[float] = []
        self.regime_hist: List[int] = []
        self.readout_hist: List[float] = []
        self.noise_hist: List[float] = []
        self.amp_scale_hist: List[float] = []

    def regime_at_slot(self, slot: int) -> int:
        return (slot // self.regime_period_slots) % self.R

    def substrate_readout(self, substrate: Substrate) -> float:
        V = substrate.V.astype(np.float64)
        if self.cfg.feedback_mode == "energy":
            return float(np.dot(V, V) / len(V))
        if self.cfg.feedback_mode == "sign":
            return float(np.mean(V))
        raise ValueError(f"Unknown feedback_mode {self.cfg.feedback_mode}")

    def _soft_sat(self, x: float) -> float:
        # smooth clip: E = E_clip * tanh(E/E_clip)
        c = float(self.cfg.E_clip)
        return float(c * np.tanh(x / max(c, 1e-9)))

    def update_environment(self, readout: float) -> None:
        phi = np.tanh(readout)
        dE = (-self.cfg.E_leak * self.E) + self.cfg.E_drive + (self.cfg.feedback_gain * phi)
        self.E = float(self.E + dE)

        if self.cfg.soft_saturate_E:
            self.E = self._soft_sat(self.E)
        else:
            self.E = float(np.clip(self.E, -self.cfg.E_clip, self.cfg.E_clip))

    def tune_thermostat(self, observed_coverage: float) -> None:
        # T <- T * exp(k*(target - observed))
        err = float(self.cfg.target_coverage - observed_coverage)
        self.T *= float(np.exp(self.cfg.T_k * err))
        self.T = float(np.clip(self.T, self.cfg.T_min, self.cfg.T_max))

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        slot = t // self.cfg.period
        r = self.regime_at_slot(slot)

        readout = self.substrate_readout(substrate)
        self.update_environment(readout)

        # scale from E (gentler than 1+E)
        amp_scale = 1.0 + self.cfg.E_scale_factor * self.E
        amp_scale = float(max(0.05, amp_scale))

        # thermostat application
        noise_std = self.cfg.noise_std
        if self.cfg.T_affects_noise:
            noise_std = float(noise_std * self.T)

        if self.cfg.T_affects_amp:
            amp_scale = float(amp_scale * self.T)

        within = t % self.cfg.period
        I = np.zeros(self.N, dtype=substrate.cfg.dtype)

        if within < self.cfg.pulse_len:
            I[self.regime_nodes[r]] = self.regime_amp[r] * amp_scale

        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15
        I += noise_std * self.rng.standard_normal(self.N)

        # logs
        self.E_hist.append(self.E)
        self.T_hist.append(self.T)
        self.regime_hist.append(int(r))
        self.readout_hist.append(readout)
        self.noise_hist.append(noise_std)
        self.amp_scale_hist.append(amp_scale)

        return I.astype(substrate.cfg.dtype)


# -----------------------------
# Report writer (Markview/MD)
# -----------------------------

def write_report_md(
    path_md: str,
    run_id: str,
    git_commit: str,
    cfg_sub: SubstrateConfig,
    cfg_pl: PlasticityConfig,
    cfg_env: AutoTuneEnvConfig,
    summary: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path_md), exist_ok=True)

    def f(x: float) -> str:
        return f"{x:.6g}"

    lines: List[str] = []
    lines.append(f"# exp_autotune_phenotype — run {run_id}")
    lines.append("")
    lines.append("## Run metadata")
    lines.append(f"- **run_id:** `{run_id}`")
    lines.append(f"- **git_commit:** `{git_commit}`")
    lines.append(f"- **npz:** `{summary['npz_path']}`")
    lines.append("")

    lines.append("## Key results (high level)")
    lines.append(f"- **total_steps:** `{summary['total_steps']}`")
    lines.append(f"- **samples:** `{summary['num_samples']}`")
    lines.append(f"- **clusters_seen:** `{summary['clusters_seen']}`")
    lines.append(f"- **core_threshold (window):** `{summary['core_threshold']}`")
    lines.append(f"- **final core_coverage (window):** `{f(summary['final_core_coverage'])}`")
    lines.append(f"- **final novelty_rate (window):** `{f(summary['final_novelty_rate'])}`")
    lines.append(f"- **E_final:** `{f(summary['E_final'])}`")
    lines.append(f"- **T_final:** `{f(summary['T_final'])}`")
    lines.append("")

    lines.append("## Interpretation (plain language)")
    lines.append(
        "This run uses a closed-loop environment with a self-tuning thermostat. "
        "The thermostat continuously adjusts exploration pressure (noise and excitation) "
        "to push the system toward a target re-use rate (core coverage) rather than unbounded novelty. "
        "If core coverage is too low (too many one-off attractors), exploration is reduced. "
        "If core coverage is too high (too repetitive), exploration is increased."
    )
    lines.append("")

    lines.append("## Parameters")
    lines.append("### Substrate")
    lines.append(f"- N: `{cfg_sub.N}`")
    lines.append(f"- C: `{cfg_sub.C}`")
    lines.append(f"- lam: `{cfg_sub.lam}`")
    lines.append(f"- dt: `{cfg_sub.dt}`")
    lines.append(f"- init_v_std: `{cfg_sub.init_v_std}`")
    lines.append(f"- init_g_scale: `{cfg_sub.init_g_scale}`")
    lines.append(f"- symmetric_g: `{cfg_sub.symmetric_g}`")
    lines.append("")
    lines.append("### Plasticity")
    lines.append(f"- eta: `{cfg_pl.eta}`")
    lines.append(f"- alpha: `{cfg_pl.alpha}`")
    lines.append(f"- use_fitness_gate: `{cfg_pl.use_fitness_gate}`")
    lines.append(f"- enable_create_prune: `{cfg_pl.enable_create_prune}`")
    lines.append("")
    lines.append("### Environment + thermostat")
    lines.append(f"- regimes: `{cfg_env.regimes}`; nodes_per_regime: `{cfg_env.nodes_per_regime}`")
    lines.append(f"- base_amp: `{cfg_env.base_amp}`; noise_std: `{cfg_env.noise_std}`")
    lines.append(f"- E_leak: `{cfg_env.E_leak}`; E_drive: `{cfg_env.E_drive}`; E_clip: `{cfg_env.E_clip}`")
    lines.append(f"- feedback_gain: `{cfg_env.feedback_gain}`; feedback_mode: `{cfg_env.feedback_mode}`")
    lines.append(f"- E_scale_factor: `{cfg_env.E_scale_factor}`; soft_saturate_E: `{cfg_env.soft_saturate_E}`")
    lines.append(f"- target_coverage: `{cfg_env.target_coverage}`; tune_every_slots: `{cfg_env.tune_every_slots}`")
    lines.append(f"- T_init: `{cfg_env.T_init}`; T_min: `{cfg_env.T_min}`; T_max: `{cfg_env.T_max}`; T_k: `{cfg_env.T_k}`")
    lines.append(f"- T_affects_noise: `{cfg_env.T_affects_noise}`; T_affects_amp: `{cfg_env.T_affects_amp}`")
    lines.append("")

    lines.append("## Probes recorded")
    lines.append("- `E_hist(t)`: environment scalar")
    lines.append("- `T_hist(t)`: thermostat scalar")
    lines.append("- `core_coverage_hist`: sliding-window reuse rate")
    lines.append("- `novelty_rate_hist`: sliding-window uniqueness rate")
    lines.append("- `sample_times`, `attractor_id`, `cluster_sizes`: online alphabet probe outputs")
    lines.append("")

    with open(path_md, "w", encoding="utf-8") as fmd:
        fmd.write("\n".join(lines) + "\n")


# -----------------------------
# Main
# -----------------------------

def main():
    # --- run id / stamp ---
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # --- configs ---
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
        noise_std=0.18,
        E_init=0.0,
        E_leak=0.008,
        E_drive=0.004,
        E_clip=2.0,
        feedback_gain=0.03,
        feedback_mode="energy",
        E_scale_factor=0.35,
        T_init=1.0,
        T_min=0.25,
        T_max=3.0,
        T_k=0.8,
        target_coverage=0.40,
        tune_every_slots=20,
        T_affects_noise=True,
        T_affects_amp=True,
        soft_saturate_E=True,
    )

    slot_cfg = SlotConfig(pulse_len=env_cfg.pulse_len, relax_len=env_cfg.relax_len)
    env = AutoTuneEnvironment(substrate_cfg.N, env_cfg, seed=7)

    # Long but not insane by default; you can push to weeks by raising total_steps.
    run_cfg = RunConfig(
        total_steps=400_000,
        slot_period=env_cfg.period,
        record_V=True,
        record_dV=False,
        record_metrics=True,
    )

    # --- probe config ---
    cluster_cfg = OnlineClusterConfig(threshold=0.995, max_reps=20000)
    clusterer = OnlineCosineCluster(cluster_cfg, dim=substrate_cfg.N, dtype=np.float32)

    window_probe = SlidingCoverageProbe(window=600, core_threshold=10)

    # Sample once per slot at slot_cfg.sample_index(t0)
    period = env_cfg.period
    slots_total = run_cfg.total_steps // period
    skip_slots = 12

    sample_times: List[int] = []
    attractor_ids: List[int] = []
    core_coverage_hist: List[float] = []
    novelty_rate_hist: List[float] = []
    T_updates: List[Tuple[int, float, float]] = []  # (slot, coverage, T)

    # Optional triggered snapshots (small) using interesting events
    snap_paths: List[str] = []
    snap_every_slots = 500  # periodic safety snapshot (small)
    snap_on_coverage_cross = 0.30
    last_crossed = False

    # --- run simulation ---
    out = run_simulation(
        substrate_cfg=substrate_cfg,
        plasticity_cfg=plasticity_cfg,
        run_cfg=run_cfg,
        env_fn=env.env_fn,
        slot_cfg=slot_cfg,
        true_token_fn=None,
        seed=0,
    )

    # We sample from recorded V_hist (simple & consistent with your current codebase).
    V_hist = out["V_hist"]

    for slot in range(skip_slots, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts >= run_cfg.total_steps:
            break

        v = V_hist[ts].astype(np.float32, copy=False)
        cid = clusterer.assign(v)

        sample_times.append(int(ts))
        attractor_ids.append(int(cid))

        window_probe.update(cid)
        cov = window_probe.coverage()
        nov = window_probe.novelty_rate()
        core_coverage_hist.append(float(cov))
        novelty_rate_hist.append(float(nov))

        # thermostat tune
        if (slot % env_cfg.tune_every_slots) == 0:
            env.tune_thermostat(cov)
            T_updates.append((int(slot), float(cov), float(env.T)))

        # periodic tiny snapshot (ids + probes only)
        if (slot % snap_every_slots) == 0 and slot > 0:
            os.makedirs("runs", exist_ok=True)
            sp = f"runs/exp_autotune_phenotype_snap_{run_id}_k{slot:06d}.npz"
            savez_safe(sp, {
                "run_id": np.array(run_id, dtype="U32"),
                "slot": np.array(slot, dtype=np.int32),
                "sample_times": np.array(sample_times, dtype=np.int32),
                "attractor_id": np.array(attractor_ids, dtype=np.int32),
                "cluster_sizes": clusterer.counts.astype(np.int32),
                "core_coverage_hist": np.array(core_coverage_hist, dtype=np.float32),
                "novelty_rate_hist": np.array(novelty_rate_hist, dtype=np.float32),
                "E_tail": np.array(env.E_hist[-5000:], dtype=np.float32),
                "T_tail": np.array(env.T_hist[-5000:], dtype=np.float32),
            })
            snap_paths.append(sp)

        # trigger snapshot on coverage crossing
        crossed = (cov >= snap_on_coverage_cross)
        if crossed and (not last_crossed):
            os.makedirs("runs", exist_ok=True)
            sp = f"runs/exp_autotune_phenotype_snap_{run_id}_CROSS_{slot:06d}.npz"
            savez_safe(sp, {
                "run_id": np.array(run_id, dtype="U32"),
                "slot": np.array(slot, dtype=np.int32),
                "sample_times": np.array(sample_times, dtype=np.int32),
                "attractor_id": np.array(attractor_ids, dtype=np.int32),
                "cluster_sizes": clusterer.counts.astype(np.int32),
                "core_coverage_hist": np.array(core_coverage_hist, dtype=np.float32),
                "novelty_rate_hist": np.array(novelty_rate_hist, dtype=np.float32),
                "E_tail": np.array(env.E_hist[-20000:], dtype=np.float32),
                "T_tail": np.array(env.T_hist[-20000:], dtype=np.float32),
            })
            snap_paths.append(sp)
            last_crossed = True
        if not crossed:
            last_crossed = False

    # --- save main NPZ ---
    os.makedirs("runs", exist_ok=True)
    npz_path = f"runs/exp_autotune_phenotype_fixed_{run_id}.npz"

    payload = dict(out)
    payload.update(
        # probe outputs
        sample_times=np.array(sample_times, dtype=np.int32),
        attractor_id=np.array(attractor_ids, dtype=np.int32),
        cluster_sizes=clusterer.counts.astype(np.int32),
        core_coverage_hist=np.array(core_coverage_hist, dtype=np.float32),
        novelty_rate_hist=np.array(novelty_rate_hist, dtype=np.float32),
        T_updates=np.array(T_updates, dtype=np.float32),  # columns: slot,cov,T

        # environment logs
        E_hist=np.array(env.E_hist, dtype=np.float32),
        T_hist=np.array(env.T_hist, dtype=np.float32),
        regime_hist=np.array(env.regime_hist, dtype=np.int32),
        substrate_readout_hist=np.array(env.readout_hist, dtype=np.float32),
        noise_hist=np.array(env.noise_hist, dtype=np.float32),
        amp_scale_hist=np.array(env.amp_scale_hist, dtype=np.float32),

        # run_id + snapshots list
        run_id=np.array(run_id, dtype="U32"),
        snapshot_paths=np.array(snap_paths, dtype="U256"),
    )

    savez_safe(npz_path, payload)

    # --- make report md ---
    git_commit = os.popen("git rev-parse --short HEAD").read().strip() or "unknown"
    report_path = f"runs/reports/exp_autotune_phenotype_{run_id}.md"

    final_cov = float(core_coverage_hist[-1]) if core_coverage_hist else 0.0
    final_nov = float(novelty_rate_hist[-1]) if novelty_rate_hist else 1.0

    summary = dict(
        npz_path=npz_path,
        total_steps=int(run_cfg.total_steps),
        num_samples=len(sample_times),
        clusters_seen=int(len(clusterer.counts)),
        core_threshold=int(window_probe.core_threshold),
        final_core_coverage=final_cov,
        final_novelty_rate=final_nov,
        E_final=float(env.E_hist[-1]) if env.E_hist else float("nan"),
        T_final=float(env.T_hist[-1]) if env.T_hist else float("nan"),
    )

    write_report_md(
        path_md=report_path,
        run_id=run_id,
        git_commit=git_commit,
        cfg_sub=substrate_cfg,
        cfg_pl=plasticity_cfg,
        cfg_env=env_cfg,
        summary=summary,
    )

    # --- print short console summary (ends up in your log file) ---
    print("\nexp_autotune_phenotype results")
    print("------------------------------")
    print(f"run_id: {run_id}")
    print(f"samples: {summary['num_samples']}")
    print(f"clusters_seen: {summary['clusters_seen']}")
    print(f"final core_coverage(window): {summary['final_core_coverage']:.3f}")
    print(f"final novelty_rate(window): {summary['final_novelty_rate']:.3f}")
    print(f"E_final: {summary['E_final']:.3f}")
    print(f"T_final: {summary['T_final']:.3f}")
    print(f"Saved NPZ: {npz_path}")
    print(f"Saved report: {report_path}")
    if snap_paths:
        print(f"Snapshots: {len(snap_paths)} (first: {snap_paths[0]})")
    print("")


if __name__ == "__main__":
    main()




    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
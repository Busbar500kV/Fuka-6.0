"""
Fuka-6.0 Experiment: Phase 5
Emergent hardware modules

Goal:
    After a run, analyze the evolved conductance topology g_last to detect:
      - hubs
      - modules (connected components in strong-edge graph)
      - slow pockets (low-variance voltage nodes -> long memory)

This is a first, physical definition of emergent "hardware" in Fuka-6.0.

Run:
    python -m experiments.exp_modules

NPZ:
    Saves to ./runs/exp_modules_<timestamp>.npz
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.substrate import SubstrateConfig, Substrate
from core.plasticity import PlasticityConfig
from core.metrics import SlotConfig
from core.run import RunConfig, run_simulation

from tools.safe_npz import savez_safe

# ---------------------------------------------------------------------
# A simple environment (reuse Phase-4 noisy regimes by default)
# ---------------------------------------------------------------------

class RegimeNoiseEnvironment:
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
        self.regime_amp = base_amp * (1.0 + 0.25 * self.rng.standard_normal(self.R))
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

        global_wave = np.sin(self.global_freq * t)
        I += global_wave * 0.15
        I += self.noise_std * self.rng.standard_normal(self.N)

        return I.astype(substrate.cfg.dtype)


# ---------------------------------------------------------------------
# Graph utilities (no external deps)
# ---------------------------------------------------------------------

def strong_edge_mask(g: np.ndarray, quantile: float = 0.98) -> np.ndarray:
    """
    Return boolean mask of edges stronger than given quantile.
    """
    flat = g.flatten()
    thr = np.quantile(flat, quantile)
    return g >= thr, thr


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """
    Connected components on an undirected version of mask graph.
    mask: (N,N) boolean strong edges.

    Returns list of arrays of node indices for each component.
    """
    N = mask.shape[0]
    adj = mask | mask.T
    visited = np.zeros(N, dtype=bool)
    comps = []

    for i in range(N):
        if visited[i]:
            continue
        # BFS/DFS
        stack = [i]
        nodes = []
        visited[i] = True
        while stack:
            u = stack.pop()
            nodes.append(u)
            neigh = np.where(adj[u])[0]
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(np.array(nodes, dtype=np.int32))

    # remove trivial singletons with no edges
    comps2 = []
    for c in comps:
        if len(c) == 1:
            u = c[0]
            if adj[u].any():
                comps2.append(c)
        else:
            comps2.append(c)
    return comps2


def hub_scores(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute:
      - out_strength_i = sum_j g_ij
      - in_strength_i  = sum_j g_ji

    Returns (out_strength, in_strength)
    """
    out_s = g.sum(axis=1)
    in_s = g.sum(axis=0)
    return out_s, in_s


def slow_pockets(V_hist: np.ndarray, window: int = 6000, low_quantile: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    Nodes with lowest variance over last window are "slow pockets".

    Returns:
        slow_nodes (indices), variance_threshold
    """
    T, N = V_hist.shape
    w0 = max(0, T - window)
    Vw = V_hist[w0:]
    var = Vw.var(axis=0)
    thr = np.quantile(var, low_quantile)
    slow_nodes = np.where(var <= thr)[0]
    return slow_nodes.astype(np.int32), float(thr)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def plot_strengths(out_s: np.ndarray, in_s: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(np.sort(out_s)[::-1], label="out-strength sorted")
    plt.plot(np.sort(in_s)[::-1], linestyle="--", label="in-strength sorted")
    plt.title("Node coupling strengths (sorted)")
    plt.xlabel("rank")
    plt.ylabel("strength")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_component_sizes(comps: List[np.ndarray]) -> None:
    sizes = sorted([len(c) for c in comps], reverse=True)
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(sizes)), sizes)
    plt.title("Strong-edge module sizes")
    plt.xlabel("module rank")
    plt.ylabel("size (nodes)")
    plt.tight_layout()
    plt.show()


def plot_slow_nodes(var: np.ndarray, slow_nodes: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(np.sort(var), label="variance sorted")
    plt.scatter(
        np.searchsorted(np.sort(var), var[slow_nodes]),
        var[slow_nodes],
        label="slow pocket nodes"
    )
    plt.title("Voltage variance (last window)")
    plt.xlabel("rank (low -> high)")
    plt.ylabel("variance")
    plt.legend()
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
        enable_create_prune=False,  # keep off for first module scan
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
        total_steps=48000,
        slot_period=period,
        record_V=True,
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
        seed=0,
    )

    V_hist = out["V_hist"]
    g_last = out["g_last"]

    # -------------------------
    # Hub detection
    # -------------------------
    out_s, in_s = hub_scores(g_last)
    hub_thr = np.quantile(out_s, 0.98)
    hubs = np.where(out_s >= hub_thr)[0].astype(np.int32)

    # -------------------------
    # Module detection (strong-edge components)
    # -------------------------
    mask, edge_thr = strong_edge_mask(g_last, quantile=0.985)
    comps = connected_components(mask)
    comps_sorted = sorted(comps, key=len, reverse=True)

    # -------------------------
    # Slow pockets
    # -------------------------
    # variance over tail window
    tail_window = 8000
    w0 = max(0, V_hist.shape[0] - tail_window)
    var_tail = V_hist[w0:].var(axis=0)
    slow_nodes, slow_thr = slow_pockets(V_hist, window=tail_window, low_quantile=0.1)

    # -------------------------
    # Print summary
    # -------------------------
    print("\nPhase 5 results")
    print("----------------")
    print(f"N = {substrate_cfg.N}")
    print(f"hubs_found = {len(hubs)}  (out-strength >= {hub_thr:.3e})")
    print(f"strong_edge_threshold = {edge_thr:.3e}")
    print(f"modules_found = {len(comps_sorted)}")
    print("top module sizes =", [len(c) for c in comps_sorted[:8]])
    print(f"slow_pocket_nodes = {len(slow_nodes)}  (variance <= {slow_thr:.3e})")

    # -------------------------
    # Plots
    # -------------------------
    plot_strengths(out_s, in_s)
    plot_component_sizes(comps_sorted)
    plot_slow_nodes(var_tail, slow_nodes)

    # -------------------------
    # Save NPZ
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_modules_{stamp}.npz"

    # pack module membership into ragged list using object array
    
    modules_json = [c.tolist() for c in comps_sorted]

    payload = dict(out)
    payload.update(
        hubs=hubs,
        hub_threshold=float(hub_thr),
        strong_edge_threshold=float(edge_thr),
        modules_json=modules_json,   # <- list of lists
        slow_nodes=slow_nodes,
        slow_variance_threshold=float(slow_thr),
        variance_tail=var_tail.astype(np.float32),
    )
    
    savez_safe(path, payload)

    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
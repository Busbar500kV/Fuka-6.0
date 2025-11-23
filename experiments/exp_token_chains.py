"""
Fuka-6.0 Experiment: Phase 2
Attractor-based tokens chaining into code

Goal:
    Drive the substrate with a repeating token string (e.g., ABBA...),
    let it self-organize, then show that:

      tokens -> attractors -> token chain

    i.e., emergent attractor IDs form a reusable codebook and reproduce
    the imposed symbolic chain with high accuracy.

Outputs:
    - fitness plot
    - emergent cluster IDs per slot
    - true vs decoded token chain plot
    - PCA view of attractor samples

Run:
    python -m experiments.exp_token_chains

NPZ:
    Saves to ./runs/exp_token_chains_<timestamp>.npz
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
from analysis.decode import build_supervised_mapping, decode_sequence, decode_accuracy
from analysis.plots import (
    plot_fitness,
    plot_cluster_ids_per_slot,
    plot_token_chain,
    plot_pca_samples
)


# ---------------------------------------------------------------------
# Environment for repeating code strings
# ---------------------------------------------------------------------

class CodeEnvironment:
    """
    Cycles through a provided code string over slots, e.g. "ABBAAABB..."

    Each symbol is implemented as a sparse pulse pattern on the substrate.
    """

    def __init__(
        self,
        N: int,
        code_string: str,
        token_names: List[str],
        nodes_per_token: int = 7,
        period: int = 240,
        pulse_len: int = 28,
        relax_len: int = 120,
        amps: Optional[List[float]] = None,
        seed: int = 1,
    ):
        self.N = N
        self.code_string = code_string
        self.token_names = token_names
        self.K = len(token_names)

        self.period = period
        self.pulse_len = pulse_len
        self.relax_len = relax_len

        self.rng = np.random.default_rng(seed)

        self.token_nodes = [
            self.rng.choice(N, nodes_per_token, replace=False)
            for _ in range(self.K)
        ]

        if amps is None:
            # default alternating sign to help separation
            amps = [1.0 if i % 2 == 0 else -1.0 for i in range(self.K)]
        self.amps = amps

    def token_at_slot(self, slot: int) -> str:
        return self.code_string[slot % len(self.code_string)]

    def env_fn(self, t: int, substrate: Substrate) -> np.ndarray:
        """
        Build current injection I(t) based on current slot token.
        """
        slot = t // self.period
        tok = self.token_at_slot(slot)
        k = self.token_names.index(tok)

        I = np.zeros(self.N, dtype=substrate.cfg.dtype)
        within = t % self.period
        if within < self.pulse_len:
            I[self.token_nodes[k]] = self.amps[k]
        return I

    def true_token_fn(self, t: int) -> str:
        slot = t // self.period
        return self.token_at_slot(slot)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():
    # -------------------------
    # Configs
    # -------------------------
    substrate_cfg = SubstrateConfig(
        N=150,
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

    period = 240
    pulse_len = 28
    relax_len = 120

    slot_cfg = SlotConfig(pulse_len=pulse_len, relax_len=relax_len)

    # repeating code string (edit freely)
    code_string = "ABBAAABBABABBBBA"
    token_names = ["A", "B"]

    env = CodeEnvironment(
        N=substrate_cfg.N,
        code_string=code_string,
        token_names=token_names,
        nodes_per_token=7,
        period=period,
        pulse_len=pulse_len,
        relax_len=relax_len,
        amps=[1.0, -1.0],
        seed=2,
    )

    run_cfg = RunConfig(
        total_steps=36000,
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
        true_token_fn=env.true_token_fn,
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
    true_tokens_samples = []

    skip_slots = 10
    for slot in range(skip_slots, slots_total):
        t0 = slot * period
        ts = slot_cfg.sample_index(t0)
        if ts < run_cfg.total_steps:
            sample_times.append(ts)
            samples.append(V_hist[ts])
            true_tokens_samples.append(env.token_at_slot(slot))

    sample_times = np.array(sample_times, dtype=np.int32)
    samples = np.array(samples)
    true_tokens_samples = np.array(true_tokens_samples)

    # -------------------------
    # Cluster attractors into tokens
    # -------------------------
    cl_cfg = CosineClusterConfig(threshold=0.996)
    cluster_ids, reps, sizes = cluster_cosine_incremental(samples, cl_cfg)

    mapping = build_supervised_mapping(cluster_ids, true_tokens_samples)
    decoded = decode_sequence(cluster_ids, mapping)
    acc = decode_accuracy(cluster_ids, true_tokens_samples, mapping)

    print("\nPhase 2 results")
    print("----------------")
    print(f"code_string: {code_string}")
    print(f"samples: {len(samples)}")
    print(f"clusters_found: {len(reps)}")
    print(f"cluster_sizes: {sizes}")
    print(f"cluster_to_token: {mapping}")
    print(f"decode_accuracy: {acc:.3f}")

    # -------------------------
    # Plots
    # -------------------------
    plot_fitness(fitness_hist, title="Fitness F(t) over early window", x_max=6000)

    plot_cluster_ids_per_slot(
        slot_ids=(sample_times // period),
        cluster_ids=cluster_ids,
        title="Emergent attractor cluster per token slot"
    )

    plot_token_chain(
        true_tokens=list(true_tokens_samples),
        decoded_tokens=decoded,
        title=f"Token chain: true vs emergent-decode (acc={acc:.2f})",
        max_len=60
    )

    plot_pca_samples(
        samples=samples,
        sample_labels=true_tokens_samples,
        title="Attractor samples by true token (PCA)"
    )

    # -------------------------
    # Save NPZ
    # -------------------------
    os.makedirs("runs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"runs/exp_token_chains_{stamp}.npz"

    np.savez_compressed(
        path,
        **out,
        sample_times=sample_times,
        attractor_samples=samples.astype(np.float32),
        attractor_id=cluster_ids.astype(np.int32),
        true_token_samples=true_tokens_samples,
        decoded_token_samples=np.array(decoded, dtype="U1"),
        cluster_reps=reps.astype(np.float32),
        cluster_sizes=np.array(sizes, dtype=np.int32),
    )

    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
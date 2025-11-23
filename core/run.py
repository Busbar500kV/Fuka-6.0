"""
Fuka-6.0 core run orchestrator
==============================

This module integrates:
    - Substrate
    - Plasticity
    - Environment callback
    - Metrics
    - Logging into canonical NPZ fields

Experiments call:
    run_simulation(cfg, env_fn, pl_cfg, slot_cfg, total_steps)

The run returns a dict with all arrays needed for analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

from .substrate import Substrate, SubstrateConfig
from .plasticity import PlasticityConfig, update_g
from .metrics import (
    fitness_from_dV,
    turbulence,
    l2_energy,
    avg_abs_voltage,
    voltage_entropy_proxy,
    g_entropy_proxy,
    SlotConfig,
    slot_index_at,
    in_slot_window,
)

Array = np.ndarray


@dataclass
class RunConfig:
    """
    Top-level configuration for a simulation run.
    """
    total_steps: int
    slot_period: Optional[int] = None  # None -> continuous run, no slot logic
    log_every: int = 1                 # not used yet; could support partial logging

    # canonical NPZ logging flags
    record_V: bool = True
    record_dV: bool = False
    record_metrics: bool = True

    dtype: str = "float32"


# -------------------------------------------------------------------
# Main run function
# -------------------------------------------------------------------

def run_simulation(
    substrate_cfg: SubstrateConfig,
    plasticity_cfg: PlasticityConfig,
    run_cfg: RunConfig,
    env_fn: Callable[[int, Substrate], Array],
    slot_cfg: Optional[SlotConfig] = None,
    true_token_fn: Optional[Callable[[int], str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Array]:
    """
    Execute a full simulation.

    Args:
        substrate_cfg: configuration for capacitor substrate
        plasticity_cfg: configuration for plasticity rules
        run_cfg: run control config
        env_fn: function (t, substrate) -> I(t)
        slot_cfg: optional slot sampling configuration
        true_token_fn: optional function returning ground truth symbol at slot t
        seed: optional RNG seed for reproducibility

    Returns:
        dict of canonical arrays for saving to NPZ.
    """

    if seed is not None:
        np.random.seed(seed)

    # Initialize substrate
    S = Substrate(substrate_cfg)

    # Prepare logging skeletons
    T = run_cfg.total_steps
    N = substrate_cfg.N

    V_hist = (
        np.zeros((T, N), dtype=run_cfg.dtype)
        if run_cfg.record_V
        else None
    )
    dV_hist = (
        np.zeros((T, N), dtype=run_cfg.dtype)
        if run_cfg.record_dV
        else None
    )

    fitness_hist = np.zeros(T, dtype=run_cfg.dtype) if run_cfg.record_metrics else None
    energy_hist = np.zeros(T, dtype=run_cfg.dtype) if run_cfg.record_metrics else None
    turbulence_hist = np.zeros(T, dtype=run_cfg.dtype) if run_cfg.record_metrics else None
    v_entropy_hist = np.zeros(T, dtype=run_cfg.dtype) if run_cfg.record_metrics else None
    g_entropy_hist = np.zeros(T, dtype=run_cfg.dtype) if run_cfg.record_metrics else None

    slot_index_arr = (
        np.zeros(T, dtype=np.int32)
        if run_cfg.slot_period is not None
        else None
    )

    true_token_arr = (
        np.empty(T, dtype="U1")
        if true_token_fn is not None
        else None
    )

    # Run loop
    for t in range(T):
        # ------------------------------
        # Environment input
        # ------------------------------
        I_t = S.drive(env_fn, t)

        # ------------------------------
        # Substrate step
        # ------------------------------
        V, dV = S.step(I_t)

        # ------------------------------
        # Plasticity update
        # ------------------------------
        g_new, F_used = update_g(
            V=V,
            g=S.g,
            dV=dV,
            cfg=plasticity_cfg,
            fitness=None,
            rng=S.rng,
        )
        S.set_g(g_new)

        # ------------------------------
        # Logging
        # ------------------------------
        if run_cfg.record_V:
            V_hist[t] = V
        if run_cfg.record_dV:
            dV_hist[t] = dV

        if run_cfg.record_metrics:
            fitness_hist[t] = F_used
            energy_hist[t] = l2_energy(V)
            turbulence_hist[t] = turbulence(dV)
            v_entropy_hist[t] = voltage_entropy_proxy(V)
            g_entropy_hist[t] = g_entropy_proxy(g_new)

        if slot_index_arr is not None:
            slot_index_arr[t] = slot_index_at(
                t, run_cfg.slot_period
            )

        if true_token_arr is not None:
            true_token_arr[t] = true_token_fn(t)

    # ------------------------------
    # Assemble results
    # ------------------------------
    out = {
        "V_hist": V_hist,
        "g_last": S.g.copy(),
    }

    if dV_hist is not None:
        out["dV_hist"] = dV_hist

    if run_cfg.record_metrics:
        out.update({
            "fitness_hist": fitness_hist,
            "energy_hist": energy_hist,
            "turbulence_hist": turbulence_hist,
            "v_entropy_hist": v_entropy_hist,
            "g_entropy_hist": g_entropy_hist,
        })

    if slot_index_arr is not None:
        out["slot_index"] = slot_index_arr

    if true_token_arr is not None:
        out["true_token"] = true_token_arr

    return out
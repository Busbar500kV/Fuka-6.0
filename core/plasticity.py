"""
Fuka-6.0 core plasticity
Local learning rules for conductances g_ij.

Default rule (from README):
    dg_ij/dt = eta * F(t) * (V_i V_j - alpha g_ij)

This module is intentionally small and physics-first:
- No backprop
- No global optimizer
- Only local correlations gated by stability pressure

Optionally supports:
- soft clipping
- symmetric g enforcement
- connection creation/pruning (disabled by default)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class PlasticityConfig:
    eta: float = 0.002          # learning rate
    alpha: float = 0.02         # decay
    dt: float = 0.05            # integrator step (should match substrate dt ideally)

    # Gating / stability pressure usage
    use_fitness_gate: bool = True
    fitness_floor: Optional[float] = None   # if set, clamp F(t) >= floor
    fitness_ceil: Optional[float] = None    # if set, clamp F(t) <= ceil

    # Conductance constraints
    symmetric_g: bool = False
    clip_g_min: float = 0.0
    clip_g_max: Optional[float] = None

    # Creation / pruning (off by default)
    enable_create_prune: bool = False
    prune_threshold: float = 1e-6     # values below pruned to 0
    create_rate: float = 0.0          # probability per step to create a tiny edge
    create_scale: float = 1e-5        # magnitude of new edges

    dtype: str = "float32"
    seed: Optional[int] = None


def compute_fitness(dV: Array) -> float:
    """
    Stability pressure:
        F(t) = -(1/N) sum_i (dV_i/dt)^2

    Args:
        dV: derivative of V at current step, shape (N,)

    Returns:
        scalar fitness (negative or zero; less negative = calmer)
    """
    dV = np.asarray(dV, dtype=np.float64)
    return -float(np.mean(dV ** 2))


def clamp_fitness(F: float, cfg: PlasticityConfig) -> float:
    """Optional clamping of fitness to stabilize learning."""
    if cfg.fitness_floor is not None:
        F = max(F, cfg.fitness_floor)
    if cfg.fitness_ceil is not None:
        F = min(F, cfg.fitness_ceil)
    return F


def update_g(
    V: Array,
    g: Array,
    dV: Array,
    cfg: PlasticityConfig,
    fitness: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, float]:
    """
    Update conductance matrix g using local plasticity rule.

    Args:
        V: voltages, shape (N,)
        g: conductances, shape (N,N)
        dV: voltage derivatives, shape (N,)
        cfg: plasticity config
        fitness: optional precomputed fitness
        rng: optional RNG (used if create/prune enabled)

    Returns:
        (g_new, fitness_used)
    """
    V = np.asarray(V, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    N = V.shape[0]
    if g.shape != (N, N):
        raise ValueError(f"g must be (N,N), got {g.shape}")

    if fitness is None:
        fitness = compute_fitness(dV)

    if cfg.use_fitness_gate:
        fitness = clamp_fitness(fitness, cfg)
        gate = fitness
    else:
        gate = 1.0

    # Outer product V_i V_j gives local correlation
    outer = np.outer(V, V)

    # Learning rule
    dg = cfg.eta * gate * (outer - cfg.alpha * g)

    g_new = g + cfg.dt * dg

    # Clean diagonal
    np.fill_diagonal(g_new, 0.0)

    # Optional symmetric enforcement
    if cfg.symmetric_g:
        g_new = 0.5 * (g_new + g_new.T)

    # Optional create/prune
    if cfg.enable_create_prune:
        if rng is None:
            rng = np.random.default_rng(cfg.seed)

        g_new = create_prune(g_new, cfg, rng=rng)

    # Clip
    g_new = clip_g(g_new, cfg)

    return g_new.astype(cfg.dtype), float(fitness)


def clip_g(g: Array, cfg: PlasticityConfig) -> Array:
    """Clip conductance matrix to configured bounds."""
    if cfg.clip_g_max is None:
        return np.clip(g, cfg.clip_g_min, np.inf)
    return np.clip(g, cfg.clip_g_min, cfg.clip_g_max)


def create_prune(g: Array, cfg: PlasticityConfig, rng: np.random.Generator) -> Array:
    """
    Optional structural evolution:
    - prune edges below prune_threshold
    - randomly create tiny new edges at create_rate

    This is OFF by default and should be enabled for Phase 5+ experiments.
    """
    g = np.asarray(g, dtype=np.float64)

    # Prune
    g[np.abs(g) < cfg.prune_threshold] = 0.0
    np.fill_diagonal(g, 0.0)

    # Create
    if cfg.create_rate > 0.0:
        mask = rng.random(g.shape) < cfg.create_rate
        # avoid diagonal creation
        np.fill_diagonal(mask, False)
        g[mask] += (rng.standard_normal(mask.sum()) * cfg.create_scale)

    if cfg.symmetric_g:
        g = 0.5 * (g + g.T)
        np.fill_diagonal(g, 0.0)

    return g
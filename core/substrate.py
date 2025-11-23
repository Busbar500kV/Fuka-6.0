"""
Fuka-6.0 core substrate
Capacitor network + environment injection + integrator.

No learning rules here. This module only evolves voltages V
given conductances g and input currents I(t).

Equation:
    C_i dV_i/dt = -lambda_i V_i + sum_j g_ij (V_j - V_i) + I_i(t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Any

import numpy as np


Array = np.ndarray


@dataclass
class SubstrateConfig:
    N: int = 200
    C: float = 1.0                 # capacitance (scalar or per-node)
    lam: float = 0.03              # leakage/decay (scalar or per-node)
    dt: float = 0.05               # integrator step
    init_v_std: float = 0.01
    init_g_scale: float = 1e-3
    symmetric_g: bool = False      # allow directed coupling if False
    clip_g_min: float = 0.0
    clip_g_max: Optional[float] = None  # None = unbounded above
    dtype: str = "float32"
    seed: Optional[int] = None


class Substrate:
    """
    Capacitor substrate:
      - Holds state V (voltages) and g (conductances)
      - Evolves V via Euler integration
      - Exposes drive() hook for environment -> I(t)
    """

    def __init__(self, cfg: SubstrateConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.N = cfg.N
        self.dt = float(cfg.dt)

        # Allow scalar or vector parameters
        self.C = self._as_vec(cfg.C, name="C")
        self.lam = self._as_vec(cfg.lam, name="lam")

        self.V = (self.rng.standard_normal(self.N) * cfg.init_v_std).astype(cfg.dtype)

        g0 = self.rng.random((self.N, self.N), dtype=np.float64) * cfg.init_g_scale
        np.fill_diagonal(g0, 0.0)

        if cfg.symmetric_g:
            g0 = 0.5 * (g0 + g0.T)

        self.g = g0.astype(cfg.dtype)

        # last dV (for metrics / fitness)
        self.dV_last = np.zeros(self.N, dtype=cfg.dtype)

    def _as_vec(self, x: float | Array, name: str) -> Array:
        if np.isscalar(x):
            return np.full(self.N, float(x), dtype=self.cfg.dtype)
        x = np.asarray(x, dtype=self.cfg.dtype)
        if x.shape != (self.N,):
            raise ValueError(f"{name} must be scalar or shape (N,), got {x.shape}")
        return x

    def set_g(self, g: Array) -> None:
        """Replace conductance matrix (used after plasticity update)."""
        g = np.asarray(g, dtype=self.cfg.dtype)
        if g.shape != (self.N, self.N):
            raise ValueError(f"g must be shape (N,N), got {g.shape}")
        np.fill_diagonal(g, 0.0)
        if self.cfg.symmetric_g:
            g = 0.5 * (g + g.T)
        self.g = self._clip_g(g)

    def _clip_g(self, g: Array) -> Array:
        if self.cfg.clip_g_max is None:
            return np.clip(g, self.cfg.clip_g_min, np.inf)
        return np.clip(g, self.cfg.clip_g_min, self.cfg.clip_g_max)

    def step(self, I_t: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        One Euler step.

        Args:
            I_t: input current vector shape (N,). If None, uses zeros.

        Returns:
            (V, dV) updated state and derivative at this step.
        """
        if I_t is None:
            I_t = np.zeros(self.N, dtype=self.cfg.dtype)
        else:
            I_t = np.asarray(I_t, dtype=self.cfg.dtype)
            if I_t.shape != (self.N,):
                raise ValueError(f"I_t must be shape (N,), got {I_t.shape}")

        # Coupling term: sum_j g_ij (V_j - V_i)
        # Compute diff matrix Vj - Vi efficiently
        diff = self.V[None, :] - self.V[:, None]      # shape (N,N)
        coupling = (self.g * diff).sum(axis=1)        # shape (N,)

        dV = (-self.lam * self.V + coupling + I_t) / self.C
        self.V = (self.V + self.dt * dV).astype(self.cfg.dtype)

        self.dV_last = dV.astype(self.cfg.dtype)
        return self.V, self.dV_last

    def drive(
        self,
        env_fn: Callable[[int, "Substrate"], Array],
        t: int
    ) -> Array:
        """
        Build I(t) from an environment callback.

        env_fn signature:
            env_fn(t: int, substrate: Substrate) -> I_t (shape N)

        This keeps core physics separate from experiment design.
        """
        I_t = env_fn(t, self)
        I_t = np.asarray(I_t, dtype=self.cfg.dtype)
        if I_t.shape != (self.N,):
            raise ValueError(f"env_fn must return shape (N,), got {I_t.shape}")
        return I_t

    def snapshot(self) -> Dict[str, Array]:
        """Minimal snapshot for logging/checkpointing."""
        return {
            "V": self.V.copy(),
            "g": self.g.copy(),
            "dV_last": self.dV_last.copy(),
        }
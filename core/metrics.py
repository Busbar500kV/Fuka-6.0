"""
Fuka-6.0 core metrics
=====================

This module encapsulates all measurement logic:
- fitness / turbulence
- energy norms
- entropy-like proxies
- attractor sampling helpers
- basin-stability helpers (optional)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


Array = np.ndarray


# ----------------------------------------------------------------------
# Basic metrics
# ----------------------------------------------------------------------

def fitness_from_dV(dV: Array) -> float:
    """
    Stability pressure:
        F = -(1/N) * sum_i (dV_i/dt)^2

    Larger (closer to zero) means calmer.
    """
    dV = np.asarray(dV, dtype=np.float64)
    return -float(np.mean(dV ** 2))


def turbulence(dV: Array) -> float:
    """
    Equivalent to -F, but returned as a positive magnitude.
    Useful for plotting "activity" directly.
    """
    dV = np.asarray(dV, dtype=np.float64)
    return float(np.mean(dV ** 2))


def l2_energy(V: Array) -> float:
    """Simple energy proxy: ||V||^2"""
    V = np.asarray(V, dtype=np.float64)
    return float(np.dot(V, V))


def avg_abs_voltage(V: Array) -> float:
    """Mean absolute voltage."""
    V = np.asarray(V, dtype=np.float64)
    return float(np.mean(np.abs(V)))


# ----------------------------------------------------------------------
# Entropy-like proxies (not true thermodynamic entropy)
# Used to track compression of state into attractors.
# ----------------------------------------------------------------------

def voltage_entropy_proxy(V: Array, bins: int = 50) -> float:
    """
    Histogram-based entropy proxy for V distribution.
    Useful to see compression during attractor formation.
    """
    V = np.asarray(V, dtype=np.float64)
    hist, edges = np.histogram(V, bins=bins, density=True)
    hist = hist + 1e-12  # avoid log(0)
    return -float(np.sum(hist * np.log(hist)))


def g_entropy_proxy(g: Array, bins: int = 50) -> float:
    """
    Entropy-like proxy for conductance distribution.
    """
    g = np.asarray(g, dtype=np.float64)
    flat = g.flatten()
    hist, edges = np.histogram(flat, bins=bins, density=True)
    hist = hist + 1e-12
    return -float(np.sum(hist * np.log(hist)))


# ----------------------------------------------------------------------
# Attractor sampling utilities
# ----------------------------------------------------------------------

@dataclass
class SlotConfig:
    """
    Configuration for slot-based attractor sampling.

    A slot is:
        [pulse ...] -> [relax ...] -> sample at time t_sample
    """
    pulse_len: int = 30
    relax_len: int = 100

    def sample_index(self, t0: int) -> int:
        """
        Given slot start t0, return the recommended sample time.
        """
        return int(t0 + self.pulse_len + self.relax_len)


def in_slot_window(t: int, slot_index: int, slot_period: int) -> bool:
    """Return True if time t is inside the slot window for given slot_index."""
    return (slot_index * slot_period) <= t < ((slot_index + 1) * slot_period)


def slot_index_at(t: int, slot_period: int) -> int:
    """Return slot number corresponding to time t."""
    return int(t // slot_period)


# ----------------------------------------------------------------------
# Transition graph helpers (Phase 3 onward)
# ----------------------------------------------------------------------

def build_transition_edges(attractor_ids: Array) -> Array:
    """
    Build transitions (i -> j) from sequential attractor IDs.
    Used to construct a directed transition graph.

    Args:
        attractor_ids: array of length S, each entry is cluster id

    Returns:
        edges: shape (E, 2)
    """
    attractor_ids = np.asarray(attractor_ids, dtype=np.int32)
    if len(attractor_ids) < 2:
        return np.zeros((0, 2), dtype=np.int32)

    src = attractor_ids[:-1]
    dst = attractor_ids[1:]
    edges = np.stack([src, dst], axis=1)
    return edges


# ----------------------------------------------------------------------
# Basin probing (optional advanced tool)
# ----------------------------------------------------------------------

def basin_similarity(a: Array, b: Array) -> float:
    """
    Cosine similarity between two attractor representatives.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = float(np.dot(a, b))
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return num / den


def assign_to_basin(
    sample: Array,
    reps: Array,
    threshold: float = 0.995
) -> Optional[int]:
    """
    Assign a sample vector to a basin represented by 'reps',
    using cosine similarity.

    Args:
        sample: shape (N,)
        reps: shape (K,N) attractor representatives
        threshold: minimum similarity

    Returns:
        basin index or None
    """
    sample = np.asarray(sample, dtype=np.float64)
    sims = [basin_similarity(sample, r) for r in reps]
    best = int(np.argmax(sims))
    return best if sims[best] > threshold else None
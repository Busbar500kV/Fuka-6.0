"""
Fuka-6.0 analysis: decode
=========================

This module handles:
    - mapping cluster IDs to symbolic tokens (A/B/C/... or numeric)
    - decoding sequences
    - computing majority mappings when ground-truth tokens exist
    - unsupervised token naming (e.g., T0, T1, T2, ...)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


Array = np.ndarray


# ---------------------------------------------------------------------
# Supervised mapping
# ---------------------------------------------------------------------

def build_supervised_mapping(
    cluster_ids: Array,
    true_tokens: Array
) -> Dict[int, str]:
    """
    Build mapping: cluster_id -> most common true token.

    Args:
        cluster_ids: (S,) attractor cluster IDs for sampled slots
        true_tokens: (S,) characters or labels provided by experiment

    Returns:
        dict cluster_id -> token label
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    true_tokens = np.asarray(true_tokens)

    mapping: Dict[int, str] = {}
    unique_clusters = np.unique(cluster_ids)

    for cid in unique_clusters:
        mask = (cluster_ids == cid)
        toks = true_tokens[mask]
        vals, counts = np.unique(toks, return_counts=True)
        mapping[cid] = vals[np.argmax(counts)]

    return mapping


def decode_sequence(
    cluster_ids: Array,
    mapping: Dict[int, str]
) -> List[str]:
    """
    Decode cluster sequence using mapping.

    Args:
        cluster_ids: (S,)
        mapping: dict cluster_id -> token label

    Returns:
        decoded token list, length S
    """
    return [mapping.get(int(cid), "?") for cid in cluster_ids]


def decode_accuracy(
    cluster_ids: Array,
    true_tokens: Array,
    mapping: Dict[int, str]
) -> float:
    """
    Compute fraction of decoded tokens that match true tokens.

    Args:
        cluster_ids
        true_tokens
        mapping

    Returns:
        accuracy in [0,1]
    """
    decoded = decode_sequence(cluster_ids, mapping)
    return float(np.mean(np.array(decoded) == np.array(true_tokens)))


# ---------------------------------------------------------------------
# Unsupervised token naming
# ---------------------------------------------------------------------

def build_unsupervised_labels(
    cluster_ids: Array,
    prefix: str = "T"
) -> Dict[int, str]:
    """
    Build unsupervised cluster label mapping:
        0 -> "T0"
        1 -> "T1"
        etc.

    Args:
        cluster_ids: (S,)
        prefix: label prefix

    Returns:
        dict cluster_id -> label
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    unique_ids = np.unique(cluster_ids)
    return {int(cid): f"{prefix}{i}" for i, cid in enumerate(unique_ids)}


# ---------------------------------------------------------------------
# Extract token chains
# ---------------------------------------------------------------------

def extract_token_chain(
    cluster_ids: Array,
    mapping: Dict[int, str]
) -> List[str]:
    """
    Convert cluster_ids into a token chain.

    Useful for Phase 2+ experiments where sequences matter.
    """
    return decode_sequence(cluster_ids, mapping)


def join_token_chain(tokens: List[str]) -> str:
    """Join token list into a string."""
    return "".join(tokens)


# ---------------------------------------------------------------------
# Transition counting
# ---------------------------------------------------------------------

def transition_counts(cluster_ids: Array) -> Dict[Tuple[int, int], int]:
    """
    Count transitions (i->j) between successive cluster IDs.

    Returns:
        dict {(i,j): count}
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    out: Dict[Tuple[int, int], int] = {}

    for a, b in zip(cluster_ids[:-1], cluster_ids[1:]):
        key = (int(a), int(b))
        out[key] = out.get(key, 0) + 1

    return out
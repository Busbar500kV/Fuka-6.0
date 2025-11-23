"""
Fuka-6.0 analysis: transition_graph
===================================

Builds a directed transition graph from attractor IDs (or decoded tokens).

Outputs:
    - adjacency matrices
    - edge lists with counts / probabilities
    - simple graph simplification utilities
    - minimal plotting helper (matplotlib only)

No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


@dataclass
class GraphConfig:
    """
    Configuration for graph building.
    """
    drop_self_loops: bool = False
    min_count: int = 1          # prune edges below this count
    normalize: bool = True      # output probabilities as well as counts


def build_transition_graph(
    ids: Array,
    cfg: GraphConfig = GraphConfig(),
) -> Dict[str, Any]:
    """
    Build directed transition graph from a sequence of attractor IDs.

    Args:
        ids: (S,) int cluster IDs in temporal order
        cfg: graph config

    Returns dict with:
        - K: number of states
        - unique_ids: sorted state ids
        - id_to_index: map original id -> compact index
        - counts: (K,K) transition counts
        - probs: (K,K) transition probabilities (row-normalized) if normalize
        - edges: (E,3) [src, dst, count] compact indices
    """
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) < 2:
        return {
            "K": 0,
            "unique_ids": np.array([], dtype=np.int32),
            "id_to_index": {},
            "counts": np.zeros((0, 0), dtype=np.int32),
            "probs": np.zeros((0, 0), dtype=np.float32),
            "edges": np.zeros((0, 3), dtype=np.int32),
        }

    unique_ids = np.unique(ids)
    K = len(unique_ids)
    id_to_index = {int(cid): i for i, cid in enumerate(unique_ids)}
    idx_seq = np.array([id_to_index[int(cid)] for cid in ids], dtype=np.int32)

    counts = np.zeros((K, K), dtype=np.int32)

    for a, b in zip(idx_seq[:-1], idx_seq[1:]):
        if cfg.drop_self_loops and a == b:
            continue
        counts[a, b] += 1

    # Prune weak edges
    if cfg.min_count > 1:
        counts[counts < cfg.min_count] = 0

    edges = []
    for i in range(K):
        for j in range(K):
            c = counts[i, j]
            if c > 0:
                edges.append((i, j, int(c)))
    edges = np.array(edges, dtype=np.int32)

    if cfg.normalize:
        row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
        row_sums[row_sums == 0] = 1.0
        probs = (counts / row_sums).astype(np.float32)
    else:
        probs = np.zeros_like(counts, dtype=np.float32)

    return {
        "K": K,
        "unique_ids": unique_ids,
        "id_to_index": id_to_index,
        "counts": counts,
        "probs": probs,
        "edges": edges,
    }


def simplify_graph_by_degree(
    counts: Array,
    max_out_degree: int = 3
) -> Array:
    """
    Keep only top-k outgoing edges by count for each node.

    Args:
        counts: (K,K)
        max_out_degree: keep this many strongest outgoing edges

    Returns:
        simplified counts (K,K)
    """
    counts = np.asarray(counts, dtype=np.int32)
    K = counts.shape[0]
    out = np.zeros_like(counts)

    for i in range(K):
        row = counts[i]
        if row.sum() == 0:
            continue
        top_idx = np.argsort(row)[::-1][:max_out_degree]
        for j in top_idx:
            if row[j] > 0:
                out[i, j] = row[j]

    return out


def edge_list_from_counts(counts: Array) -> List[Tuple[int, int, int]]:
    """
    Convert counts matrix to edge list.
    """
    counts = np.asarray(counts, dtype=np.int32)
    K = counts.shape[0]
    edges: List[Tuple[int, int, int]] = []
    for i in range(K):
        for j in range(K):
            c = int(counts[i, j])
            if c > 0:
                edges.append((i, j, c))
    return edges


# ---------------------------------------------------------------------
# Plotting helpers (Phase 3 quick look)
# ---------------------------------------------------------------------

def plot_adjacency(
    counts: Array,
    title: str = "Transition counts",
    show_values: bool = False
) -> None:
    """
    Heatmap for adjacency / counts.
    """
    counts = np.asarray(counts)
    plt.figure(figsize=(6, 5))
    plt.imshow(counts, aspect="auto")
    plt.title(title)
    plt.xlabel("to")
    plt.ylabel("from")
    plt.colorbar(label="count")

    if show_values:
        K = counts.shape[0]
        for i in range(K):
            for j in range(K):
                if counts[i, j] > 0:
                    plt.text(j, i, str(int(counts[i, j])),
                             ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_transition_graph(
    edges: Array,
    probs: Optional[Array] = None,
    labels: Optional[Dict[int, str]] = None,
    title: str = "Transition graph",
) -> None:
    """
    Very lightweight directed graph plot.
    Uses circular layout.

    Args:
        edges: (E,3) [src,dst,count]
        probs: (K,K) optional probabilities (for edge labels)
        labels: optional dict node_index -> label
    """
    edges = np.asarray(edges, dtype=np.int32)
    if edges.size == 0:
        print("No edges to plot.")
        return

    K = int(edges[:, :2].max()) + 1
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis("off")

    # nodes
    for i in range(K):
        x, y = xy[i]
        plt.scatter([x], [y], s=300)
        label = labels[i] if labels and i in labels else str(i)
        plt.text(x, y, label, ha="center", va="center", fontsize=10, color="white")

    # edges
    for src, dst, cnt in edges:
        x1, y1 = xy[src]
        x2, y2 = xy[dst]
        plt.arrow(
            x1, y1,
            (x2 - x1) * 0.85, (y2 - y1) * 0.85,
            length_includes_head=True,
            head_width=0.04,
            alpha=0.6,
            linewidth=1.2,
        )
        if probs is not None:
            p = float(probs[src, dst])
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            plt.text(xm, ym, f"{p:.2f}", fontsize=8)

    plt.tight_layout()
    plt.show()
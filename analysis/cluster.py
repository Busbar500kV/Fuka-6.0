"""
Fuka-6.0 analysis: attractor clustering
======================================

We treat each sampled post-relaxation voltage vector as a candidate attractor.
We cluster these vectors into discrete "tokens" using cosine similarity.

Primary algorithm:
    incremental clustering with cosine threshold

Optional fallback:
    PCA + density clustering (implemented without sklearn)

This module is used in Phase 1–4 experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------
# Core cosine clustering
# ---------------------------------------------------------------------

def cosine_similarity(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


@dataclass
class CosineClusterConfig:
    threshold: float = 0.995   # higher -> fewer clusters, tighter basins
    min_cluster_size: int = 1  # prune tiny clusters if desired


def cluster_cosine_incremental(
    samples: Array,
    cfg: CosineClusterConfig = CosineClusterConfig(),
) -> Tuple[Array, Array, List[int]]:
    """
    Incremental cosine clustering.

    Args:
        samples: shape (S, N)
        cfg: clustering config

    Returns:
        cluster_ids: shape (S,)
        reps: shape (K, N) representatives (means)
        sizes: list of cluster sizes
    """
    samples = np.asarray(samples, dtype=np.float64)
    S, N = samples.shape

    clusters: List[Dict[str, Array]] = []
    cluster_ids: List[int] = []

    for s in samples:
        assigned = False
        for ci, c in enumerate(clusters):
            if cosine_similarity(s, c["rep"]) >= cfg.threshold:
                c["members"].append(s)
                c["rep"] = np.mean(c["members"], axis=0)
                cluster_ids.append(ci)
                assigned = True
                break

        if not assigned:
            clusters.append({"rep": s.copy(), "members": [s]})
            cluster_ids.append(len(clusters) - 1)

    # Gather outputs
    reps = np.stack([c["rep"] for c in clusters], axis=0)
    sizes = [len(c["members"]) for c in clusters]

    # Optional pruning by size (relabeling is stable)
    if cfg.min_cluster_size > 1:
        keep = [i for i, sz in enumerate(sizes) if sz >= cfg.min_cluster_size]
        remap = {old: new for new, old in enumerate(keep)}
        cluster_ids = np.array([remap[c] for c in cluster_ids if c in remap], dtype=np.int32)
        reps = reps[keep]
        sizes = [sizes[i] for i in keep]
    else:
        cluster_ids = np.array(cluster_ids, dtype=np.int32)

    return cluster_ids, reps.astype(np.float32), sizes


# ---------------------------------------------------------------------
# PCA helper (used for optional fallback / visualization)
# ---------------------------------------------------------------------

def pca_project(samples: Array, n_components: int = 2) -> Tuple[Array, Array, Array]:
    """
    Simple PCA via SVD.

    Args:
        samples: (S, N)
        n_components: number of principal components

    Returns:
        projected: (S, n_components)
        components: (n_components, N)
        mean: (N,)
    """
    X = np.asarray(samples, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    components = VT[:n_components]
    projected = Xc @ components.T

    return projected, components, mean.squeeze()


# ---------------------------------------------------------------------
# Optional lightweight DBSCAN (no sklearn)
# ---------------------------------------------------------------------

@dataclass
class DBSCANConfig:
    eps: float = 0.15
    min_samples: int = 3


def dbscan(points: Array, cfg: DBSCANConfig = DBSCANConfig()) -> Array:
    """
    Tiny DBSCAN implementation for fallback clustering on PCA space.

    Args:
        points: (S, d)
        cfg: eps radius, min_samples

    Returns:
        labels: (S,) cluster labels (-1 = noise)
    """
    P = np.asarray(points, dtype=np.float64)
    S = P.shape[0]
    labels = -np.ones(S, dtype=np.int32)
    visited = np.zeros(S, dtype=bool)

    def neighbors(i: int) -> List[int]:
        dists = np.linalg.norm(P - P[i], axis=1)
        return list(np.where(dists <= cfg.eps)[0])

    cluster_id = 0
    for i in range(S):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors(i)

        if len(neigh) < cfg.min_samples:
            labels[i] = -1
            continue

        # start new cluster
        labels[i] = cluster_id
        seed_set = neigh.copy()

        while seed_set:
            j = seed_set.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors(j)
                if len(neigh_j) >= cfg.min_samples:
                    # expand
                    seed_set.extend([x for x in neigh_j if x not in seed_set])
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels


def cluster_with_fallback(
    samples: Array,
    cosine_cfg: CosineClusterConfig = CosineClusterConfig(),
    use_dbscan_fallback: bool = False,
    dbscan_cfg: DBSCANConfig = DBSCANConfig(),
) -> Tuple[Array, Array, List[int]]:
    """
    Run cosine clustering first; optionally fallback to DBSCAN on PCA
    if cosine produces extreme fragmentation.

    Args:
        samples: (S, N)

    Returns:
        cluster_ids, reps, sizes
    """
    cids, reps, sizes = cluster_cosine_incremental(samples, cosine_cfg)

    if not use_dbscan_fallback:
        return cids, reps, sizes

    # Heuristic: if too many clusters relative to samples, fallback
    if len(reps) > max(10, 0.5 * len(samples)):
        proj, comps, mean = pca_project(samples, n_components=3)
        labels = dbscan(proj, cfg=dbscan_cfg)

        # Relabel noise as separate clusters for stability
        noise = np.where(labels == -1)[0]
        if len(noise) > 0:
            base = labels.max() + 1
            for k, idx in enumerate(noise):
                labels[idx] = base + k

        # Build reps/sizes from labels
        K = labels.max() + 1
        reps2 = []
        sizes2 = []
        for k in range(K):
            members = samples[labels == k]
            reps2.append(members.mean(axis=0))
            sizes2.append(len(members))
        reps2 = np.stack(reps2, axis=0).astype(np.float32)

        return labels.astype(np.int32), reps2, sizes2

    return cids, reps, sizes
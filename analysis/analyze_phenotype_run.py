# analysis/analyze_phenotype_run.py
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ----------------------------
# Helpers
# ----------------------------

def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _entropy_bits_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy in bits given integer counts."""
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _topk(items: List[Tuple], k: int) -> List[Tuple]:
    return sorted(items, key=lambda t: t[0], reverse=True)[:k]


# ----------------------------
# Loading (robust)
# ----------------------------

@dataclass
class PhenotypeRun:
    npz_path: str
    fitness_hist: Optional[np.ndarray]
    E_hist: Optional[np.ndarray]
    attractor_id: Optional[np.ndarray]
    cluster_sizes: Optional[np.ndarray]
    sample_times: Optional[np.ndarray]
    unsupervised_token_samples: Optional[np.ndarray]

    # optional extras
    regime_hist: Optional[np.ndarray] = None
    hidden_regime_labels: Optional[np.ndarray] = None
    substrate_readout_hist: Optional[np.ndarray] = None


def load_phenotype_run(npz_path: str) -> PhenotypeRun:
    data = np.load(npz_path, allow_pickle=False)

    # These are the keys we *try* to read. Some runs won't have all of them.
    fitness_hist = data["fitness_hist"] if "fitness_hist" in data else None
    E_hist = data["E_hist"] if "E_hist" in data else None

    attractor_id = data["attractor_id"] if "attractor_id" in data else None
    cluster_sizes = data["cluster_sizes"] if "cluster_sizes" in data else None
    sample_times = data["sample_times"] if "sample_times" in data else None

    # tokens are OPTIONAL: synthesize from attractor_id if missing
    if "unsupervised_token_samples" in data:
        tokens = data["unsupervised_token_samples"]
    elif attractor_id is not None:
        ids = _as_1d(attractor_id).astype(int)
        tokens = np.array([f"T{i}" for i in ids], dtype="U16")
    else:
        tokens = None

    run = PhenotypeRun(
        npz_path=npz_path,
        fitness_hist=fitness_hist,
        E_hist=E_hist,
        attractor_id=attractor_id,
        cluster_sizes=cluster_sizes,
        sample_times=sample_times,
        unsupervised_token_samples=tokens,
        regime_hist=data["regime_hist"] if "regime_hist" in data else None,
        hidden_regime_labels=data["hidden_regime_labels"] if "hidden_regime_labels" in data else None,
        substrate_readout_hist=data["substrate_readout_hist"] if "substrate_readout_hist" in data else None,
    )

    return run


# ----------------------------
# Analysis
# ----------------------------

def summarize_fitness(fitness_hist: Optional[np.ndarray]) -> None:
    if fitness_hist is None:
        print("\n--- Fitness F(t) ---")
        print("fitness_hist: (missing)")
        return

    F = _as_1d(fitness_hist).astype(np.float64)
    print("\n--- Fitness F(t) ---")
    print(f"length          : {len(F)}")
    print(f"F_min / F_max   : {F.min():.4f} / {F.max():.4f}")
    print(f"F_mean / F_std  : {F.mean():.4f} / {F.std():.4f}")
    print(f"F_first 5       : {np.array2string(F[:5], precision=4)}")
    print(f"F_last  5       : {np.array2string(F[-5:], precision=4)}")


def summarize_environment(E_hist: Optional[np.ndarray]) -> None:
    if E_hist is None:
        print("\n--- Environment E(t) ---")
        print("E_hist: (missing)")
        return

    E = _as_1d(E_hist).astype(np.float64)
    print("\n--- Environment E(t) ---")
    print(f"E_hist length   : {len(E)}")
    print(f"E_min / E_max   : {E.min():.3f} / {E.max():.3f}")
    print(f"E_final         : {E[-1]:.3f}")


def alphabet_stats(attractor_id: Optional[np.ndarray],
                   cluster_sizes: Optional[np.ndarray],
                   core_threshold: int) -> None:
    print("\n--- Alphabet statistics ---")

    if attractor_id is None:
        print("attractor_id: (missing)")
        return

    ids = _as_1d(attractor_id).astype(int)
    total_samples = len(ids)

    # cluster_sizes might not include all observed IDs (depending on how saved)
    # So compute counts from ids directly.
    uniq, counts = np.unique(ids, return_counts=True)
    total_clusters = len(uniq)

    # Core clusters: those with count >= core_threshold (based on observed samples)
    core_mask = counts >= core_threshold
    core_clusters = uniq[core_mask]
    core_counts = counts[core_mask]
    core_coverage = float(core_counts.sum() / total_samples) if total_samples else 0.0

    # Coverage top-K clusters
    sorted_counts = np.sort(counts)[::-1]
    def cov_top(k: int) -> float:
        if total_samples == 0:
            return 0.0
        return float(sorted_counts[:k].sum() / total_samples) if len(sorted_counts) else 0.0

    print(f"total_samples       : {total_samples}")
    print(f"total_clusters      : {total_clusters}")
    print(f"core_threshold      : {core_threshold}")
    print(f"core_clusters       : {len(core_clusters)}")
    print(f"core_coverage       : {core_coverage:.3f}")
    print(f"coverage top-5  : {cov_top(5):.3f}")
    print(f"coverage top-10 : {cov_top(10):.3f}")
    print(f"coverage top-20 : {cov_top(20):.3f}")
    print(f"coverage top-50 : {cov_top(50):.3f}")

    print("\nlargest cluster sizes (top 15):")
    top15 = sorted_counts[:15]
    print(np.array2string(top15, separator=" "))

    if cluster_sizes is not None:
        cs = _as_1d(cluster_sizes).astype(int)
        print(f"\ncluster_sizes array length: {len(cs)}")
        print("cluster_sizes (sorted, top 15):", np.sort(cs)[::-1][:15])

    # Core entropies: for each core cluster, entropy of NEXT token distribution.
    # This gives “grammar branching” measure.
    if total_samples >= 2 and len(core_clusters) > 0:
        print("\nCore cluster entropies (bits):")
        # Build transitions
        next_ids = ids[1:]
        cur_ids = ids[:-1]
        for cid in core_clusters:
            mask = cur_ids == cid
            if mask.sum() == 0:
                continue
            nxt = next_ids[mask]
            u2, c2 = np.unique(nxt, return_counts=True)
            H = _entropy_bits_from_counts(c2)
            print(f"  cluster {cid:>3d}  count = {mask.sum():>4d}   H = {H:.3f}")


def transition_stats(attractor_id: Optional[np.ndarray],
                     E_hist: Optional[np.ndarray],
                     core_threshold: int,
                     max_print: int = 10) -> None:
    if attractor_id is None:
        return

    ids = _as_1d(attractor_id).astype(int)
    if len(ids) < 2:
        return

    # Determine core clusters by frequency in ids
    uniq, counts = np.unique(ids, return_counts=True)
    core = set(uniq[counts >= core_threshold].tolist())

    cur = ids[:-1]
    nxt = ids[1:]

    # global core->core transitions
    trans_counts: Dict[Tuple[int, int], int] = {}
    from_counts: Dict[int, int] = {}
    for a, b in zip(cur, nxt):
        if a in core and b in core:
            trans_counts[(a, b)] = trans_counts.get((a, b), 0) + 1
            from_counts[a] = from_counts.get(a, 0) + 1

    if len(trans_counts) == 0:
        print("\nTop core->core transitions (global):")
        print("  (none)")
    else:
        scored = []
        for (a, b), c in trans_counts.items():
            p = c / from_counts[a]
            scored.append((p, c, a, b))
        scored.sort(reverse=True, key=lambda t: (t[0], t[1]))
        print("\nTop core->core transitions (global):")
        for p, c, a, b in scored[:max_print]:
            print(f"  P({a:>3d}->{b:>3d}) = {p:6.3f}   count = {c:>4d}")

    # Split by environment band (if available)
    if E_hist is None:
        return

    E = _as_1d(E_hist).astype(np.float64)

    # We need E at sample times; if E is recorded per step and attractor_id per sample,
    # we approximate by using evenly-spaced mapping if lengths differ.
    # Best case: E_hist length == total_steps and sample_times exists elsewhere,
    # but here we only have E_hist and ids. So do robust mapping:
    if len(E) == len(ids):
        E_s = E
    else:
        # map sample index i -> E index round(i*(len(E)-1)/(len(ids)-1))
        if len(ids) <= 1:
            return
        idx = np.round(np.linspace(0, len(E) - 1, num=len(ids))).astype(int)
        E_s = E[idx]

    med = float(np.median(E_s))
    low_mask = E_s[:-1] < med
    high_mask = ~low_mask

    print("\nEnvironment split:")
    print(f"  median E(sample) = {med:.3f}")
    print(f"  low-E transitions   : {int(low_mask.sum())}")
    print(f"  high-E transitions  : {int(high_mask.sum())}")

    def _band(name: str, band_mask: np.ndarray) -> None:
        band_counts: Dict[Tuple[int, int], int] = {}
        band_from: Dict[int, int] = {}
        for (a, b, m) in zip(cur, nxt, band_mask):
            if not m:
                continue
            if a in core and b in core:
                band_counts[(a, b)] = band_counts.get((a, b), 0) + 1
                band_from[a] = band_from.get(a, 0) + 1
        print(f"\n{name} band: top core->core transitions (prob, count, i->j):")
        if not band_counts:
            print("  (none)")
            return
        scored2 = []
        for (a, b), c in band_counts.items():
            p = c / band_from[a]
            scored2.append((p, c, a, b))
        scored2.sort(reverse=True, key=lambda t: (t[0], t[1]))
        for p, c, a, b in scored2[:max_print]:
            print(f"  P({a:>3d}->{b:>3d}) = {p:6.3f}   count = {c:>4d}")

    _band("Low-E", low_mask)
    _band("High-E", high_mask)


def ngram_stats(attractor_id: Optional[np.ndarray],
                tokens: Optional[np.ndarray],
                core_threshold: int,
                E_hist: Optional[np.ndarray],
                max_print: int = 20) -> None:
    if attractor_id is None or tokens is None:
        return

    ids = _as_1d(attractor_id).astype(int)
    toks = _as_1d(tokens).astype(str)
    if len(ids) != len(toks):
        # should not happen; if it does, fall back to IDs only
        toks = np.array([f"T{i}" for i in ids], dtype="U16")

    uniq, counts = np.unique(ids, return_counts=True)
    core = set(uniq[counts >= core_threshold].tolist())

    # Core bigrams
    if len(ids) >= 2:
        pairs = []
        pair_counts: Dict[Tuple[int, int], int] = {}
        total = 0
        for a, b in zip(ids[:-1], ids[1:]):
            if a in core and b in core:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
                total += 1
        print("\nGlobal core bigrams:")
        if total == 0:
            print("  (none)")
        else:
            scored = []
            for (a, b), c in pair_counts.items():
                scored.append((c / total, c, a, b))
            scored.sort(reverse=True)
            for p, c, a, b in scored[:max_print]:
                print(f"  ({a}:{toks[np.where(ids==a)[0][0]]}, {b}:{toks[np.where(ids==b)[0][0]]})"
                      f"{' ' * 30}".rstrip() + f"  count = {c:>4d}   P = {p:6.3f}")

    # Core 3-grams
    if len(ids) >= 3:
        tri_counts: Dict[Tuple[int, int, int], int] = {}
        total3 = 0
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            if a in core and b in core and c in core:
                tri_counts[(a, b, c)] = tri_counts.get((a, b, c), 0) + 1
                total3 += 1
        print("\nGlobal core 3-grams:")
        print(f"  distinct 3-grams: {len(tri_counts)}")
        print(f"  total    3-gram count: {total3}")
        if total3 > 0:
            scored3 = []
            for (a, b, c), cnt in tri_counts.items():
                scored3.append((cnt / total3, cnt, a, b, c))
            scored3.sort(reverse=True)
            for p, cnt, a, b, c in scored3[:max_print]:
                print(f"  ({a}:{toks[np.where(ids==a)[0][0]]}, {b}:{toks[np.where(ids==b)[0][0]]}, {c}:{toks[np.where(ids==c)[0][0]]})"
                      f"{' ' * 20}".rstrip() + f"  count = {cnt:>4d}   P = {p:6.3f}")

    # Environment bands bigrams (same logic you saw earlier)
    if E_hist is None or len(ids) < 2:
        return

    E = _as_1d(E_hist).astype(np.float64)
    if len(E) == len(ids):
        E_s = E
    else:
        idx = np.round(np.linspace(0, len(E) - 1, num=len(ids))).astype(int)
        E_s = E[idx]

    med = float(np.median(E_s))
    low = E_s < med
    high = ~low

    print("\nEnvironment bands:")
    print(f"  median E(sample)   : {med:.3f}")
    print(f"  low-E samples      : {int(low.sum())}")
    print(f"  high-E samples     : {int(high.sum())}")

    def band_bigrams(name: str, mask: np.ndarray) -> None:
        total = 0
        cnts: Dict[Tuple[int, int], int] = {}
        for i in range(len(ids) - 1):
            if not mask[i]:
                continue
            a, b = ids[i], ids[i + 1]
            if a in core and b in core:
                cnts[(a, b)] = cnts.get((a, b), 0) + 1
                total += 1
        print(f"\n{name} core bigrams:")
        if total == 0:
            print("  (none)")
            return
        scored = []
        for (a, b), c in cnts.items():
            scored.append((c / total, c, a, b))
        scored.sort(reverse=True)
        for p, c, a, b in scored[:max_print]:
            print(f"  ({a}:T{a}, {b}:T{b})".ljust(44) + f"count = {c:>4d}   P = {p:6.3f}")

    band_bigrams("low-E", low[:-1])
    band_bigrams("high-E", high[:-1])


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Path to a phenotype/longrun/autotune NPZ")
    ap.add_argument("--core-threshold", type=int, default=10, help="Min count for a cluster to be considered 'core'")
    ap.add_argument("--no-plots", action="store_true", help="(reserved) No plotting in this script")
    args = ap.parse_args()

    run = load_phenotype_run(args.npz)

    # header
    print(f"Loading NPZ: {run.npz_path}")

    # run summary
    samples = int(len(_as_1d(run.attractor_id))) if run.attractor_id is not None else 0
    clusters_found = 0
    if run.attractor_id is not None:
        clusters_found = int(np.unique(_as_1d(run.attractor_id).astype(int)).shape[0])

    print("\n--- Phenotype run summary ---")
    print(f"file           : {run.npz_path}")
    print(f"samples        : {samples}")
    print(f"clusters_found : {clusters_found}")

    alphabet_stats(run.attractor_id, run.cluster_sizes, core_threshold=args.core_threshold)
    summarize_environment(run.E_hist)

    # extra structure reports (transitions + ngrams)
    transition_stats(run.attractor_id, run.E_hist, core_threshold=args.core_threshold, max_print=10)
    ngram_stats(run.attractor_id, run.unsupervised_token_samples, core_threshold=args.core_threshold, E_hist=run.E_hist, max_print=20)

    # fitness at end (nice to keep at bottom)
    summarize_fitness(run.fitness_hist)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from collections import Counter, defaultdict

import numpy as np


# ============================================================
# Data model
# ============================================================

@dataclass
class PhenotypeRun:
    path: str
    attractor_id: np.ndarray                 # [S]
    cluster_sizes: np.ndarray                # [K] (optional but recommended)
    unsupervised_token_samples: Optional[np.ndarray] = None  # [S] strings
    E_hist: Optional[np.ndarray] = None      # [S] preferred (sample-aligned) OR [T] raw env history OR None


# ============================================================
# Helpers (robust to missing / empty E)
# ============================================================

def _as_np(a, dtype=None) -> np.ndarray:
    if a is None:
        return np.array([], dtype=dtype if dtype is not None else np.float32)
    arr = np.asarray(a)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr

def _is_empty(a) -> bool:
    return a is None or (hasattr(a, "size") and a.size == 0)

def _safe_minmax(x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if x is None or x.size == 0:
        return None, None
    return float(np.min(x)), float(np.max(x))

def _maybe_sample_aligned_E(E_hist: Optional[np.ndarray], ids_len: int) -> Optional[np.ndarray]:
    """
    We only use E for band splitting if it is sample-aligned (len(E)==len(ids)).
    Otherwise return None.
    """
    if E_hist is None:
        return None
    E = np.asarray(E_hist)
    if E.size == 0:
        return None
    if E.shape[0] != ids_len:
        return None
    return E.astype(np.float32, copy=False)


# ============================================================
# Loading
# ============================================================

def load_phenotype_run(npz_path: str) -> PhenotypeRun:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    npz = np.load(npz_path, allow_pickle=False)

    # Required: attractor_id
    if "attractor_id" in npz:
        attractor_id = _as_np(npz["attractor_id"], np.int64)
    elif "attractor_ids" in npz:
        attractor_id = _as_np(npz["attractor_ids"], np.int64)
    else:
        raise KeyError("Required key 'attractor_id' missing in NPZ")

    # Optional: cluster_sizes (if not present, we can derive a vector of counts per cluster ID)
    if "cluster_sizes" in npz:
        cluster_sizes = _as_np(npz["cluster_sizes"], np.int64)
    else:
        # derive sparse -> dense (max_id+1)
        if attractor_id.size == 0:
            cluster_sizes = np.array([], dtype=np.int64)
        else:
            K = int(np.max(attractor_id)) + 1
            cluster_sizes = np.bincount(attractor_id, minlength=K).astype(np.int64)

    # Optional token samples
    token_samples = None
    if "unsupervised_token_samples" in npz:
        token_samples = _as_np(npz["unsupervised_token_samples"])

    # Optional environment history
    E_hist = None
    for k in ("E_hist", "environment_E_hist", "env_E_hist", "E"):
        if k in npz:
            E_hist = _as_np(npz[k], np.float32)
            break

    return PhenotypeRun(
        path=npz_path,
        attractor_id=attractor_id,
        cluster_sizes=cluster_sizes,
        unsupervised_token_samples=token_samples,
        E_hist=E_hist,
    )


# ============================================================
# Summaries
# ============================================================

def summarize_alphabet(ids: np.ndarray, core_threshold: int = 10) -> Dict[str, float]:
    ids = _as_np(ids, np.int64)
    total_samples = int(ids.size)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "total_clusters": 0,
            "core_clusters": 0,
            "core_coverage": 0.0,
            "cov_top5": 0.0,
            "cov_top10": 0.0,
            "cov_top20": 0.0,
            "cov_top50": 0.0,
        }

    counts = np.bincount(ids)
    total_clusters = int(np.count_nonzero(counts))
    sorted_counts = np.sort(counts[counts > 0])[::-1]

    def cov_top(n: int) -> float:
        if sorted_counts.size == 0:
            return 0.0
        return float(np.sum(sorted_counts[: min(n, sorted_counts.size)]) / total_samples)

    core_clusters = int(np.sum(sorted_counts >= core_threshold))
    core_coverage = float(np.sum(sorted_counts[sorted_counts >= core_threshold]) / total_samples) if core_clusters > 0 else 0.0

    return {
        "total_samples": float(total_samples),
        "total_clusters": float(total_clusters),
        "core_clusters": float(core_clusters),
        "core_coverage": float(core_coverage),
        "cov_top5": cov_top(5),
        "cov_top10": cov_top(10),
        "cov_top20": cov_top(20),
        "cov_top50": cov_top(50),
    }


def print_run_header(run: PhenotypeRun, core_threshold: int) -> None:
    print("\n--- Phenotype run summary ---")
    print(f"file           : {run.path}")
    print(f"samples        : {int(run.attractor_id.size)}")
    # clusters_found = count of unique ids
    uniq = int(np.unique(run.attractor_id).size) if run.attractor_id.size else 0
    print(f"clusters_found : {uniq}")

    stats = summarize_alphabet(run.attractor_id, core_threshold=core_threshold)

    print("\n--- Alphabet statistics ---")
    print(f"total_samples       : {int(stats['total_samples'])}")
    print(f"total_clusters      : {int(stats['total_clusters'])}")
    print(f"core_threshold      : {core_threshold}")
    print(f"core_clusters       : {int(stats['core_clusters'])}")
    print(f"core_coverage       : {stats['core_coverage']:.3f}")
    print(f"coverage top-5  : {stats['cov_top5']:.3f}")
    print(f"coverage top-10 : {stats['cov_top10']:.3f}")
    print(f"coverage top-20 : {stats['cov_top20']:.3f}")
    print(f"coverage top-50 : {stats['cov_top50']:.3f}")

    # Largest cluster sizes
    cs = _as_np(run.cluster_sizes, np.int64)
    if cs.size:
        top = np.sort(cs)[::-1][:15]
        print("\nlargest cluster sizes (top 15):")
        print(top)
        print(f"\ncluster_sizes array length: {cs.size}")
        print(f"cluster_sizes (sorted, top 15): {top}")
    else:
        print("\ncluster_sizes: (missing/empty)")

    # Environment summary (robust)
    summarize_environment(run.E_hist, ids_len=int(run.attractor_id.size))


def summarize_environment(E_hist: Optional[np.ndarray], ids_len: int) -> None:
    print("\n--- Environment E(t) ---")
    if E_hist is None:
        print("E_hist          : (missing)")
        return

    E = np.asarray(E_hist)
    print(f"E_hist length   : {int(E.size)}")
    if E.size == 0:
        print("E_hist          : (empty) — this run did not record environment history")
        return

    mn, mx = _safe_minmax(E)
    if mn is not None:
        print(f"E_min / E_max   : {mn:.3f} / {mx:.3f}")
        print(f"E_final         : {float(E[-1]):.3f}")

    # Tell user whether it's usable for band split
    E_s = _maybe_sample_aligned_E(E_hist, ids_len)
    if E_s is None:
        print(f"E(sample)       : (not sample-aligned) — band split will be skipped")
    else:
        print(f"E(sample)       : sample-aligned OK")


# ============================================================
# Transitions / grammar (robust to missing E)
# ============================================================

def core_set_from_ids(ids: np.ndarray, core_threshold: int) -> List[int]:
    ids = _as_np(ids, np.int64)
    if ids.size == 0:
        return []
    counts = Counter(ids.tolist())
    core = sorted([k for k, v in counts.items() if v >= core_threshold])
    return core

def transition_stats(ids: np.ndarray, E_hist: Optional[np.ndarray], core_threshold: int = 10, max_print: int = 10) -> None:
    """
    Prints top core->core transitions globally.
    If E_hist is sample-aligned (len==len(ids)), also prints low/high band transitions.
    Otherwise band sections are skipped cleanly.
    """
    ids = _as_np(ids, np.int64)
    if ids.size < 2:
        print("\nTop core->core transitions (global):")
        print("  (none)")
        return

    core = core_set_from_ids(ids, core_threshold=core_threshold)
    core_set = set(core)

    big = Counter()
    out_counts = Counter()

    for a, b in zip(ids[:-1], ids[1:]):
        if a in core_set and b in core_set:
            big[(int(a), int(b))] += 1
            out_counts[int(a)] += 1

    print("\nTop core->core transitions (global):")
    if not big:
        print("  (none)")
    else:
        for (a, b), c in big.most_common(max_print):
            p = c / out_counts[a] if out_counts[a] else 0.0
            print(f"  P({a:3d}->{b:3d}) = {p:6.3f}   count = {c:4d}")

    # --- Optional env split ---
    E = _maybe_sample_aligned_E(E_hist, ids_len=int(ids.size))
    if E is None:
        if E_hist is None:
            print("\nEnvironment split: (skipped) E_hist missing")
        else:
            Eh = np.asarray(E_hist)
            if Eh.size == 0:
                print("\nEnvironment split: (skipped) E_hist empty")
            else:
                print("\nEnvironment split: (skipped) E_hist not sample-aligned with attractor_id")
        return

    med = float(np.median(E))
    low_idx = np.where(E[:-1] < med)[0]
    high_idx = np.where(E[:-1] >= med)[0]

    print("\nEnvironment split:")
    print(f"  median E(sample) = {med:.3f}")
    print(f"  low-E transitions   : {int(low_idx.size)}")
    print(f"  high-E transitions  : {int(high_idx.size)}")

    def band_stats(sel_idx: np.ndarray, name: str) -> None:
        big_b = Counter()
        out_b = Counter()
        for i in sel_idx:
            a = int(ids[i]); b = int(ids[i + 1])
            if a in core_set and b in core_set:
                big_b[(a, b)] += 1
                out_b[a] += 1

        print(f"\n{name} band: top core->core transitions (prob, count, i->j):")
        if not big_b:
            print("  (none)")
            return
        for (a, b), c in big_b.most_common(max_print):
            p = c / out_b[a] if out_b[a] else 0.0
            print(f"  P({a:3d}->{b:3d}) = {p:6.3f}   count = {c:4d}")

    band_stats(low_idx, "Low-E")
    band_stats(high_idx, "High-E")


def ngram_stats(ids: np.ndarray, core_threshold: int = 10, max_print: int = 10) -> None:
    ids = _as_np(ids, np.int64)
    if ids.size < 3:
        print("\nGlobal core 3-grams:")
        print("  (none)")
        return

    core = core_set_from_ids(ids, core_threshold=core_threshold)
    core_set = set(core)

    tri = Counter()
    total = 0
    for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
        if a in core_set and b in core_set and c in core_set:
            tri[(int(a), int(b), int(c))] += 1
            total += 1

    print("\nGlobal core 3-grams:")
    if total == 0:
        print("  (none)")
        return

    print(f"  distinct 3-grams: {len(tri)}")
    print(f"  total    3-gram count: {total}")
    for (a, b, c), cnt in tri.most_common(max_print):
        p = cnt / total
        print(f"  ({a:3d}, {b:3d}, {c:3d})                              count = {cnt:4d}   P = {p:6.3f}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None, help="Path to NPZ file. Defaults to latest phenotype-like NPZ in ./runs.")
    ap.add_argument("--core-threshold", type=int, default=10)
    ap.add_argument("--no-plots", action="store_true")  # kept for compatibility; we don't plot here
    args = ap.parse_args()

    npz_path = args.npz
    if not npz_path:
        # pick latest NPZ in runs that looks like phenotype
        runs_dir = "runs"
        cands = []
        if os.path.isdir(runs_dir):
            for fn in os.listdir(runs_dir):
                if fn.endswith(".npz") and ("phenotype" in fn):
                    cands.append(os.path.join(runs_dir, fn))
        if not cands:
            raise FileNotFoundError("No phenotype NPZ found in ./runs. Provide --npz path.")
        npz_path = sorted(cands, key=lambda p: os.path.getmtime(p))[-1]

    print(f"Loading NPZ: {npz_path}")
    run = load_phenotype_run(npz_path)

    print_run_header(run, core_threshold=args.core_threshold)

    # transitions and 3-grams (safe when E missing)
    transition_stats(run.attractor_id, run.E_hist, core_threshold=args.core_threshold, max_print=10)
    ngram_stats(run.attractor_id, core_threshold=args.core_threshold, max_print=10)


if __name__ == "__main__":
    main()
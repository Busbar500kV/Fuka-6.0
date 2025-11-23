"""
tools/view_npz.py
-----------------

Quick NPZ inspector.

Shows:
  - keys
  - shapes / dtypes
  - basic stats for numeric arrays
  - metadata dict if present

Usage:
    python tools/view_npz.py runs/exp_token_chains_20251122_120101.npz

Optional:
    python tools/view_npz.py runs/file.npz --head 5
"""

from __future__ import annotations

import argparse
import numpy as np
from typing import Any, Dict, Optional


def summarize_array(name: str, arr: np.ndarray, head: int = 0) -> None:
    print(f"\n[{name}]")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

    if np.issubdtype(arr.dtype, np.number):
        flat = arr.ravel()
        print(f"  min / max: {float(flat.min()):.4g} / {float(flat.max()):.4g}")
        print(f"  mean / std: {float(flat.mean()):.4g} / {float(flat.std()):.4g}")

        if head > 0:
            print(f"  head({head}): {flat[:head]}")
    else:
        if head > 0:
            flat = arr.ravel()
            print(f"  head({head}): {flat[:head]}")


def try_print_meta(meta_arr: np.ndarray) -> None:
    """
    meta is stored as an object array with one dict entry.
    """
    try:
        meta_obj = meta_arr.item()
        if isinstance(meta_obj, dict):
            print("\n[meta]")
            for k, v in meta_obj.items():
                print(f"  {k}: {v}")
        else:
            print("\n[meta] present but not a dict:")
            print(meta_obj)
    except Exception as e:
        print("\n[meta] could not decode:")
        print(e)


def view_npz(path: str, head: int = 0) -> None:
    data = np.load(path, allow_pickle=True)
    keys = data.files

    print(f"\nFile: {path}")
    print(f"Keys ({len(keys)}):")
    for k in keys:
        print(f"  - {k}")

    if "meta" in keys:
        try_print_meta(data["meta"])

    for k in keys:
        if k == "meta":
            continue
        summarize_array(k, data[k], head=head)

    print("\nDone.\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect contents of an NPZ file")
    parser.add_argument("path", help="Path to .npz file")
    parser.add_argument("--head", type=int, default=0,
                        help="Print first N entries of flattened arrays")
    args = parser.parse_args()

    view_npz(args.path, head=args.head)


if __name__ == "__main__":
    main()
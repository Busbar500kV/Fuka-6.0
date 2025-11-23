"""
tools/pack_npz.py
-----------------

Utility to package simulation outputs into a canonical NPZ blob.

Why this tool exists:
    - Some experiments produce many arrays and metadata dicts.
    - This script standardizes the naming and ensures every NPZ contains:
          meta.json (as a dict inside the NPZ)
          arrays: V_hist, g_last, fitness_hist, ...
          any additional items supplied by user
    - Makes downstream analysis uniform.

Usage:
    python tools/pack_npz.py \
        --input-json run_outputs.json \
        --out runs/packed_run.npz

Or from Python:

    from tools.pack_npz import pack_npz
    pack_npz("runs/myrun_out.json", "runs/myrun_final.npz")

The JSON can contain paths to npy/npz files or direct metadata fields.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from typing import Dict, Any, Optional


# -------------------------------------------------------------
# Core packing function
# -------------------------------------------------------------

def pack_npz(input_path: str, output_path: str, compress: bool = True) -> None:
    """
    Read a JSON dictionary describing arrays and metadata, then
    save them into a single NPZ file.

    JSON format example:

        {
            "meta": {
                "experiment": "exp_phenotype",
                "timestamp": "2025-02-22_13:05:11",
                "substrate_cfg": {...},
                "plasticity_cfg": {...}
            },

            "arrays": {
                "V_hist":      "runs/tmp/V_hist.npy",
                "fitness_hist":"runs/tmp/F.npy",
                "g_last":      "runs/tmp/g_last.npy"
            }
        }

    The tool loads .npy, .npz, or raw lists into arrays.
    """

    with open(input_path, "r") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    arrays = data.get("arrays", {})

    out = {}

    # attach metadata as a python dict inside the NPZ
    out["meta"] = np.array([meta], dtype=object)

    # load numeric arrays
    for key, path_or_value in arrays.items():
        if isinstance(path_or_value, str):
            # treat as file on disk
            if path_or_value.endswith(".npy"):
                out[key] = np.load(path_or_value)
            elif path_or_value.endswith(".npz"):
                # load each item from NPZ; flatten into this NPZ
                sub = np.load(path_or_value)
                for subkey in sub.files:
                    out[f"{key}_{subkey}"] = sub[subkey]
            else:
                raise ValueError(f"Unsupported file extension: {path_or_value}")
        else:
            # raw python list or nested list
            out[key] = np.array(path_or_value)

    # final write
    if compress:
        np.savez_compressed(output_path, **out)
    else:
        np.savez(output_path, **out)

    print(f"[pack_npz] wrote {output_path} with {len(out)} entries.")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pack arrays + metadata into NPZ")
    parser.add_argument("--input-json", required=True,
                        help="Input JSON describing arrays + metadata")
    parser.add_argument("--out", required=True,
                        help="Output .npz file")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable np.savez_compressed")
    args = parser.parse_args()

    pack_npz(
        input_path=args.input_json,
        output_path=args.out,
        compress=not args.no_compress
    )


if __name__ == "__main__":
    main()
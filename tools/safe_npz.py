# tools/safe_npz.py
from __future__ import annotations
import json
import numpy as np
from typing import Any, Dict

def _is_object_array(x: np.ndarray) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == object

def make_npz_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert payload to NPZ-safe form:
      - keeps numpy numeric arrays as-is
      - converts lists/tuples to numeric arrays if possible
      - converts dicts to JSON strings (plus a _json key)
      - converts object arrays to JSON lists (plus a _json key)
    Result: no pickles; everything loads with allow_pickle=False.
    """

    safe: Dict[str, Any] = {}

    for k, v in payload.items():

        # --- numpy arrays ---
        if isinstance(v, np.ndarray):
            if _is_object_array(v):
                # ragged / object arrays -> JSON
                safe[k + "_json"] = json.dumps([np.asarray(x).tolist() for x in v], separators=(",", ":"))
            else:
                safe[k] = v
            continue

        # --- dicts -> JSON ---
        if isinstance(v, dict):
            safe[k + "_json"] = json.dumps(v, separators=(",", ":"))
            continue

        # --- lists/tuples ---
        if isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    safe[k + "_json"] = json.dumps(arr.tolist(), separators=(",", ":"))
                else:
                    safe[k] = arr
            except Exception:
                safe[k + "_json"] = json.dumps(list(v), separators=(",", ":"))
            continue

        # --- scalars ---
        if np.isscalar(v):
            safe[k] = v
            continue

        # --- fallback: stringify ---
        safe[k + "_json"] = json.dumps(str(v))

    return safe


def savez_safe(path: str, payload: Dict[str, Any]) -> None:
    safe = make_npz_safe(payload)
    np.savez_compressed(path, **safe)
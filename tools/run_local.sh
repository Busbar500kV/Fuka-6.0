#!/usr/bin/env bash
#
# tools/run_local.sh
#
# Unified local runner for all Fuka-6.0 experiments.
#
# Usage:
#     chmod +x tools/run_local.sh
#     ./tools/run_local.sh exp_token_chains
#
# This runs:  python -m experiments.exp_token_chains
#
# Optional flags:
#     ./tools/run_local.sh exp_token_chains --headless
#
# The --headless flag disables matplotlib blocking (for servers).
#

set -e

# -------------------------------
# Determine repo root
# -------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$REPO_ROOT"

# -------------------------------
# Find experiment argument
# -------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [--headless]"
    echo "Example: $0 exp_token_chains"
    exit 1
fi

EXP_NAME="$1"
shift

# -------------------------------
# Optional flags
# -------------------------------
HEADLESS=0
for arg in "$@"; do
    if [ "$arg" == "--headless" ]; then
        HEADLESS=1
    fi
done

# -------------------------------
# Activate venv if found
# -------------------------------
if [ -d "$REPO_ROOT/venv" ]; then
    echo "[run_local] Activating venv: venv/"
    source "$REPO_ROOT/venv/bin/activate"
elif [ -d "$REPO_ROOT/.venv" ]; then
    echo "[run_local] Activating venv: .venv/"
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "[run_local] No virtual environment found — running with system python."
fi

# -------------------------------
# Optional headless mode
# -------------------------------
if [ "$HEADLESS" -eq 1 ]; then
    export MPLBACKEND=Agg
    echo "[run_local] Using headless backend: Agg"
fi

# -------------------------------
# Ensure repo root on PYTHONPATH
# -------------------------------
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
echo "[run_local] PYTHONPATH set."

# -------------------------------
# Run experiment
# -------------------------------
MODULE="experiments.${EXP_NAME}"

echo "[run_local] Running: python -m $MODULE"
python -m "$MODULE"

echo "[run_local] Done."
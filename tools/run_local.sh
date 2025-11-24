#!/bin/bash
# ------------------------------------------------------------
# Fuka-6.0  |  run_local.sh  (enhanced)
#
# Runs any experiment and automatically logs output to:
#   runs/logs/<expname>_<timestamp>.log
#
# Enhancements:
#   - Git branch/commit/dirty status recorded
#   - Python version recorded
#   - Runtime duration recorded
#   - Exit code recorded
#   - Detects latest NPZ saved during run
#   - Supports passing through extra args to experiment
#
# Usage:
#   ./tools/run_local.sh exp_phenotype
#   ./tools/run_local.sh exp_modules
#   ./tools/run_local.sh exp_phenotype --long_run 1
#
# ------------------------------------------------------------

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./tools/run_local.sh <experiment_name> [extra args...]"
    exit 1
fi

EXP_NAME="$1"
shift
EXTRA_ARGS=("$@")

# ------------------------------------------------------------
# Activate venv
# ------------------------------------------------------------
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[run_local] ERROR: venv/ not found. Run: python3 -m venv venv"
    exit 1
fi

echo "[run_local] Activating venv: $VENV_DIR/"
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------
# Set Python path for local execution
# ------------------------------------------------------------
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "[run_local] PYTHONPATH set."

# ------------------------------------------------------------
# Prepare log directory and logfile
# ------------------------------------------------------------
mkdir -p runs/logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="runs/logs/${EXP_NAME}_${TIMESTAMP}.log"

# ------------------------------------------------------------
# Snapshot state BEFORE run
# ------------------------------------------------------------
START_EPOCH=$(date +%s)

# Newest NPZ before run (may be empty)
PRE_NPZ=$(ls -1t runs/*.npz 2>/dev/null | head -n 1 || true)

# Git info (if repo)
GIT_BRANCH=""
GIT_COMMIT=""
GIT_DIRTY="unknown"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "")
    if git diff --quiet && git diff --cached --quiet; then
        GIT_DIRTY="clean"
    else
        GIT_DIRTY="dirty"
    fi
fi

PY_VER=$(python --version 2>&1 || echo "python-unknown")
HOST=$(hostname || echo "host-unknown")
USER_NAME=$(whoami || echo "user-unknown")
PWD_NOW=$(pwd)

echo "[run_local] Logging to: $LOGFILE"
echo "[run_local] Running: python -m experiments.${EXP_NAME} ${EXTRA_ARGS[*]}"

# ------------------------------------------------------------
# Write header to logfile
# ------------------------------------------------------------
{
    echo "=== Fuka-6.0 | Run started: $(date) ==="
    echo "Experiment: ${EXP_NAME}"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        echo "Extra args: ${EXTRA_ARGS[*]}"
    else
        echo "Extra args: (none)"
    fi
    echo "----------------------------------------"
    echo "Host: ${HOST}"
    echo "User: ${USER_NAME}"
    echo "Working dir: ${PWD_NOW}"
    echo "Python: ${PY_VER}"
    if [ -n "$GIT_COMMIT" ]; then
        echo "Git branch: ${GIT_BRANCH}"
        echo "Git commit: ${GIT_COMMIT}"
        echo "Git status: ${GIT_DIRTY}"
    else
        echo "Git: (not a git repo)"
    fi
    echo "Pre-run latest NPZ: ${PRE_NPZ:-none}"
    echo "----------------------------------------"
} | tee "$LOGFILE" >/dev/null

# ------------------------------------------------------------
# Run experiment and tee output
# ------------------------------------------------------------
set +e
python -m experiments."${EXP_NAME}" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))

# ------------------------------------------------------------
# Detect newest NPZ after run
# ------------------------------------------------------------
POST_NPZ=$(ls -1t runs/*.npz 2>/dev/null | head -n 1 || true)

NEW_NPZ="none"
if [ -n "$POST_NPZ" ] && [ "$POST_NPZ" != "$PRE_NPZ" ]; then
    NEW_NPZ="$POST_NPZ"
fi

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
{
    echo "----------------------------------------"
    echo "Exit code: ${EXIT_CODE}"
    echo "Runtime: ${DURATION} seconds"
    echo "Post-run latest NPZ: ${POST_NPZ:-none}"
    echo "New NPZ from this run: ${NEW_NPZ}"
    echo "=== Run finished: $(date) ==="
} | tee -a "$LOGFILE" >/dev/null

echo "[run_local] Done."
echo "[run_local] Output saved in: $LOGFILE"
if [ "$NEW_NPZ" != "none" ]; then
    echo "[run_local] New NPZ saved: $NEW_NPZ"
fi

exit $EXIT_CODE
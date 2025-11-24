#!/bin/bash
# ------------------------------------------------------------
# Fuka-6.0  |  run_local.sh  (enhanced + auto log sync)
#
# Runs any experiment and automatically:
#   1) logs output to runs/logs/<expname>_<timestamp>.log
#   2) copies that log into repo logs/<expname>_<timestamp>.log
#   3) git add/commit/push the log to GitHub (SSH remote)
#
# Usage:
#   ./tools/run_local.sh exp_phenotype
#   ./tools/run_local.sh exp_modules
#   ./tools/run_local.sh exp_phenotype --foo 3
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
IN_GIT_REPO="no"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    IN_GIT_REPO="yes"
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
    if [ "$IN_GIT_REPO" = "yes" ]; then
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

# ------------------------------------------------------------
# Auto-sync log to GitHub
# ------------------------------------------------------------
if [ "$IN_GIT_REPO" = "yes" ]; then
    mkdir -p logs
    REPO_LOG="logs/$(basename "$LOGFILE")"
    cp "$LOGFILE" "$REPO_LOG"

    # Stage log
    git add "$REPO_LOG"

    # Only commit if there is something staged
    if ! git diff --cached --quiet; then
        COMMIT_MSG="Auto-sync log $(basename "$LOGFILE")"
        set +e
        git commit -m "$COMMIT_MSG" >/dev/null 2>&1
        COMMIT_OK=$?
        set -e

        if [ $COMMIT_OK -eq 0 ]; then
            set +e
            git push >/dev/null 2>&1
            PUSH_OK=$?
            set -e

            if [ $PUSH_OK -eq 0 ]; then
                echo "[run_local] Log auto-pushed to GitHub: $REPO_LOG"
            else
                echo "[run_local] WARNING: log committed but push failed. You can push later."
            fi
        else
            echo "[run_local] WARNING: log staged but commit failed. You can commit later."
        fi
    else
        echo "[run_local] No new log changes to commit."
    fi
else
    echo "[run_local] Git repo not detected. Skipping log auto-sync."
fi

exit $EXIT_CODE
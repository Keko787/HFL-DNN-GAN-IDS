#!/usr/bin/env bash
# Experiment 1 — launch the smallest viable topology on one Linux box
#
# Spins up 4 client subprocesses + 1 server in the background, runs the
# trial grid, then shuts everything down. Output CSV lands at
# results/exp1.csv (override with $OUT).
#
# Usage:
#     bash experiments/exp1/setup/launch_local.sh                # full grid
#     N_TRIALS=2 FILTER=Dpd=10MB,R=5 bash .../launch_local.sh    # smoke
#     OUT=results/exp1_jittery.csv bash .../launch_local.sh      # alt path
#
# Pair with shape_link.sh (apply/remove) for the shaping ablation.

set -euo pipefail

OUT="${OUT:-results/exp1.csv}"
N_TRIALS="${N_TRIALS:-20}"
BASE_SEED="${BASE_SEED:-42}"
SERVER_PORT="${SERVER_PORT:-9000}"
N_CLIENTS="${N_CLIENTS:-4}"
FILTER="${FILTER:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

# Resolve the venv python regardless of platform.
if [[ -x "${REPO}/.venv310/Scripts/python.exe" ]]; then
    PY="${REPO}/.venv310/Scripts/python.exe"
elif [[ -x "${REPO}/.venv310/bin/python" ]]; then
    PY="${REPO}/.venv310/bin/python"
else
    PY="python"
fi

mkdir -p "$(dirname "${OUT}")" logs

# Build the per-client flags: --client d1@127.0.0.1:partition=0 etc.
SERVER_FLAGS=()
for ((i=1; i<=N_CLIENTS; i++)); do
    SERVER_FLAGS+=(--client "d${i}@127.0.0.1:partition=$((i-1))")
done
SERVER_FLAGS+=(--bind-port "${SERVER_PORT}")
SERVER_FLAGS+=(--output "${OUT}")
SERVER_FLAGS+=(--n-trials "${N_TRIALS}")
SERVER_FLAGS+=(--base-seed "${BASE_SEED}")
if [[ -n "${FILTER}" ]]; then
    SERVER_FLAGS+=(--filter "${FILTER}")
fi

cleanup() {
    echo "[launch_local] cleaning up ..."
    for pid in "${CLIENT_PIDS[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start the server in the background; redirect stdout+stderr to a log.
echo "[launch_local] starting server on 127.0.0.1:${SERVER_PORT} (output → ${OUT})"
"${PY}" -m experiments.exp1.server "${SERVER_FLAGS[@]}" > logs/exp1_server.log 2>&1 &
SERVER_PID=$!

# Wait briefly for the listener to be ready before clients connect.
sleep 0.5

CLIENT_PIDS=()
for ((i=1; i<=N_CLIENTS; i++)); do
    cid="d${i}"
    partition=$((i-1))
    "${PY}" -m experiments.exp1.client \
        --client-id "${cid}" \
        --server "127.0.0.1:${SERVER_PORT}" \
        --data-partition "${partition}" \
        > "logs/exp1_${cid}.log" 2>&1 &
    CLIENT_PIDS+=($!)
done
echo "[launch_local] launched ${#CLIENT_PIDS[@]} clients (pids: ${CLIENT_PIDS[*]})"
echo "[launch_local] waiting for server to finish..."

# Wait for the server; clients exit on SHUTDOWN broadcast.
wait "${SERVER_PID}"
echo "[launch_local] server exited; CSV at ${OUT}"

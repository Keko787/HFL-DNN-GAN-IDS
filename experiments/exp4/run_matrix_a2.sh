#!/usr/bin/env bash
# Sweep A2 only — the jittery surface (H0 vs H1), 12 cells x 20 seeds.
#
# Split out because A1 (clean) and C (L1) completed cleanly and must NOT be
# re-run, while A2's first attempt was discarded: two concurrent script
# instances appended to the same CSV and produced duplicate (cell, arm, trial)
# rows. See run_matrix.sh's lock comment for the two rules that prevents.
set -uo pipefail

LOCK="results/exp4_matrix/.matrix.lock"
mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING TO START: $LOCK exists — another run is active." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

PY="${PY:-/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe}"
OUT="results/exp4_matrix"

echo "[$(date +%H:%M:%S)] SWEEP A2 — jittery surface (H0 vs H1), 12 cells x 20 seeds"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/A_jittery.csv" --trace-dir "$OUT/A_jittery_traces" \
    --arms H0 H1 \
    --regime jittery \
    --dead-zone 0.0 0.2 0.4 0.6 \
    --link-quality 0.3 0.5 0.7 \
    --N 6 --rrf 60 --n-missions 4 \
    --n-trials 20 --base-seed 42 \
    --real-model --data-source canonical \
    --realism \
    --mission-budget-s 120 \
    --keep-event-traces \
    --trial-budget-s 300 > "$OUT/A_jittery.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] sweep A2 finished rc=$rc"
[ -f "$OUT/A_jittery.csv" ] && echo "  rows: $(($(wc -l < "$OUT/A_jittery.csv") - 1))"
echo "A2 COMPLETE"

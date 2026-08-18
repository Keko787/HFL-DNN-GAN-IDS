#!/usr/bin/env bash
# Extend the 60 s (tight-budget) point from 20 to 40 paired seeds.
#
# Why: at 60 s our arm produced better models than both baselines — accuracy
# +0.074 (p=0.0329) vs D1 and +0.089 (p=0.0342) vs D2 — but neither survives
# correction for the 20 tests run across the budget axis. Doubling the seeds is
# what settled the L1 result: if the effect is real the CI tightens around the
# same mean and p falls; if it was noise it regresses. Either answer is useful.
#
# ONE runner, not sharded. The shards elsewhere write DIFFERENT csv files; this
# resumes a single file, and two processes appending to one csv is exactly what
# produced 344 duplicated rows in sweep A2. Sequential is the correct choice
# here even though it costs wall-clock.
#
# Resumes b60.csv: identical configuration and base seed, trials 0..19 already
# present, so the runner skips them and adds 20..39.
set -uo pipefail

LOCK="results/exp4_sota/.sota.lock"
mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING TO START: $LOCK exists — another run is active." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

PY="${PY:-/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe}"
OUT="results/exp4_sota"

echo "[$(date +%H:%M:%S)] B60 extend — 20 -> 40 seeds (adds trials 20..39)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/b60.csv" --trace-dir "$OUT/b60_traces" \
    --arms H1 D1 D2 \
    --N 6 --rrf 60 --n-missions 4 \
    --regime jittery \
    --n-trials 40 --base-seed 42 \
    --real-model --data-source canonical \
    --realism \
    --tau 0.85 \
    --mission-budget-s 60 \
    --keep-event-traces \
    --trial-budget-s 300 > "$OUT/b60_extend.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] finished rc=$rc"
[ -f "$OUT/b60.csv" ] && echo "  rows: $(($(wc -l < "$OUT/b60.csv") - 1))  (expect 120)"
echo "B60 EXTEND COMPLETE"

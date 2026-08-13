#!/usr/bin/env bash
# S3c pilot — checklist §5.0.
#
# Two configurations differing by EXACTLY ONE FLAG, so the comparison is a clean
# one-flag delta. Same base seed, same grid, same everything else — that is the
# whole reason S3c was built as a toggle.
#
#   off : deadline enforcement ON, window adaptation OFF   (the control)
#   on  : deadline enforcement ON, window adaptation ON
#
# Enforcement is on in BOTH arms deliberately. With no budget the deadline never
# binds, so a wider window rescues nothing and the comparison is a tie by
# construction — that would test the harness, not the mechanism.
#
# n_missions=8 because the adapter's default history window is 5 and the first
# mission has none: at the headline n_missions=4 S3c can barely act, and a null
# there would be uninterpretable rather than informative.
#
# Stub mode (no --real-model): the S3c question is about PARTICIPATION —
# who gets served — which the stub produces. Convergence is downstream.
set -uo pipefail

PY="${PY:-/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe}"
OUT="${OUT:-results/exp4_s3c}"
SEEDS="${SEEDS:-20}"
BUDGET="${BUDGET:-120}"
NMISS="${NMISS:-8}"

COMMON=(
  --arms H1
  --N 6
  --rrf 50
  --n-missions "$NMISS"
  --n-trials "$SEEDS"
  --base-seed 42
  --realism
  --h1-field-radius-m 150
  --mission-budget-s "$BUDGET"
  --keep-event-traces
  --trial-budget-s 180
)

mkdir -p "$OUT"

run_arm() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] starting arm '$name'"
  "$PY" -m experiments.exp4.runner_main \
      --csv "$OUT/$name.csv" \
      --trace-dir "$OUT/${name}_traces" \
      "${COMMON[@]}" "$@" \
      > "$OUT/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] arm '$name' finished rc=$?"
}

run_arm off &
PID_OFF=$!
sleep 5           # stagger so the two process trees do not bind ports at once
run_arm on --mission-window-adaptation &
PID_ON=$!

wait $PID_OFF $PID_ON
echo "[$(date +%H:%M:%S)] pilot complete"
wc -l "$OUT"/off.csv "$OUT"/on.csv 2>/dev/null || true

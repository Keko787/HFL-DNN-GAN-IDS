#!/usr/bin/env bash
# L1 confirmation — does H3 > H2 survive scrutiny?
#
# Sweep C gave final_auc +0.042 at p=0.018 with n=20 at ONE cell. That passes the
# claim rule but FAILS Bonferroni for five metrics (alpha=0.01), and this project
# has already retracted an L1 effect once — so the bar here is higher, not lower.
#
# Two independent ways of being wrong, so two shards:
#
#   C1  MORE SEEDS at the same operating point (20 -> 40). If the effect is real
#       the CI tightens around the same mean; if it was noise it regresses.
#       Resumes C_h2h3.csv, adding trial_index 20..39 on the same base_seed.
#
#   C2  A SECOND OPERATING POINT (n_missions 4 -> 6). Independent evidence, and
#       it tests a directional prediction: L1 picks a channel per mission, so
#       more missions should mean a LARGER cumulative effect, not an equal one.
#
# NOTE on the axis choice: dead_zone / link_quality are NOT used as the second
# point. Freeze D5 — they are consumed only in the H0 branch, so they vary
# nothing for H2/H3. n_missions genuinely changes what the L1 controller does.
#
# The shards write DIFFERENT files, so they run concurrently for real. Sweep A
# ran one runner at a time and therefore had no concurrency at all, which is why
# it took 2.4x its costed time (see cost_matrix.py).
set -uo pipefail

LOCK="results/exp4_matrix/.l1confirm.lock"
mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING TO START: $LOCK exists — another run is active." >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

PY="${PY:-/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe}"
OUT="results/exp4_matrix"

COMMON=(
  --arms H2 H3
  --N 6 --rrf 60
  --regime jittery
  --l1-channel
  --base-seed 42
  --real-model --data-source canonical
  --realism
  --mission-budget-s 120
  --keep-event-traces
  --trial-budget-s 300
)

echo "[$(date +%H:%M:%S)] C1 — extend to 40 seeds at n_missions=4 (resumes, adds 20..39)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/C_h2h3.csv" --trace-dir "$OUT/C_traces" \
    --n-missions 4 --n-trials 40 \
    "${COMMON[@]}" > "$OUT/C1.log" 2>&1 &
PID1=$!

sleep 20   # stagger so the two process trees do not bind ports simultaneously

echo "[$(date +%H:%M:%S)] C2 — second operating point, n_missions=6, 20 seeds"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/C2_nm6.csv" --trace-dir "$OUT/C2_traces" \
    --n-missions 6 --n-trials 20 \
    "${COMMON[@]}" > "$OUT/C2.log" 2>&1 &
PID2=$!

wait $PID1; echo "[$(date +%H:%M:%S)] C1 finished rc=$?"
wait $PID2; echo "[$(date +%H:%M:%S)] C2 finished rc=$?"

for f in "$OUT"/C_h2h3.csv "$OUT"/C2_nm6.csv; do
  [ -f "$f" ] && echo "  $(basename "$f"): $(($(wc -l < "$f") - 1)) rows"
done
echo "L1 CONFIRM COMPLETE"

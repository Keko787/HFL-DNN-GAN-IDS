#!/usr/bin/env bash
# Whole-scheduler SOTA comparison over a TWO-POINT BUDGET AXIS.
#
# H1 (ours) vs D1 (MAX-AoI) vs D2 (Oort statistical utility), each owning its own
# admission decision, at budget 120 s and 60 s.
#
# Why two budgets rather than one. The pilot showed the arms differ at 120 s but
# that admission binds only weakly there — the served sets diverge on 22% of
# missions, concentrated in mission 1. A single loose point risks reporting a
# trade-off that is really an artefact of a constraint that barely binds. The
# axis makes "how tight is the budget" the variable rather than a hidden
# assumption.
#
# tau=0.85, not the 0.82 default: 0.82 was set from the whole 640-trial matrix,
# which pooled H0 and the full degradation surface. At THIS operating point
# accuracy runs higher and 0.82 saturates (0.90/0.80/0.90); 0.85 discriminates
# (0.20/0.20/0.50). Reach-rate is recomputable from traces at any tau anyway.
#
# The two shards write DIFFERENT csv files, so they are genuinely parallel. One
# runner executes its grid sequentially — that is what made sweep A take 2.4x its
# estimate.
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
SEEDS="${SEEDS:-20}"

COMMON=(
  --arms H1 D1 D2
  --N 6 --rrf 60 --n-missions 4
  --regime jittery
  --n-trials "$SEEDS" --base-seed 42
  --real-model --data-source canonical
  --realism
  --tau 0.85
  --keep-event-traces
  --trial-budget-s 300
)

# B120 RESUMES the pilot: identical configuration and identical seeds (base 42,
# trials 0..9 already present), so the runner skips them and adds 10..19. That is
# what the resumable csv is for, and it saves 30 trials of compute. Verified by
# reconciling the final row count against the designed grid, as always.
echo "[$(date +%H:%M:%S)] B120 — budget 120 s (resumes the 10 pilot seeds, adds 10)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/pilot.csv" --trace-dir "$OUT/pilot_traces" \
    --mission-budget-s 120 \
    "${COMMON[@]}" > "$OUT/b120.log" 2>&1 &
PID1=$!

sleep 20   # stagger so the two process trees do not bind ports simultaneously

echo "[$(date +%H:%M:%S)] B60 — budget 60 s (fresh)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/b60.csv" --trace-dir "$OUT/b60_traces" \
    --mission-budget-s 60 \
    "${COMMON[@]}" > "$OUT/b60.log" 2>&1 &
PID2=$!

wait $PID1; echo "[$(date +%H:%M:%S)] B120 finished rc=$?"
wait $PID2; echo "[$(date +%H:%M:%S)] B60 finished rc=$?"

for f in "$OUT"/pilot.csv "$OUT"/b60.csv; do
  [ -f "$f" ] && echo "  $(basename "$f"): $(($(wc -l < "$f") - 1)) rows (expect 60)"
done
echo "SOTA AXIS COMPLETE"

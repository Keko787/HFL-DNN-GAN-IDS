#!/usr/bin/env bash
# Phase-3 matrix — checklist §5.1.
#
# Sweeps A and C only. **Sweep B is deliberately NOT here**: at the designed
# operating point H1/B1/B2 produce identical results, because S3b fixes WHO is
# served before the policy ever runs and the policy only reorders an
# already-decided set. See §5.1b — it needs re-targeting at the binding band
# before it can produce a result rather than a tie.
#
# Concurrency 3: at 5 the box exhausted memory and ~30 % of trials failed.
set -uo pipefail

PY="${PY:-/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe}"
OUT="${OUT:-results/exp4_matrix}"
SEEDS="${SEEDS:-20}"
BUDGET="${BUDGET:-120}"

mkdir -p "$OUT"

COMMON=(
  --N 6 --rrf 60 --n-missions 4
  --n-trials "$SEEDS" --base-seed 42
  --real-model --data-source canonical
  --realism
  --mission-budget-s "$BUDGET"
  --keep-event-traces
  --trial-budget-s 300
)

# Sweep A is TWO invocations, not one cross product.
#
# `dead_zone` and `link_quality` describe how the CLEAN regime is degraded into
# the jittery one — under `clean` they are inert. Passing all three axes to one
# grid yields 2 x 4 x 3 = 24 cells, of which the 12 clean ones are the SAME
# configuration repeated: ~440 wasted trials, and 12 cells in the analysis that
# look distinct but are not. (Same class of error as Freeze D5 — sweeping an
# axis that varies nothing for the condition under test.)
#
# So: clean at ONE point, jittery across the surface. 1 + 12 = the 13 cells the
# committed design used and the cost model assumes.

echo "[$(date +%H:%M:%S)] SWEEP A1 — clean baseline (H0 vs H1), 1 cell x $SEEDS seeds"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/A_clean.csv" --trace-dir "$OUT/A_clean_traces" \
    --arms H0 H1 \
    --regime clean \
    --dead-zone 0.0 --link-quality 1.0 \
    "${COMMON[@]}" > "$OUT/A_clean.log" 2>&1
echo "[$(date +%H:%M:%S)] sweep A1 finished rc=$?"

echo "[$(date +%H:%M:%S)] SWEEP A2 — jittery surface (H0 vs H1), 12 cells x $SEEDS seeds"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/A_jittery.csv" --trace-dir "$OUT/A_jittery_traces" \
    --arms H0 H1 \
    --regime jittery \
    --dead-zone 0.0 0.2 0.4 0.6 \
    --link-quality 0.3 0.5 0.7 \
    "${COMMON[@]}" > "$OUT/A_jittery.log" 2>&1
echo "[$(date +%H:%M:%S)] sweep A2 finished rc=$?"

echo "[$(date +%H:%M:%S)] SWEEP C — L1 adaptivity (H2 vs H3), 1 cell x $SEEDS seeds"
"$PY" -m experiments.exp4.runner_main \
    --csv "$OUT/C_h2h3.csv" --trace-dir "$OUT/C_traces" \
    --arms H2 H3 \
    --regime jittery \
    --l1-channel \
    "${COMMON[@]}" > "$OUT/C.log" 2>&1
echo "[$(date +%H:%M:%S)] sweep C finished rc=$?"

echo "[$(date +%H:%M:%S)] MATRIX COMPLETE"
for f in "$OUT"/A_clean.csv "$OUT"/A_jittery.csv "$OUT"/C_h2h3.csv; do
  [ -f "$f" ] && echo "  $(basename "$f"): $(($(wc -l < "$f") - 1)) rows"
done

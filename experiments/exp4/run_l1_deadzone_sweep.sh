#!/usr/bin/env bash
# EX-4.3 — H2 (fixed band) vs H3 (adaptive L1 band) across the dead-zone axis.
#
# Clean re-run of the earlier h2h3_m_* sweep, which was contaminated: 22/190
# status=ok rows had produced NO model evaluation at all (rounds_evaluated=0,
# blank AUC but hard-zero participation), so they were dropped from the AUC
# means yet averaged as real 0.0 into the participation means — flipping the
# sign of several H3-H2 differences. driver.py now records those as
# status=no_eval so `status == "ok"` excludes them from EVERY metric.
#
# Same grid as before (N=6, rrf=60, n_missions=4, 20 seeds, dz 0.0-0.6 jittery
# + a clean reference at dz=0.0), with a larger trial budget + startup timeout
# to cut the timeout attrition (the old run lost 2/shard, unevenly across arms:
# H3 6 timeouts vs H2 2 — differential attrition biases a paired comparison).
#
# 5 concurrent shards, per-trial TF threads capped so they share the cores.
set -uo pipefail
cd /d/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS

export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

PY="/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe"
D=results/exp4_paper
SEEDS=20
COMMON="--arms H2 H3 --N 6 --rrf 60 --n-missions 4 --n-trials $SEEDS \
        --real-model --data-source canonical --realism --l1-channel \
        --local-epochs 2 --link-quality 0.5 \
        --trial-budget-s 420 --startup-timeout-s 120"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$PY" -m experiments.exp4.runner_main "$@" > "$D/shard_l1dz_$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)"
}

run clean --csv $D/h2h3_dz_clean.csv $COMMON --regime clean   --dead-zone 0.0 &
run dz00  --csv $D/h2h3_dz_dz00.csv  $COMMON --regime jittery --dead-zone 0.0 &
run dz02  --csv $D/h2h3_dz_dz02.csv  $COMMON --regime jittery --dead-zone 0.2 &
run dz04  --csv $D/h2h3_dz_dz04.csv  $COMMON --regime jittery --dead-zone 0.4 &
run dz06  --csv $D/h2h3_dz_dz06.csv  $COMMON --regime jittery --dead-zone 0.6 &

wait
echo "======================================================================"
echo "[$(date +%H:%M:%S)] L1 DEAD-ZONE SWEEP COMPLETE"
echo "======================================================================"

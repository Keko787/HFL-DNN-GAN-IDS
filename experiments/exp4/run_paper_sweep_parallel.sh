#!/usr/bin/env bash
# EX-4 paper-grade paired sweep, PARALLEL shards (20-core box, ~7am deadline).
#
# Same grid + 20 seeds as the serial run — no validity trimmed — but split
# into 6 concurrent shards, each into its own resumable CSV, with per-trial
# TF threads capped so 6 trees share the cores instead of oversubscribing.
#   * jittery surface: one shard per dead_zone (0.0 reuses the partial CSV)
#   * clean reference
#   * H2-vs-H3 L1 (n_missions=6)
# Merge h0h1_*.csv for the surface analysis; h2h3_l1.csv stands alone.
set -uo pipefail
cd /d/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS

export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

PY="/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe"
D=results/exp4_paper
SEEDS=20
COMMON="--N 6 --rrf 60 --real-model --data-source canonical --realism --local-epochs 2 --trial-budget-s 300 --startup-timeout-s 90 --n-trials $SEEDS"

run_shard() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$PY" -m experiments.exp4.runner_main "$@" > "$D/shard_$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)"
}

# --- jittery participation surface: 4 dead-zone shards -----------------------
run_shard dz00 --csv $D/h0h1_surface.csv --arms H0 H1 $COMMON --n-missions 4 \
    --regime jittery --dead-zone 0.0 --link-quality 0.3 0.5 0.7 &
run_shard dz02 --csv $D/h0h1_dz02.csv    --arms H0 H1 $COMMON --n-missions 4 \
    --regime jittery --dead-zone 0.2 --link-quality 0.3 0.5 0.7 &
run_shard dz04 --csv $D/h0h1_dz04.csv    --arms H0 H1 $COMMON --n-missions 4 \
    --regime jittery --dead-zone 0.4 --link-quality 0.3 0.5 0.7 &
run_shard dz06 --csv $D/h0h1_dz06.csv    --arms H0 H1 $COMMON --n-missions 4 \
    --regime jittery --dead-zone 0.6 --link-quality 0.3 0.5 0.7 &
# --- clean reference ---------------------------------------------------------
run_shard clean --csv $D/h0h1_clean.csv  --arms H0 H1 $COMMON --n-missions 4 \
    --regime clean --dead-zone 0.0 --link-quality 0.5 &
# --- H2-vs-H3 L1 (adaptive channel), n_missions=6 ----------------------------
run_shard l1 --csv $D/h2h3_l1.csv --arms H2 H3 --l1-channel $COMMON --n-missions 6 \
    --regime jittery clean --dead-zone 0.0 --link-quality 0.5 &

wait
echo "======================================================================"
echo "[$(date +%H:%M:%S)] ALL SHARDS COMPLETE"
echo "======================================================================"

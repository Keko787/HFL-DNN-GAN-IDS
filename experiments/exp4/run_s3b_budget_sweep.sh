#!/usr/bin/env bash
# EX-4 — where does the S3b deadline/feasibility gate bind, and what does it cost?
#
# The offline probe (python -m experiments.exp4.probe_s3b_binding) already
# answers the mechanism question exactly, on real device layouts and the real
# scheduler, in under a second:
#
#   * DEADLINE floor  ~34% of contacts are dropped at ANY budget - they cannot
#                     be reached before their own deadline at 5 m/s over a
#                     100 m field. Independent of mission_budget_s.
#   * BUDGET knee     ~60 s. Below it, budget drops appear and grow:
#                     46% dropped @50s, 58% @40s, 72% @30s, 93% @5s.
#
# This sweep measures the CONSEQUENCE of that on the experiment's own metrics -
# how much participation and round-closure the enforcement actually costs -
# using the ladder the probe identified:
#
#   (control)  gate OFF          - reproduces the committed configuration
#   120 s      slack budget      - deadline enforcement only
#    60 s      at the knee       - budget just starts to bite
#    30 s      tight             - budget dominates
#    15 s      very tight        - most of the queue is refused
#
# Arm H1 only: this is an L2 scheduling question, and H1 is the deterministic
# mule arm, so nothing is confounded by the (untrained) selector. Jittery regime,
# N=6, n_missions=4, 20 paired seeds - matched to the paper run so the control
# shard is directly comparable to results/exp4_paper/.
#
# Read the result as: at which budget do update_yield / mission_completion_rate /
# round_close_rate depart from the control? That is where enforcement starts
# costing real participation.
set -uo pipefail
cd /d/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS

export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

PY="/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe"
D=results/exp4_s3b
mkdir -p "$D"
SEEDS=20

# Concurrency capped for the same reason as the L1 sweep: each trial spawns a
# cluster + mule + N device processes, all loading TensorFlow, and at 5+ shards
# the box ran out of memory and trials failed to bootstrap.
MAX_PAR="${MAX_PAR:-3}"
STAGGER_S="${STAGGER_S:-25}"

COMMON="--arms H1 --N 6 --rrf 60 --n-missions 4 --n-trials $SEEDS \
        --real-model --data-source canonical --realism \
        --regime jittery --dead-zone 0.4 --link-quality 0.5 \
        --local-epochs 2 --trial-budget-s 420 --startup-timeout-s 120"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$PY" -m experiments.exp4.runner_main "$@" > "$D/shard_$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)"
}

launch() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
  run "$@" &
  sleep "$STAGGER_S"
}

# Control: no --mission-budget-s at all -> gate is a strict no-op.
launch control --csv $D/s3b_control.csv $COMMON
launch b120    --csv $D/s3b_b120.csv    $COMMON --mission-budget-s 120
launch b060    --csv $D/s3b_b060.csv    $COMMON --mission-budget-s 60
launch b030    --csv $D/s3b_b030.csv    $COMMON --mission-budget-s 30
launch b015    --csv $D/s3b_b015.csv    $COMMON --mission-budget-s 15

wait
echo "======================================================================"
echo "[$(date +%H:%M:%S)] S3B BUDGET SWEEP COMPLETE"
echo "======================================================================"
echo "Analyse with:"
echo "  python -m experiments.exp4.analyze_s3b_sweep"

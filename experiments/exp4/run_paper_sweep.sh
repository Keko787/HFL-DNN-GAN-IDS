#!/usr/bin/env bash
# EX-4 paper-grade paired sweep (>=20 seeds), resumable.
#
# Phase 1: H0-vs-H1 jittery participation SURFACE (dead_zone x link_quality)
# Phase 2: H0-vs-H1 CLEAN reference (single cell; dead_zone/link N/A under clean)
# Phase 3: H2-vs-H3 L1 adaptive-channel effect (jittery + clean, --l1-channel)
#
# Each runner_main call writes a resumable per-trial CSV: re-running skips
# rows already present, so an interrupted sweep continues where it left off.
# Canonical CICIOT, N=6, real model. Two output CSVs (two paired comparisons).
set -euo pipefail

cd /d/networkIntrusionDetectionSystem/FL-DNN-GAN-IDS
export TF_CPP_MIN_LOG_LEVEL=3
PY="/c/Users/kskos/AppData/Local/Programs/Python/Python311/python.exe"
H0H1="results/exp4_paper/h0h1_surface.csv"
H2H3="results/exp4_paper/h2h3_l1.csv"
SEEDS=20

banner() { echo "======================================================================"; echo "PHASE: $1  [$(date +%H:%M:%S)]"; echo "======================================================================"; }

banner "1/3  H0-vs-H1 jittery surface (4 dead_zone x 3 link x ${SEEDS} seeds)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$H0H1" --arms H0 H1 --N 6 --rrf 60 --n-missions 4 --n-trials "$SEEDS" \
    --regime jittery --dead-zone 0.0 0.2 0.4 0.6 --link-quality 0.3 0.5 0.7 \
    --real-model --data-source canonical --realism --local-epochs 2 \
    --trial-budget-s 240 --startup-timeout-s 60

banner "2/3  H0-vs-H1 clean reference (1 cell x ${SEEDS} seeds)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$H0H1" --arms H0 H1 --N 6 --rrf 60 --n-missions 4 --n-trials "$SEEDS" \
    --regime clean --dead-zone 0.0 --link-quality 0.5 \
    --real-model --data-source canonical --realism --local-epochs 2 \
    --trial-budget-s 240 --startup-timeout-s 60

banner "3/3  H2-vs-H3 L1 (jittery + clean, --l1-channel, n_missions=6, ${SEEDS} seeds)"
"$PY" -m experiments.exp4.runner_main \
    --csv "$H2H3" --arms H2 H3 --N 6 --rrf 60 --n-missions 6 --n-trials "$SEEDS" \
    --regime jittery clean --dead-zone 0.0 --link-quality 0.5 \
    --real-model --data-source canonical --realism --l1-channel --local-epochs 2 \
    --trial-budget-s 300 --startup-timeout-s 60

echo "======================================================================"
echo "ALL PHASES COMPLETE  [$(date +%H:%M:%S)]"
echo "======================================================================"

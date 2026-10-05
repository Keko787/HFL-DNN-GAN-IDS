#!/usr/bin/env sh
# Exp 5: every stage, its levers and its reports, through scripts/exp5/launch.py.
#
#   scripts/exp5/exp5.sh check                  is this machine ready; what each stage needs
#   scripts/exp5/exp5.sh run pilots             the pilots, stopping at each report
#   scripts/exp5/exp5.sh report knee --apply    write a finished pilot's outputs to params.toml
#   scripts/exp5/exp5.sh run batch1 --yes       a stage (or a group: pilots, rl, batches, all)
#   scripts/exp5/exp5.sh score batch1           the studies' paired comparisons
#   scripts/exp5/exp5.sh status                 every stage's progress
#   scripts/exp5/exp5.sh --help                 every command, stage and lever
#
# The interpreter: EXP5_PYTHON if set (e.g. the conda environment's python),
# else python3 on PATH.
HERE=$(cd "$(dirname "$0")" && pwd)
exec "${EXP5_PYTHON:-python3}" "$HERE/launch.py" "$@"

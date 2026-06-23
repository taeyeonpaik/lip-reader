#!/usr/bin/env bash
set -euo pipefail
cd /teamspace/studios/this_studio/lipreader
export PYTHONPATH="$PWD/outputs/lipreading-studio"
python work/lightning/restore_data.py > work/lightning/data-restore.log 2>&1 &
data_pid=$!
python work/lightning/restore_weights.py > work/lightning/weight-restore.log 2>&1 &
weight_pid=$!
wait "$data_pid"
wait "$weight_pid"
cd outputs/lipreading-studio
python -u -m training.lrs2_cuda --verify-only > ../../work/lightning/verify.log 2>&1
python -u -m training.lrs2_cuda --check-only > ../../work/lightning/cpu-readiness.log 2>&1
python -m pip freeze > ../../work/lightning/cloud-environment.txt
printf '%s\n' 'Setup verified. CUDA smoke check and GPU approval still required. Training has not started.'

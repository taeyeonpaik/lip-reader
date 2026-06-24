#!/usr/bin/env bash
# Run only after the user approves a GPU and the bounded test's cost.
# Includes model load and worst-length probe within the 120-second process cap.
set -u
cd /teamspace/studios/this_studio/lipreader || exit 1
export PYTHONPATH="$PWD/outputs/lipreading-studio"
finish() {
  lightning studio switch --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project --machine CPU >> work/lightning/batched-control.log 2>&1 || lightning studio stop --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project >> work/lightning/batched-control.log 2>&1
}
trap finish EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ' > work/lightning/batched-started.txt
timeout --signal=TERM --kill-after=5s 120s python -u work/lightning/batched_test.py > work/lightning/batched-test.log 2>&1
result=$?
printf '%s\n' "$result" > work/lightning/batched-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > work/lightning/batched-finished.txt
exit "$result"

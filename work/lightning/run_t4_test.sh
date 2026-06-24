#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio/lipreader
export PYTHONPATH="$PWD/outputs/lipreading-studio"
finish() {
  printf '%s\n' 'Returning Studio to CPU after bounded test.' >> work/lightning/t4-control.log
  lightning studio switch --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project --machine CPU >> work/lightning/t4-control.log 2>&1 || lightning studio stop --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project >> work/lightning/t4-control.log 2>&1
}
trap finish EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ' > work/lightning/t4-started.txt
timeout --signal=TERM --kill-after=15s 900s python -u work/lightning/t4_test.py > work/lightning/t4-test.log 2>&1
result=$?
printf '%s\n' "$result" > work/lightning/t4-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > work/lightning/t4-finished.txt
exit "$result"

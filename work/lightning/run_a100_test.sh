#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio/lipreader
export PYTHONPATH="$PWD/outputs/lipreading-studio"
finish() {
  printf '%s\n' 'Returning Studio to CPU after bounded test.' >> work/lightning/a100-control.log
  lightning studio switch --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project --machine CPU >> work/lightning/a100-control.log 2>&1 || lightning studio stop --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project >> work/lightning/a100-control.log 2>&1
}
trap finish EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ' > work/lightning/a100-started.txt
timeout --signal=TERM --kill-after=5s 120s python -u work/lightning/a100_test.py > work/lightning/a100-test.log 2>&1
result=$?
printf '%s\n' "$result" > work/lightning/a100-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > work/lightning/a100-finished.txt
exit "$result"

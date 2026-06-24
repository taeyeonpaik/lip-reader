#!/usr/bin/env bash
set -u
export PATH="/home/zeus/miniconda3/envs/cloudspace/bin:$PATH"
cd /teamspace/studios/this_studio/lipreader/outputs/lipreading-studio || exit 1
/home/zeus/miniconda3/envs/cloudspace/bin/python -u -m training.lrs2_batched &
trainer_pid=$!
printf '%s\n' "$trainer_pid" > ../../work/lightning/batched-training.pid
forward_stop() { kill -TERM "$trainer_pid" 2>/dev/null || true; }
trap forward_stop TERM INT
wait "$trainer_pid"
result=$?
# If wait was interrupted, let the trainer finish its atomic checkpoint.
while kill -0 "$trainer_pid" 2>/dev/null; do
  wait "$trainer_pid"
  result=$?
done
printf '%s\n' "$result" > ../../work/lightning/batched-training-exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > ../../work/lightning/batched-training-finished.txt
# Release the user's GPU when training finishes or fails.
/home/zeus/miniconda3/envs/cloudspace/bin/lightning studio stop --name auto-avsr-model-training-devbox --teamspace tpaik-6qr3e-org/pytorch-auto-avsr-lip-reading-project >> ../../work/lightning/batched-training-control.log 2>&1
exit "$result"

# Execution speed review

The original 4-thread CPU decoder configuration was retained. Forward/backward-only variants showed small warm improvements, but complete optimizer-update tests gave median 9.87s/update for 4 threads and 10.27s/update for 1 thread. GPU decoder placement was not adopted on the strength of its small forward/backward gain. No substantial training throughput gain is established.

All benchmark samples came from training, all benchmark gradients/updates were discarded, and the update-100 checkpoint was migrated with model tensors, optimizer tensors and progress independently compared equal. An unchanged backup remains before-speedup.pt. Model, data, precision, effective batch, learning rate, loss, augmentation and beam remain unchanged.

Restart verification rehashes every source file and reuses full decoding/pixel checks only for the same verified source bytes and manifest. All 48,139 files passed the resumed hash check. This saves restart overhead, not training compute.

Managed training resumed in session41405. No cloud compute was started or money spent. The user's decision about another GPU and any budget is pending.

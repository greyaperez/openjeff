# GPU pilot runbook

This is the overall execution sequence. The user has explicitly authorized
spending within $200. The first H100 probe has completed; consult
`reports/spending.json` for the current rental cleanup state.
The local `nvidia-smi` call failed to communicate with the driver. See
[the prepared $10 probe](probe-ready.md) for the upload package and exact commands.

1. Choose a U.S.-region provider instance and record the actual all-in quote,
   GPU SKU, RAM, storage, billing increment and termination controls. Charge setup
   and idle time to the allocation. Do not start above the config's spending stop.
2. Configure and verify billing controls and an explicit termination watchdog.
   Runpod's former `--terminate-after` and `--stop-after` flags were removed for
   silently failing; do not use them. See `probe-ready.md` for primary evidence
   and the small-prepaid-balance fallback. Include residual storage charges.
   The repository's budget command does not enforce any of this.
3. Resolve a tested container/package set from the exact model and trainer recipe.
   Capture container digest, CUDA/driver, PyTorch, Transformers, PEFT/Unsloth commits.
   This scaffold intentionally does not present an untested GPU lockfile as verified.
4. Download approved weights/tokenizer at the revisions in `provenance/models.json`.
   Record SHA-256 for each file and capture actual license/NOTICE files; metadata
   pins are not weight-file checksums. Check disk capacity before downloads.
5. Stage the artifacts, disable telemetry and outbound serving traffic, and enable
   local-only model loading. Keep private data off the pilot; use synthetic/public
   permitted examples. Run the CPU tests before charging a long GPU session.
6. Under a $10 compatibility cap, run the 200 diagnostic decisions, completion code IDs,
   all-candidate logits, no silent truncation and exact configuration signatures.
   Compare BF16 reference scores before accepting a quantized/exported path.
7. Run E0 and E3 separately on the same fixed inputs, recording latency, failures,
   peak memory and every runtime setting. For djev use runtime commit
   `3ce907e6835212f27ee82b4cee9039198c4abe35` and its documented B200 environment
   first. A newest-upstream vLLM build is a separate compatibility experiment.
8. Fit temperature on calibration-fit and audit it on calibration-check. Freeze
   it with model/prompt/readout/runtime identity. It must not migrate silently to
   another quantization, adapter, canvas or IR-conditioned path.
9. For E1, run 100 training steps first, estimate full wall-clock cost from measured
   tokens/sec, and reduce scope if it exceeds the allocation. Train on candidate
   discrimination; do not accidentally optimize all prompt tokens.
10. Freeze the winner, evaluate untouched groups, write the model card and results,
    export permitted artifacts, terminate compute and storage, verify provider
    termination and reconcile invoices. Keep the reserve inside the $200 ceiling.

Budget example (no cloud action):

```bash
python -m openjeff budget --spent 0 --committed 0 --hourly-rate 3.49 --hours 14 --overhead 10
```

Refresh the provider quote and account balance before each subsequent rental.
The current user instruction permits the bounded pilot; the budget preflight
is not a spending enforcement system.

# Reproduce the diffusion contribution experiment

This experiment uses the frozen OpenJeff 0.1.0 adapter. It does not retrain either
foundation, transfer hidden vectors between models, or introduce a learned neural
bridge. Diffusion produces bounded evidence-reference guidance; the AR model sees
that guidance together with the entire original input. The separate probability
mixture combines independently scored candidate distributions.

The design and selection rules are in
[the frozen protocol](diffusion-hybrid-protocol-v2.md). Setup failures, runtime
adjustments, and the actual cash ledger are retained under `reports/`.

## Runtime and execution

Use Linux/Python 3.12, a CUDA 13-compatible NVIDIA driver, sufficient GPU memory
for the BF16 26B diffusion checkpoint, and approximately 160 GB of temporary disk.
The experiment configuration uses an H200 141 GB and a 50 GB persistent result
disk at `/workspace`. Its container is pinned in
`reports/runpod-image-config.json`; dependencies and vLLM sources are pinned in
`configs/diffusion-gpu-lock.txt` and `research/vendor/djev/runtime/sources.json`.

Foundation weights are downloaded from their pinned Google Hugging Face revisions
and are not included in the bundle. Inference occurs locally on the rented GPU;
evaluation examples are not sent to a hosted model API. Only original synthetic
data, redistributable public benchmark inputs, source, and the original adapter
are included in the transfer bundle. No payment credentials are included.

Build and stage the bundle:

```bash
python -m scripts.build_hybrid_bundle
python -m scripts.make_pod_start_command
```

The second command prints the supervisor startup command for that exact bundle.
Use it as the pod's startup command when using the automatic rental workflow,
and transfer the bundle with SSH/SCP within its five-minute upload window. The
experiment encountered long browser-upload stalls; direct SSH transfer succeeded.
Use an existing authorized SSH key and the pod's displayed direct TCP endpoint.

For manual execution on an already managed machine, extract the bundle under
`/workspace`, giving `/workspace/openjeff-diffusion`. Then run:

```bash
cd /workspace/openjeff-diffusion
export HF_HOME=/tmp/openjeff-model-cache
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
bash scripts/run_hybrid_v2.sh
```

The driver installs a separate environment under `/tmp/openjeff-venv`, verifies
and patches the pinned vLLM sources, and runs the vendored runtime tests. The model
server binds `127.0.0.1:19001`; the typed Djev API binds `127.0.0.1:8000`. Port 19001
avoids the stock Runpod template's nginx proxy on 8001. Readiness verifies the model
JSON response instead of accepting an arbitrary HTTP 200 page.

The diffusion pass scores 2,631 inputs and saves its evidence guidance. Its GPU
process group is stopped before the AR pass loads the unchanged adapter. AR runs
four conditions: original input, evidence wrapper alone, diffusion guidance, and
its own guidance. The driver refuses to overwrite an existing scored arm.

This sequential GPU schedule reduces rental needs. It is not a test of both
models serving concurrently on one GPU. Reported pipeline times add warmed stage
measurements; they exclude model loading, GPU swaps, queues, and concurrency.
If the selected blend weight is exactly zero or one, its derived serving-time
estimate includes only the required backend. Both were scored during the study.

## Validate and analyze

After copying `runs/hybrid-v2` back into the repository:

```bash
python -m scripts.analyze_hybrid_v2
python -m scripts.validate_hybrid_v2
```

The completed study is summarized in [the results report](../reports/HYBRID-RESULTS.md).
With Matplotlib installed, `python -m scripts.plot_hybrid_v2` regenerates the static
PNG/PDF figure. After validation, `python -m scripts.build_hybrid_release` creates
the separate study archive and SHA-256 sidecar without overwriting the original
0.1.0 release. Every included file is listed in `STUDY-MANIFEST.json`.

Analysis chooses the probability-mixture weight using development NLL only, then
fits each scorer's temperature using calibration-fit only. The validator checks
all inputs, targets, candidate dimensions, recorded guidance, mixture selection,
temperature fits, and the original adapter's hash. It does not rerun inference.

The public benchmark is a 231-item diagnostic, not the full official leaderboard.
Synthetic partitions contain documented repeated normalized states. The 403-case
final subset without a literal normalized training-state match is supplementary;
it does not establish new-domain or semantic novelty.

## Rental shutdown and exports

The shell driver's timeouts stop computation. They do not stop billing by
themselves. The separately tested `scripts/pod_supervisor_v2.py` starts the driver,
accepts only a hash-matching public bundle, stops if upload is absent after five
minutes, and calls Runpod's own-pod stop operation at completion or the three-hour
deadline. Results are exported to `/workspace/openjeff-hybrid-results.tar.gz` with
a SHA-256 sidecar. The ordinary pod volume survives stopping, but is erased when
the pod is terminated; download and verify it first. The actual experiment also
uses a local SSH monitor to retrieve the export before shutdown.

Disabling automatic top-ups and maintaining the cash ledger remain part of this
experiment's cost controls. The supervisor cannot guarantee cleanup if the host
or provider control plane fails. Inspect actual provider state after completion.

# Run OpenJeff locally

The pilot is a Gemma 4 12B BF16 base plus a small LoRA adapter, a frozen temperature,
and a typed decision endpoint. The release manifest identifies the backend chosen
on the development split. No external inference API is required. The adapter does
not contain the 24 GB foundation weights; stage those explicitly once.

## Runtime

Use Linux, Python 3.12, an NVIDIA A100 80 GB, and the tested PyTorch CUDA 12.8 image:
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. The current calibrator binds the
GPU model, PyTorch, Transformers, PEFT, prompt, and readout. Another environment
must produce its own calibration artifact; it is rejected instead of silently
inheriting a probability-quality claim. The H100 compatibility experiment is
separate from the A100 trained pilot. An 80 GB card is the tested training target;
minimum serving VRAM and other GPU configurations have not been established.

In that image, from the repository root:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r configs/train-gpu-requirements.txt
.venv/bin/python -m scripts.stage_weights
.venv/bin/python -m openjeff.serve --run runs/pilot-v1 --port 8765
```

The system-site-packages option is deliberate in this specific tested container:
it reuses its working CUDA PyTorch installation. Do not apply it to an arbitrary
machine with conflicting Torch/Transformers packages.

The server listens on `127.0.0.1` only and provides `GET /health` and
`POST /v1/decide`. It performs no network downloads at startup, has no tool runner,
and does not log input state. Stop it with Ctrl+C. The serial local endpoint is a
prototype, not an authenticated multi-user production gateway.

```bash
curl http://127.0.0.1:8765/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":{"allowed":["read"],"denied":["export"],"action":"export"},"question":"Allow only explicitly allowed actions. Explicit deny overrides allow. What applies?","candidates":[{"id":"allow","description":"Allow the action"},{"id":"deny","description":"Deny the action"}]}'
```

Responses contain every candidate's probability, the suggested candidate, the
accepted decision or `null`, an explicit `abstained` flag, and model/calibration
identifiers. The default confidence threshold is 0.9. This is a configurable
abstention rule, not a guarantee of 90% correctness on your workflow. Calibration
was fitted on synthetic rule tasks; distribution shift can invalidate confidence.

The service accepts 2–26 distinct candidates and at most 1,536 prompt tokens. It
rejects oversized requests instead of truncating evidence. The separate public
JevBench diagnostic permits 8,192 tokens and measures longer inputs; that extension
is not silently included in the calibrated service contract.

## Reproduce the pilot

```bash
python -m scripts.build_curriculum
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u -m scripts.train_pilot --download
.venv/bin/python -m scripts.verify_pilot_service
.venv/bin/python -m scripts.jevbench_public
```

The output directory must be absent before starting a new run. Move an existing
run to an archive rather than overwriting it. Default training: 500 optimizer
steps, microbatch 4, accumulation 4, 8,000 examples, LoRA rank 16/alpha 32/dropout
0.05 on text attention projections, AdamW, peak learning rate 1e-4, warmup and
cosine decay, candidate-only cross entropy, BF16 base, float32 adapter parameters.
No teacher API or benchmark item supplies training targets. The generated data
and code are released under Apache-2.0.

Development checkpoints at steps 125, 250, and 500 compete by accuracy, then NLL.
The best adapter competes with the unmodified base by that same rule. Only after
selection is frozen does the run fit temperatures on 600 calibration-fit examples
and evaluate the separate check, final, adversarial, and production-simulation
sets. Candidate order is randomized in training and reversed in a final diagnostic.
Related reversals are not counted as independent samples in the reported paired
bootstrap interval.

The numerical limits stop computation only. They do not stop cloud billing.
Export and verify the output before stopping or terminating an ephemeral pod.

The original generator uses unique case IDs but repeats some logical states. A
post-hoc normalized-state audit found 197/600 final cases matching training states.
Read `reports/curriculum-overlap.json` alongside the headline synthetic metrics;
these splits are not semantic or domain holdouts. Do not regenerate or tune the
released checkpoint after inspecting its benchmark outcomes.

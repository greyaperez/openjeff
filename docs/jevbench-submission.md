# JevBench evaluation handoff

Evaluation requested in [JevBench issue #54](https://github.com/fstandhartinger/jevbench/issues/54).
OpenJeff has not received an official JevBench rank. Request evaluation of the
frozen AR-only `openjeff-pilot-v1`, not the optional diffusion experiment.

## Evaluator-controlled runtime

Use the original A100 SXM 80 GB runtime in [the usage guide](using-openjeff.md):
Linux/Python 3.12, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, BF16 base,
SDPA, pinned dependency versions. The original image tag is recorded; its digest
was not captured. The scorer rejects a runtime that differs from the frozen
calibration identity. Contact the project if different hardware is needed; do
not disable the check or fit temperatures on benchmark decisions.

```bash
git clone https://github.com/greyaperez/openjeff.git
cd openjeff
# Check out the immutable commit specified in the submission issue.
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r configs/train-gpu-requirements.txt
.venv/bin/python -m scripts.stage_weights
.venv/bin/python -m openjeff.benchmark_serve --run runs/pilot-v1 --port 8766
```

The adapter is also distributed at
https://huggingface.co/greyecho/openjeff-pilot-v1 . The GitHub checkout includes
the identical adapter and the complete original run directory.

The benchmark endpoint listens on **127.0.0.1 only**. Run the evaluator's harness
on the same machine or use an evaluator-controlled SSH tunnel. It writes no
request bodies, evidence, predictions, or keys to disk. No hosted service or
always-on rental is required from the submitter.

Use the upstream `typesafe` adapter with endpoint `http://127.0.0.1:8766`, model
`openjeff-pilot-v1`, and an empty key-env. POST `/v1/systemone` accepts one question
named `decision`, with `type`, `instructions`, and `criteria`. Supported types:
`noul`, `choice`, and `score`. Responses contain the full exact-label probability
distribution; noul also returns numeric `noul`, choice returns `choice`, and score
returns the expected numeric level. No confidence threshold suppresses answers.

The evaluator must determine the GPU rental cost basis; unknown cost is not zero.
Report networking, cold-start treatment, concurrency, and hardware with timings.

## Mapping and calibration disclosure

The endpoint accepts 2–26 options, a 192 KB HTTP body, a 128 KB JSON state, a
32,000-character rubric, and at most 8,192 compiled tokens. Oversized evidence is
rejected, never truncated. It uses the original prompt and candidate-token
readout. The synthetic A100 temperature is transferred unchanged to this longer
context diagnostic, as in the original public evaluation; it is not a new claim
of long-context calibration.

**Choice order follows the received criteria object's insertion order.** The
upstream TypeSafe wire format does not transmit the task's separate `labels`
order, which the original direct public evaluator used. Consequently the new
endpoint's scores must be measured afresh; 90.04% is historical direct-evaluator
accuracy, not a measured score for this HTTP integration. Noul order is no/yes;
score order is the criteria list's order. Tests verify evidence and per-label
meaning against all 231 public tasks, plus wire response shapes. The new endpoint
has not yet been GPU-benchmarked; saved-weight reload and HTTP parity evidence
in the pilot applies to the original `/v1/decide` service.

## Evidence and request

- Initial public subset: 208/231 correct (90.04%); hard 89/111 (80.18%).
- Training excludes JevBench items and teacher API outputs. Foundation pretraining
  contamination is unknown. Synthetic state overlap is documented.
- Public results do not imply an official composite score or rank.
- Request evaluator-controlled execution and the current v1.4 sealed evaluation.
- Do not return or publish sealed item text, answer keys, or per-item outputs.
- [Model card](../model_cards/openjeff-pilot.md) and [results](../reports/RESULTS.md).

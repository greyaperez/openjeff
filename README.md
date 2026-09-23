# OpenJeff

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/openjeff-logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/branding/openjeff-logo.png">
    <img src="assets/branding/openjeff-logo.png" alt="OpenJeff name-badge mascot" width="240">
  </picture>
</p>

**Small answers. Serious judgment.**

A self-hosted, experimental judgment primitive: evidence and a finite set of
choices go in; probabilities, a proposed decision, and an explicit abstention
come out. Named, deliberately, for “My name is Jeff.”

## Initial benchmarks: OpenJeff vs. Jev

**90.04% accuracy on 231 public JevBench items — 3.46 percentage points above
Jev 1.13.0's published results on the same items.**

| Public JevBench subset | OpenJeff 0.1.0 | Jev 1.13.0¹ | Difference |
|---|---:|---:|---:|
| **All public · 231 items** | **90.04%** | 86.58% | **+3.46 pp** |
| Hard · 111 items | **80.18%** | 72.97% | **+7.21 pp** |
| Original · 72 items | 98.61% | 98.61% | Tie |
| Easy · 48 items | 100.00% | 100.00% | Tie |

¹ Jev's results are recomputed from [the benchmark publisher's recorded outcomes](https://github.com/fstandhartinger/jevbench/blob/f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84/results/v1.2/jevbench-v1.2-per-task.json)
on the same public task IDs. OpenJeff is our initial A100 pilot; Jev was **not
rerun here**, and prompts, hardware, and runtimes differ. This is a descriptive
accuracy comparison on the **231-item public subset**, not the full 534-item
benchmark, an official JevBench composite score, or a claim of overall superiority.

[Results, methodology, and raw evidence](reports/RESULTS.md#public-jevbench-transfer-test)
· [Reproduce the pilot](docs/using-openjeff.md)
· [Diffusion follow-up](reports/HYBRID-RESULTS.md)

OpenJeff also improved public accuracy over its unadapted Gemma 4 12B foundation
from **70.13% to 90.04%** in the same pilot. Median local request time was
**72.3 ms** on an A100 80 GB, excluding networking, cold loading, and multi-user
contention; it is not a speed comparison with Jev's hosted API.

<details>
<summary>More pilot results and limitations</summary>

| Measured result | Base Gemma 4 12B | OpenJeff |
|---|---:|---:|
| Synthetic final set, 600 examples | 79.67% | **97.50%** |
| Final-set candidate-order disagreement (lower is better) | 22.00% | **1.83%** |

A normalized-state audit found 197/600 synthetic final cases matching training
states; OpenJeff scored 96.28% on the remaining 403. Synthetic calibration does
not establish probability quality in an unfamiliar workflow. The public-hard
calibration error remains 0.142 (10-bin ECE). No Jev comparison was run on this
synthetic set.

The later H200 diffusion study is a separate experiment: AR-only scored 90.91%
on the public subset and diffusion-guided AR scored 89.61%. The initial pilot
above is preserved rather than mixing measurements across environments.

</details>

**Version 0.1.0 pilot is trained and evaluated.** It pairs Google's Gemma 4 12B IT
with an original LoRA adapter and a frozen temperature calibrator. Original code,
adaptation data, and adapter are Apache-2.0; upstream artifacts retain their terms.
No external inference API is required.

## Start here

- [Completed diffusion study: mixed gains, AR remains the default](reports/HYBRID-RESULTS.md)
- [Reproduce and validate the diffusion comparison](docs/reproduce-hybrid.md)
- [Results, benchmark comparison, limitations, and cost](reports/RESULTS.md)
- [Run the trained model and reproduce the pilot](docs/using-openjeff.md)
- [Model card](model_cards/openjeff-pilot.md)
- [Validated release evidence](reports/release-validation.json)
- [Architecture and experimental program](docs/research-plan.md)
- [Architecture decision](docs/adr-001.md)
- [Cash and resource ledger](reports/spending.json)

The release includes the trained adapter in `runs/pilot-v1/adapter`, both frozen
calibrators, raw evaluation scores, the training recipe, the original 10,400-example
curriculum, provenance, and a loopback-only HTTP prototype. Stage the approximately
24 GB pinned foundation weights separately. The tested environment is Linux,
Python 3.12, and an A100 80 GB; the current calibrator checks the exact scorer and
runtime identity. Other environments need fresh calibration.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r configs/train-gpu-requirements.txt
.venv/bin/python -m scripts.stage_weights
.venv/bin/python -m openjeff.serve --run runs/pilot-v1 --port 8765
```

Use those commands inside the tested container described in the usage guide.
The endpoint accepts 2–26 candidates, scores all of them, and rejects requests
above 1,536 prompt tokens. Its default confidence threshold is 0.9. It executes
no tools or business actions. It is a research prototype, not a production gateway.

## Inspect without a GPU

Clone this repository, then use Python 3.10+ for the numerical core and exported-result checks:

```bash
git clone https://github.com/greyaperez/openjeff.git
cd openjeff
python -m unittest discover -s tests -v
python -m scripts.validate_release
python -m scripts.summarize_training
python -m scripts.validate_hybrid_v2
```

Tests requiring optional model libraries skip when those libraries are absent.
The release validator recomputes metrics from saved scores; it does not rerun model
inference. GPU save/reload and HTTP probability parity were also tested separately.

## Architecture and release scope

The trained AR-only model remains the default. The completed diffusion study
tested diffusion-only, evidence guidance into the frozen AR model, an evidence-map
control, AR self-guidance, and probability fusion on 2,631 inputs per condition.
Diffusion guidance raised synthetic final accuracy from 97.50% to 98.00%, but
lowered public accuracy from 90.91% to 89.61% while improving calibrated public
NLL from 0.4434 to 0.3181. Warmed public median stage time increased from 67.1 ms
to 130.5 ms. The development-selected probability blend assigned diffusion zero
weight. [Full results and limits](reports/HYBRID-RESULTS.md). Its
[frozen protocol](docs/diffusion-hybrid-protocol-v2.md) and
[execution record](reports/diffusion-hybrid-v2.json) distinguish completed
measurements from setup attempts. The original unsuccessful diffusion attempt
is retained in [its experiment record](reports/diffusion-attempt.json).
A hybrid needs evidence that refinement improves quality enough to justify its
cost; no learned bridge or production concurrency claim is included.

The foundation's Google developer and base-weight lineage meet the requested
vendor constraint. This does not establish exclusively U.S.-sourced pretraining
data or reproduce the foundation's undisclosed training corpus. The original
curriculum contains no teacher API outputs or benchmark training examples.

The initial research used **$60 in prepaid GPU credit**, with approximately
**$35.84 consumed**, including unsuccessful setup attempts. The
[cost record](reports/spending.json) is transparent about those failures. All
experimental rentals were removed after the results were backed up.

## Support OpenJeff

OpenJeff is free and open source. If it is useful to you, optional support helps
cover GPU time for training and reproducible evaluations. The code, adapters,
and results remain available to everyone. Useful bug reports, independent tests,
and documentation improvements help just as much.

[Support OpenJeff on GitHub Sponsors](https://github.com/sponsors/greyaperez)
or see [how to contribute](CONTRIBUTING.md).

This project is independent of TypeSafe, Google, and the film's owners. Source, adapter, data, and evaluation records are published here. No hosted
inference service is provided.

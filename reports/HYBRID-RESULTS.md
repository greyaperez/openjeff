# OpenJeff: diffusion contribution study

Completed 2026-09-23. **Keep AR alone as the default.** The diffusion evidence-guidance layer ran successfully and showed a probability-quality benefit, but it did not deliver a consistent accuracy improvement sufficient to justify making two models the default.

On the synthetic final set it added three correct answers (97.50% → 98.00%). An evidence-map wrapper alone added two, and AR self-guidance added four. On the public diagnostic it lost three correct answers (90.91% → 89.61%), although its calibrated NLL, Brier score, and calibration error improved. Its warmed public median stage time was about 1.95× AR alone. The frozen development-only mixture selection chose **zero diffusion weight**.

This is a completed inference-only diffusion → structured guidance → AR experiment. Neither model was retrained. No hidden-state transfer or learned latent bridge was tested. Those remain separate hypotheses; this result does not establish that all diffusion hybrids fail.

## Accuracy

All AR conditions below were rerun on the same H200 with the unchanged original adapter. Percent correct:

| Method | Final 600 | Final without normalized training match, 403 | Public 231 | Public hard 111 |
|---|---:|---:|---:|---:|
| AR alone | 97.50% | 96.28% | 90.91% | 81.98% |
| AR + evidence map | 97.83% | 97.02% | 88.74% | 80.18% |
| Diffusion alone | 76.33% | 75.93% | 83.55% | 66.67% |
| Diffusion guidance → AR | 98.00% | 97.52% | 89.61% | 80.18% |
| AR guidance → AR | 98.17% | 97.52% | 88.74% | 80.18% |
| Selected probability blend | 97.50% | 96.28% | 90.91% | 81.98% |

The probability blend reduces to AR alone. Its independent temperature refit differs only at numerical rounding scale. The H200 AR public score is 210/231, versus 208/231 in the earlier A100 pilot; that change is a runtime rerun difference, not a new training gain. The historical pilot and its calibrators remain unchanged.

| Method | Adversarial 200 | Production simulation 200 |
|---|---:|---:|
| AR alone | 97.00% | 98.00% |
| AR + evidence map | 97.00% | 98.00% |
| Diffusion alone | 76.50% | 73.50% |
| Diffusion guidance → AR | 96.50% | 97.50% |
| AR guidance → AR | 97.50% | 96.50% |
| Selected probability blend | 97.00% | 98.00% |

## What changed on paired examples?

| Comparison | Set | Repaired mistakes | Broke correct answers | Net correct |
|---|---|---:|---:|---:|
| ar → diffusion_guided_ar | final | 7 | 4 | +3 |
| ar_view → diffusion_guided_ar | final | 4 | 3 | +1 |
| ar_guided_ar → diffusion_guided_ar | final | 2 | 3 | -1 |
| ar → diffusion_guided_ar | public | 3 | 6 | -3 |
| ar_view → diffusion_guided_ar | public | 5 | 3 | +2 |
| ar_guided_ar → diffusion_guided_ar | public | 2 | 0 | +2 |

The synthetic gain versus AR alone was +0.50 percentage points, with a descriptive family-cluster bootstrap interval of −1.17 to +2.67 points. Public change was −1.30 points, interval −2.94 to +0.47. These intervals include zero and concern the observed families, not new-domain guarantees. Family resampling used 2,000 draws with seed 23; there are only 20 synthetic and 18 public families. Full paired comparisons, error correlations, and oracle unions are in the machine-readable report.

## Probability quality

Lower is better for all metrics below. Each changed scorer has a temperature fitted only on the 600 calibration-fit examples. Public labels did not select temperatures, mixture weights, or guidance settings.

| Method | Final calibrated NLL | Public raw NLL | Public calibrated NLL | Public Brier sum | Public ECE, 10 bins |
|---|---:|---:|---:|---:|---:|
| AR alone | 0.0522 | 0.4795 | 0.4434 | 0.1655 | 0.0634 |
| AR + evidence map | 0.0410 | 0.5098 | 0.3531 | 0.1707 | 0.0405 |
| Diffusion alone | 0.6111 | 0.5616 | 0.4485 | 0.2253 | 0.0433 |
| Diffusion guidance → AR | 0.0478 | 0.4812 | 0.3181 | 0.1515 | 0.0335 |
| AR guidance → AR | 0.0400 | 0.5016 | 0.3515 | 0.1703 | 0.0462 |
| Selected probability blend | 0.0522 | 0.4795 | 0.4434 | 0.1655 | 0.0634 |

Diffusion guidance has a useful **calibrated probability-quality signal**: public NLL improves from 0.4434 to 0.3181, and Brier sum from 0.1655 to 0.1515, despite lower accuracy. It also beats the evidence-map and AR self-guidance controls on those public metrics. Raw public NLL is essentially unchanged versus AR (0.4795 → 0.4812), so the calibrated benefit must not be presented as an uncalibrated reasoning improvement. Public hard ECE falls from 0.1204 to 0.0615. No claim of calibrated confidence on arbitrary new workflows follows.

Only 10 public tasks provide soft gold distributions. Their calibrated probability MAE is 0.2066 for AR, 0.1807 for diffusion guidance, and 0.1466 for the map-only control. This very small slice does not establish a diffusion advantage over all controls.

## Timing and execution

| Method | Public median | Public p95 | Synthetic final median |
|---|---:|---:|---:|
| AR alone | 67.1 ms | 275.0 ms | 65.7 ms |
| AR + evidence map | 67.6 ms | 346.6 ms | 65.7 ms |
| Diffusion alone | 46.6 ms | 163.4 ms | 45.3 ms |
| Diffusion guidance → AR | 130.5 ms | 558.5 ms | 122.8 ms |
| AR guidance → AR | 460.4 ms | 3254.0 ms | 223.7 ms |
| Selected probability blend | 67.1 ms | 275.0 ms | 65.7 ms |

These are warmed observed stage times, summed per case for the pipelines. Diffusion and AR were loaded in separate phases on one H200. Times exclude downloads, cold startup, GPU swaps, queues, and concurrent service. They are not end-to-end deployment latency or a serving SLA. The zero-weight blend omits diffusion in its derived serving estimate, although both models were evaluated in this study. AR self-guidance batches its binary relevance questions in groups of eight; this is a concrete control, not an optimized equal-GPU-time search.

The diffusion model was pinned DiffusionGemma 26B-A4B, BF16, one structured denoising read, seed 0 and one sample, through the pinned Djev/vLLM runtime. The frozen Gemma 4 12B OpenJeff adapter used BF16 base weights and SDPA. Full revisions, source hashes, package versions, and settings are in `runs/hybrid-v2/*-runtime.json`, the lock file, and [reproduction instructions](../docs/reproduce-hybrid.md).

Guidance selects up to three evidence references from up to eight deterministic fields/spans, plus missing/conflicting-evidence flags. The AR refiner retains the full original input. Stage-one final answers and free-form rationales are excluded from this primary guidance object. This is a coarse evidence-selection interface, not a learned semantic relation parser.

## Validation and limits

- All 2,631 inputs completed for diffusion and each of four AR conditions. The derived blend brings the saved-score total to 15,786; every row passed source/hash/target/dimension checks.
- Six temperature fits, development-only mixture selection, and recorded guidance were independently checked. A local/cloud floating-point difference in one temperature changes probabilities by at most 2.76e-7; the cloud calibrator is preserved and the validator records tight numerical tolerances.
- 39 local tests and 183 vendored runtime tests passed. These checks do not prove benchmark cleanliness or model correctness.
- The final synthetic set has 197 normalized training-state matches. The remaining 403 still share generators and rule families; calibration-fit states also recur across evaluation. This is not a semantic or new-domain holdout.
- The 231 public JevBench items were already viewed during the AR pilot. They are a secondary transfer diagnostic, not a new blinded test or the full 534-item official ranking. Foundation pretraining contamination is unknown.
- No production concurrency, injection-resilience guarantee, learned routing gate, diffusion fine-tuning, or latent bridge was established.

## Cost and resource cleanup

**$60 total prepaid purchases** against the $200 cumulative cash cap; approximately **$35.84 consumed** at the final check. Billing details lag, so consumption is provisional. Do not add prepaid purchases and credit usage together. Automatic top-ups are disabled, the completed pod and its ordinary volume were terminated, and the console showed **$0.000/hour**.

This includes the earlier failed rentals: a browser upload/terminal call stalled about 4.16 hours and the old detached watchdog failed, wasting approximately $28 without model scores. That was an execution failure, not a useful training expense. A replacement foreground supervisor was tested against real provider stop operations, SSH replaced browser uploads, and a template port collision was corrected before the successful H200 run. The successful result archive and the complete 416 MB volume backup were downloaded, hashed, and checked before cleanup. [Cash ledger](spending.json) and [execution record](diffusion-hybrid-v2.json).

## Artifacts and decision

Keep the original AR service as the default. Retain diffusion guidance as an experimental backend in the reproducible evaluation harness: its public probability metrics justify a future, genuinely unseen probability-focused study, but these measurements do not justify enabling it universally or spending the remaining budget on a latent bridge now.

- [Complete metrics and paired comparisons](hybrid-v2-results.json)
- [Saved-score validation](hybrid-v2-validation.json)
- [Result figure](figures/hybrid-v2-results.png) and [PDF](figures/hybrid-v2-results.pdf)
- [Frozen design](../docs/diffusion-hybrid-protocol-v2.md) and [reproduction instructions](../docs/reproduce-hybrid.md)
- New local study archive: `artifacts/OpenJeff-diffusion-study-v2.tar.gz`, with SHA-256 sidecar and per-file `STUDY-MANIFEST.json`.

The original `OpenJeff-0.1.0-pilot.tar.gz` remains unchanged. Foundation weights, credentials, private attachments, and billing receipts are excluded from the public study package. Source, adapter, and reproducible study records are now published in this repository; no model-hub publication is claimed.

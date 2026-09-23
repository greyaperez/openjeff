# Diffusion contribution experiment v2

Frozen design, 2026-09-23, before obtaining any diffusion model scores.
The user authorized resuming the diffusion experiment within the original $200
cash ceiling. The completed AR adapter and original 0.1.0 archive remain fixed.

## Question and comparisons

Does a diffusion first pass improve this trained AR decision model enough to
justify its extra inference? Compare:

1. Diffusion-only: pinned Djev/DiffusionGemma, one denoising read.
2. AR-only: frozen OpenJeff 12B adapter, rerun on the hybrid evaluation hardware.
3. Diffusion evidence guidance -> AR: original evidence plus bounded, validated
   evidence-reference guidance from diffusion. No stage-one final answer in this
   primary intermediate representation.
4. AR evidence guidance -> AR: same guidance questions and interface, using the
   AR model for the first pass. This tests whether an extra pass alone explains gains.
5. Probability fusion: a development-selected mixture of independent AR and
   diffusion scores. This is a late-fusion control, not a latent neural bridge.

Guidance uses up to eight deterministic spans/fields from the original state.
Each first-pass question asks whether its referenced evidence is decisive for the
original question. References are application-generated and checked. Select up to
three fields above probability 0.5, plus separately assessed missing/conflicting
information flags. Both stages retain the full original evidence. The guidance
is untrusted; no unrestricted generated rationale or invented evidence is passed.

## Selection, calibration, and limits

Use the same 400 development, 600 calibration-fit, 400 calibration-check, 600 final,
200 adversarial, and 200 production-simulation examples as the AR pilot. The full
synthetic task distribution has documented repeated normalized states; report
the 403 final cases without a normalized training-state match as a diagnostic.
Original model weights never change. Fit each changed scorer's temperature on
calibration-fit only. Choose a fusion weight from 0, 0.25, 0.5, 0.75, 1 using
minimum development NLL, before calibration/final evaluation. Also report raw
accuracy, NLL, Brier, ECE, paired gains/damage, and serial stage latency.

The 231 public JevBench questions are a secondary transfer diagnostic. AR results
have already been viewed, so this is not a fresh blinded benchmark. Never use its
labels to select weights, guidance prompts, thresholds, or checkpoints. It is not
the full 534-item official ranking. No superiority claim from accuracy alone.

A useful result needs improved held-out accuracy or probability quality against
AR-only, and must also be compared with AR self-guidance. A negative result is
valid: report damage and inference cost rather than retaining a harmful layer.
A learned latent bridge or retraining is a later experiment, not something this
inference-only comparison can establish.

## Execution and cost

Start with a standard Runpod PyTorch image, since the previous custom startup
failed before Jupyter became ready. Check startup promptly and stop if it fails.
Run shutdown control after initialization using only the existing pod-scoped key;
never print/export credentials. Use the provider's official own-pod GraphQL stop
mutation (same primitive as the previously successful A100 CLI stop), not the
REST endpoint that rejected that key. A watchdog is a mitigation, not a guarantee
if the host crashes. Manual cleanup and disabled automatic top-ups remain required.

Budget this iteration at at most $30 additional prepaid cash initially, with no
simultaneous rentals and no network volume. Absolute project cash ceiling stays
$200. Extend compute only with verified balance and recorded ledger entries.
Export and verify all artifacts before stopping ephemeral storage.

Implementation detail before scoring: also report an AR-only view control using
exactly the same original-evidence wrapper and evidence-reference map, with no
intermediate guidance. This separates formatting effects from first-stage effects.
For strings, references address up to eight contiguous character spans with short
prefix excerpts; for objects, they address up to eight top-level fields. The
complete original state is retained. Guidance does not claim a validated semantic
parser of every policy relationship. AR guidance evaluates its binary relevance
questions in batches of eight; primary decision scoring remains serial.

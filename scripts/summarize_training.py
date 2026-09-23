"""Write the pilot report from exported, validated artifacts."""
import json
from pathlib import Path
from openjeff.calibration import probabilities
from openjeff.evaluation import evaluate


def main():
    root=Path('runs/pilot-v1');s=json.loads((root/'summary.json').read_text());b=json.loads((root/'jevbench-public/summary.json').read_text())
    ref=json.loads(Path('reports/published-public-comparison.json').read_text());ledger=json.loads(Path('reports/spending.json').read_text())
    release=json.loads((root/'release.json').read_text());runtime=s['runtime']
    overlap=json.loads(Path('reports/curriculum-overlap.json').read_text())['final_breakdown']
    m=s['models'];pct=lambda x:f'{100*x:.2f}%'
    text=['# OpenJeff 0.1.0 — measured pilot results','',
      'OpenJeff now has an original synthetic-data LoRA adapter, a frozen calibrator, and a checked local typed-decision endpoint. '
      'The development-selected backend is **'+s['release_choice']+'**, from step **'+str(s['best_step'])+'**. '
      'This is an experimental release. A synthetic-task score does not establish broad judgment quality or production readiness.','',
      '## What ran','',
      '- Foundation: Google Gemma 4 12B IT, revision `'+runtime['revision']+'`; BF16, no quantization.',
      '- Training: 8,000 original rule-grounded examples, 20 families, one epoch, 500 optimizer steps. No teacher API, private data, or JevBench item was used for training.',
      '- LoRA: rank 16, alpha 32, dropout 0.05; text attention projections only. Candidate-only cross entropy. Microbatch 4 × accumulation 4; peak learning rate 1e-4.',
      '- Hardware: '+runtime['gpu']+'; PyTorch '+runtime['torch']+', Transformers '+runtime['transformers']+', PEFT '+runtime['peft']+'.',
      f'- Full training/evaluation script wall time: {s["wall_seconds"]/60:.1f} minutes, including staging. Peak allocated memory: {s["peak_memory_bytes"]/2**30:.2f} GiB.',
      '- Checkpoints at steps 125, 250, and 500 competed only on development accuracy, then NLL. Selection was frozen before calibration and final model scoring.',
      '- The saved adapter reloaded successfully, matched recorded probabilities through the loopback HTTP endpoint, and rejected an invalid request. See `runs/pilot-v1/service-verification.json`.','',
      '## Synthetic held-out results','',
      '| Split | Decisions | Base accuracy | OpenJeff accuracy | Base calibrated ECE | OpenJeff calibrated ECE |',
      '|---|---:|---:|---:|---:|---:|']
    for split in ('calibration_check','final','adversarial','production_sim'):
        x,y=m['base']['splits'][split]['calibrated'],m['adapter']['splits'][split]['calibrated']
        text.append(f'| {split} | {x["n"]} | {pct(x["accuracy"])} | {pct(y["accuracy"])} | {x["ece"]:.4f} | {y["ece"]:.4f} |')
    iv=s['paired_final']['interval'];change=s['paired_final']['refinement']['net_accuracy_change']
    text += ['',f'Final-set accuracy gain: **{100*change:.2f} percentage points**, paired row-bootstrap 95% interval **[{100*iv["lower"]:.2f}, {100*iv["upper"]:.2f}]**. '
      'The interval concerns this synthetic generator distribution, not unseen domains. Reversed options are excluded from the independent-row bootstrap.',
      '',f'Candidate-order disagreement on the final scenarios: base **{pct(m["base"]["final_order_disagreement"])}**, adapter **{pct(m["adapter"]["final_order_disagreement"])}**. '
      'These are paired original/reversed reads of the same scenarios.',
      '',f'Temperatures fitted on the separate 600-example calibration-fit set: base **{m["base"]["temperature"]:.4f}**, adapter **{m["adapter"]["temperature"]:.4f}**. '
      'All synthetic ECE values use 15 equal-width bins; NLL, Brier, reliability, and risk/coverage data are in `summary.json`.','',
      'Weakest adapter families in the final set:','', '| Family | Base accuracy | OpenJeff accuracy |','|---|---:|---:|']
    fam=m['adapter']['splits']['final']['families']
    for name in sorted(fam,key=lambda k:fam[k]['accuracy'])[:5]:
        text.append(f'| {name} | {pct(m["base"]["splits"]["final"]["families"][name]["accuracy"])} | {pct(fam[name]["accuracy"])} |')
    text += ['', '## Normalized-state overlap audit','', 'A post-hoc check removes decorative case IDs and normalizes their copies in entity names. **197/600 final cases match a training state** under this rule. It ignores question wording and candidate order; it does not prove semantic novelty for the remaining cases. Finite boolean families account for many repeats. Exact request hashes include unique IDs and therefore are a weaker separation check.', '', '| Final subset | Cases | Base accuracy | OpenJeff accuracy |', '|---|---:|---:|---:|', f'| Matches a normalized training state | {overlap["base"]["matches_training_state"]["rows"]} | {pct(overlap["base"]["matches_training_state"]["accuracy"])} | {pct(overlap["adapter"]["matches_training_state"]["accuracy"])} |', f'| No match under that normalization | {overlap["base"]["does_not_match_training_state"]["rows"]} | {pct(overlap["base"]["does_not_match_training_state"]["accuracy"])} | {pct(overlap["adapter"]["does_not_match_training_state"]["accuracy"])} |', '', 'This is a diagnostic slice, not a replacement benchmark or new model-selection criterion. Weights and temperatures remain frozen. Calibration-fit states also recur in later synthetic splits; see [the full audit](curriculum-overlap.json). Synthetic confidence metrics should not be read as independent-domain calibration evidence.', '', '## Public JevBench transfer test','',
      'This diagnostic uses the pinned **231 public items** (48 easy, 72 original, 111 hard). It is not the full 534-item ranked benchmark and has no official composite score. '
      'The upstream scoring function is used, including exact labels and its implemented argmax rule for ordinal accuracy. Gold answers, rationales, author metadata, and gold probability distributions are excluded from prompts. '
      'The request guard is raised to 8,192 tokens for this diagnostic (longest observed prompt: 4,071); no evidence is truncated. The temperature is transferred from synthetic calibration without refitting.',
      '', '| System | Easy (48) | Original (72) | Hard (111) | All public (231) |','|---|---:|---:|---:|---:|']
    for label,name in [('base','Base Gemma 4 12B — this run'),('adapter','OpenJeff — this run')]:
        text.append('| '+name+' | '+' | '.join(pct(b['models'][label][tier]['accuracy']) for tier in ('easy','original','hard','all'))+' |')
    for label in ('jev-1.13.0','djev','winnow-12b'):
        rs=ref['systems'][label]
        text.append('| '+rs['name']+' — publisher result | '+' | '.join(pct(rs['tiers'][tier]['accuracy']) for tier in ('easy','original','hard','all'))+' |')
    text += ['', 'Reference rows are recomputed from the publisher’s recorded outcomes on the **same public IDs**. They use different runtimes and prompts and were not rerun here. '
      '[Pinned publisher evidence](https://github.com/fstandhartinger/jevbench/blob/f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84/results/v1.2/jevbench-v1.2-per-task.json).',
      '', '| Public hard probability quality | Base | OpenJeff |','|---|---:|---:|',
      f'| ECE, 10 bins | {b["models"]["base"]["hard"]["ece_10_bins_valid_labeled"]:.4f} | {b["models"]["adapter"]["hard"]["ece_10_bins_valid_labeled"]:.4f} |',
      f'| Mean absolute error versus exact gold distributions | {b["models"]["base"]["hard"]["gold_probability_mae"]:.4f} | {b["models"]["adapter"]["hard"]["gold_probability_mae"]:.4f} |',
      '', 'These public-task measurements are a transfer test, not evidence that synthetic calibration is valid for every enterprise workflow. No checkpoint or temperature was changed after this benchmark.','',
      '## Runtime and cost','',
      '| Public subset, serial local requests | Base | OpenJeff |','|---|---:|---:|',
      f'| Median request time | {1000*b["models"]["base"]["all"]["latency"]["p50_s"]:.1f} ms | {1000*b["models"]["adapter"]["all"]["latency"]["p50_s"]:.1f} ms |',
      f'| p95 request time | {1000*b["models"]["base"]["all"]["latency"]["p95_s"]:.1f} ms | {1000*b["models"]["adapter"]["all"]["latency"]["p95_s"]:.1f} ms |',
      '', 'These times include local prompt compilation and scoring. They exclude cold model loading, networking, admission queues, multi-user contention, and deployment overhead. No production concurrency or SLA is established.',
      '',f'**Cash paid: ${ledger["cash_paid"]:.2f} of the $200 cap.** Credit consumption is accounted for separately in [the ledger](spending.json). '
      f'Automatic top-ups are disabled; all three pods were terminated and the final account spend rate is $0/hour. Last observed credit: ${ledger["last_observed_balance_usd"]:.2f}, provisional because billing can lag. A Python timeout does not terminate rental billing.','',
      '## What the results support','',
      '- A working, reproducible adaptation and probability-readout pipeline over an accepted Google weight lineage, with a real trained adapter and raw evidence.',
      '- A narrow synthetic curriculum can improve these structured rule tasks substantially. The tasks have executable oracles and could be solved deterministically; they do not prove an LLM is needed for those rules.',
      '- Unique case IDs prevent exact serialized request reuse but do not establish distinct reasoning problems. The normalized-state audit found 197/600 final cases matching training states; shared templates remain. This is not a natural-language domain holdout.',
      '- The foundation pretraining corpus is not reproduced. U.S. vendor/weight lineage does not establish exclusively U.S.-sourced pretraining data.',
      '- Container tag and package/model hashes are recorded. The OCI image digest was not captured; bitwise retraining reproducibility is not claimed.',
      '- The B200 diffusion attempt failed during startup before model scoring. Its preflight source/dependency checks passed, but no diffusion inference score was obtained. See `reports/diffusion-attempt.json`. No latent bridge or learned refinement controller is claimed.',
      '- Employment, medical, legal, or other consequential deployment quality has not been established by this pilot.','',
      '## Artifacts and reproduction','',
      '- `runs/pilot-v1/adapter/`: trained LoRA safetensors and loading configuration.',
      '- `runs/pilot-v1/*-calibration.json`: both frozen calibrators, fit hashes, and scorer bindings.',
      '- `runs/pilot-v1/`: raw scores, training log, frozen selection, base-file hashes, runtime, and summaries.',
      '- `data/curriculum-v1/`: original Apache-2.0 adaptation data and manifest.',
      '- [Usage and reproduction](../docs/using-openjeff.md); [architecture program](../docs/research-plan.md); [model card](../model_cards/openjeff-pilot.md).',
      '', '![Measured pilot results](figures/pilot-results.png)','']
    Path('reports/RESULTS.md').write_text('\n'.join(text))
    print('Wrote reports/RESULTS.md')

if __name__=='__main__':main()

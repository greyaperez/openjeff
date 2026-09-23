# Compatibility probe: execution recipe and measured status

Prepared 2026-09-23 UTC. Spending is authorized up to **$200 total**, including
setup, storage, idle time and fees. This first experiment has a **$10 all-in cap**.
The overall plan stops at $150 and keeps $50 in reserve. The first rental and two
model probes have now run; $10 in prepaid credits was purchased. See
[measured results](../reports/gpu-pilot-2026-09-23/README.md) and
[the spending/cleanup ledger](../reports/spending.json). The live
console quote is one U.S.-region H100 PCIe 80 GB with 251 GB RAM and 80 GB
temporary disk: $2.89/hour compute plus $0.011/hour disk (rounded total $2.90).
The 90-minute target is approximately $4.35 before any checkout taxes/fees.

## What has actually passed

- All 21 CPU tests, including probability/calibration checks and the diagnostic
  cases' grouping, labels and answer-token validation.
- Real pinned tokenizers and model-class imports on Transformers 5.17.0 for both
  Gemma 4 E2B and Gemma 4 12B. Each compiled 200 cases successfully. The models use
  different classes: `Gemma4ForConditionalGeneration` and
  `Gemma4UnifiedForConditionalGeneration` respectively.
- An isolated CPU environment avoids the host's incompatible torchvision install.
  Its package inventory is recorded in `reports/probe-cpu-requirements.txt`.

Both pinned models were downloaded and executed on H100. Training remains
untested. These diagnostics are original deterministic examples,
not JevBench or a generalization claim: 100 scenarios across five families, each
in original and reversed candidate order. They cover strict thresholds, policy
precedence, missing evidence, table lookup and date boundaries.

## Launch conditions

Use one U.S.-region GPU with at least 48 GB VRAM and sufficient CPU RAM to stage a
BF16 12B model. Prefer H100 80 GB for the first short compatibility run if its
actual quote fits; L40S 48 GB is an alternative. Provision at least 80 GB temporary
disk, inspect actual free space and model download size before staging. Do not
retain a separately billed network volume for this probe.

Record the checkout quote, taxes/fees, storage and billing increments before
launch. Keep any prepaid funding inside the total cash ceiling; do not count
prepayment and its later consumption twice. Leave automatic top-ups disabled.

Target termination within **90 minutes**, counting image pull and setup.
**Do not use `--terminate-after` or `--stop-after`: Runpod removed these flags
because the API silently ignored their timers.** Their restoration remains a
draft PR as of this check. Older provider examples still contain the broken flags.
See [the merged removal](https://github.com/runpod/runpodctl/pull/330) and
[unmerged restoration](https://github.com/runpod/runpodctl/pull/331).

Use a small prepaid balance with auto-pay verified off, no network volume, an
explicit watchdog that calls the real stop/delete API, and active confirmation
of cleanup. Runpod documents that a zero balance terminates Pods without network
volumes. Billing runs in five-minute cycles, so this is a fallback boundary, not
a precise 90-minute timer or proven zero-overshoot guarantee. See
[billing behavior](https://docs.runpod.io/accounts-billing/billing).
A shell timeout only bounds the process; it does not stop cloud billing.
A best-effort detached stop watchdog was installed on the pilot pod. Its provider
permission was not verified (API reads failed); direct cleanup remains necessary.

Official references: [Runpod pricing](https://www.runpod.io/pricing),
[pod CLI](https://docs.runpod.io/runpodctl/reference/runpodctl-pod).
The displayed checkout and provider record govern the actual rental.

## Prepared job

Build the public-code upload archive locally:

```bash
python -m scripts.build_probe_bundle
```

The bundle uses an explicit source-file allowlist. It excludes local environments,
attachments, payment information, credentials, research downloads and model
weights. `reports/probe-bundle.json` records its contents and checksum.

On the provisioned CUDA 12.8 GPU host, from the extracted `openjeff-probe` directory:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r configs/probe-gpu-requirements.txt
.venv/bin/python -m unittest discover -s tests -v
timeout --signal=TERM --kill-after=30s 45m .venv/bin/python -m scripts.gpu_probe \
  --model google/gemma-4-12B-it --download --max-work-seconds 2400 \
  --out reports/gemma-4-12b-probe
```

The commands above use the tested official Runpod PyTorch 2.8.0 template's
preinstalled CUDA packages. The actual runtime inventory is in the result report;
it is not a fully locked container (its digest was not captured). On another host,
use a clean venv and install the correct official CUDA PyTorch wheels first.
Capture the container digest, driver/CUDA and package inventory on the rented
machine. If setup fails, preserve the error and terminate the pod;
do not turn a compatibility probe into an open-ended environment repair.

The runner checks CUDA before downloading, loads pinned weights without remote
model code, records model-file hashes, performs two warmups, then saves every
candidate-score vector plus serial latency, peak allocated/reserved memory,
accuracy/calibration diagnostics and candidate-order disagreement. Scores are
conditional on the offered candidates and are not calibrated probabilities yet.

Export `reports/gemma-4-12b-probe`, the runtime inventory and logs before deleting
the instance. If the main model succeeds within budget, the E2B control can use
the same command with `--model google/gemma-4-E2B-it` and a separate output path,
only while sufficient time remains before the original termination deadline.

Review these results before allocating training or diffusion experiments. A
successful probe establishes that the measurement path works, not Jev-class
performance.

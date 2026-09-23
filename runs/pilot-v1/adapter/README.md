---
base_model: google/gemma-4-12B-it
base_model_relation: adapter
library_name: peft
license: apache-2.0
language:
- en
tags:
- lora
- calibrated-decisions
- experimental
---

# OpenJeff 0.1.0 pilot adapter

A trained LoRA for finite-choice judgment research. Use the pinned base revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, OpenJeff candidate readout, and matching
calibration artifact. Loading these weights as an ordinary chat model does not
reproduce the reported probability interface.

The adapter has 21,331,968 trainable parameters (rank 16) and was trained on 8,000
original synthetic rule examples. Development selected step 500. No benchmark
example or external teacher API supplied training targets.

See the [complete model card](../../../model_cards/openjeff-pilot.md),
[results](../../../reports/RESULTS.md), and [usage guide](../../../docs/using-openjeff.md).
The package contains the adapter, not the approximately 24 GB foundation weights.

Public JevBench: 208/231 correct, including 89/111 hard items. This is a public-subset
diagnostic, not an official benchmark rank or a production-readiness claim.

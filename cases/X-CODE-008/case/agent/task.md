# X-CODE-008

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Task

On fixed offline small code repositories and unit tests, optimize the training/fine-tuning recipe of a code model so that patch correctness, regression safety and resource cost reach a better trade-off.

On fixed offline small Python repositories, issue/function-level tasks, unit tests and a code model of ≤ 350M, design a training or fine-tuning recipe. You may change LoRA/QLoRA/full fine-tuning, data filtering and deduplication, sample mixing, loss, number of training steps and decoding configuration, but you may not use the network, external code search, browsers, desktop environments, OSWorld, online APIs or robot/environment simulators. The goal is for generated patches to pass the tests on the fixed visible tasks and on tasks, while controlling training GPU-h, peak memory and patch-generation latency and staying stable across seeds.

Anchor: **SWE-bench** (arXiv:2310.06770).

## What you submit

1. Model code `model.py` and trained parameters `checkpoint/`, written against the interface in `interface/`; after submission only these two parts are loaded to run your model.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, definitions, algorithm, complexity and principles of the method. The report explains the design only, with no experimental procedure or numerical results.
3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; when you borrow a component from a paper, state what was borrowed and where it is in the code.

## Constraints

- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.
- Development resources: one 24GB GPU, at most 4 GPU-hours in total during development.
- Submission is one-shot.

## Current status

The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.

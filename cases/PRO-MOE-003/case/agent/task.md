# PRO-MOE-003

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Task

MoE architecture pretraining: about 250M active and ≤ 3B total parameters, pretrained from scratch within 24 hours (on the order of 30B tokens).

Anchor: **DeepSeekMoE** (arXiv:2401.06066).

## What you submit

1. Model code `model.py` and trained parameters `checkpoint/`, written against the interface in `interface/`; after submission only these two parts are loaded to run your model.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, definitions, algorithm, complexity and principles of the method. The report explains the design only, with no experimental procedure or numerical results.
3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; when you borrow a component from a paper, state what was borrowed and where it is in the code.

## Constraints

- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.
- Development resources: 4 × H100, 24 hours wall clock, at most 96 H100-hours in total during development.
- Submission is one-shot.

## Current status

The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.

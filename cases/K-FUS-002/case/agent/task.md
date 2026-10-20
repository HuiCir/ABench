# K-FUS-002

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Task

Operator-fusion kernels.

Anchor: **KernelBench (Level 2 fusion)** (arXiv:2502.10517).

## What you submit

1. The complete implementation of the method, with entry point `method.py`, written against the interface in `interface/`. After submission the code is trained and run from scratch in another environment, so trained parameters need not be submitted.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, definitions, algorithm, complexity and principles of the method. The report explains the design only, with no experimental procedure or numerical results.
3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; when you borrow a component from a paper, state what was borrowed and where it is in the code.

## Constraints

- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.
- Development resources: one 24GB GPU, at most 3 GPU-hours in total during development.
- Submission is one-shot.

## Current status

The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.

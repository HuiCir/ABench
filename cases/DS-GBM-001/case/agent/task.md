# DS-GBM-001: Tabular classification model

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

## Task

Design and train a model for a binary tabular classification task. Tree models, neural networks or a combination of both are all allowed.

Anchor method: **XGBoost** (Chen & Guestrin, arXiv:1603.02754).

## What you submit

1. Model code `model.py` and trained parameters `checkpoint/`, following `interface/interface.md`.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, definitions, algorithm, complexity and rationale of the method. The report explains the design only; it does not describe experiments or numeric results.
3. `references.yaml`: every work cited in the report, in the format of `templates/references_template.yaml`.

## Constraints

- Use only the data provided under `data/`; do not download data or models from the network.
- Training resources: CPU, at most 4 CPU-hours in total during development.
- Submission is one-shot.

# X-ROBUST-009

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Task

On a fixed small encoder and offline intent/safety data, jointly optimize OOD detection, selective abstention and inference cost.

On a fixed small encoder and offline intent/safety data, design OOD-detection and selective-abstention training schemes that are with a better trade-off among reliability, coverage, safety constraints and inference cost.

Anchor: **On Calibration of Modern Neural Networks** (arXiv:1706.04599).

## What you submit

1. Model code `model.py` and trained parameters `checkpoint/`, written against the interface in `interface/`; after submission only these two parts are loaded to run your model.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, definitions, algorithm, complexity and principles of the method. The report explains the design only, with no experimental procedure or numerical results.
3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; when you borrow a component from a paper, state what was borrowed and where it is in the code.

## Constraints

- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.
- Development resources: one 24GB GPU, at most 3 GPU-hours in total during development.
- Submission is one-shot.

## Current status

The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.

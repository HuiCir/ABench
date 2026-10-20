# A-LCTX-006

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Task

Under fixed training and inference budgets, design a long-context/RAG architecture whose recall, retrieval, short-context loss, memory and throughput reach a better trade-off.

Under fixed training and inference budgets, design a position encoding, state cache or context-compression scheme so that long-context recall, short-context loss, memory and throughput reach a better trade-off

Anchor: **YaRN: Efficient Context Window Extension of Large Language Models** (arXiv:2309.00071).

## What you submit

1. The complete implementation of the method, with entry point `method.py`, written against the interface in `interface/`. After submission the code is trained and run from scratch in another environment, so trained parameters need not be submitted.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, definitions, algorithm, complexity and principles of the method. The report explains the design only, with no experimental procedure or numerical results.
3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; when you borrow a component from a paper, state what was borrowed and where it is in the code.

## Constraints

- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.
- Development resources: one 24GB GPU, at most 4 GPU-hours in total during development.
- Submission is one-shot.

## Current status

The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.

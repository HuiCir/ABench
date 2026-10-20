# GN-T7-003: Pitfalls of Graph Neural Network Evaluation

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Question to judge

Judge whether the gains of a given GNN are stable across splits and seeds.

Anchor: **Pitfalls of Graph Neural Network Evaluation** (arXiv:1811.05868).

## What you submit

1. The judgment report is the main deliverable; see the next item.
2. Judgment report `report/index.html`: following the structure of `templates/report_template.html`, give your judgment of the claim, the reasons and a minimal verification plan. Write only analysis and a plan, no numerical results.
3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; when you borrow a component from a paper, state what was borrowed and where it is in the code.

## Constraints

- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.
- This task needs no training.
- Submission is one-shot.

## Current status

The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.

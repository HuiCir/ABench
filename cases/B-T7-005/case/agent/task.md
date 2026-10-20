# B-T7-005: Fantastic Pretraining Optimizers

The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.

## Question to judge

Given a set of published optimizer acceleration claims, predict their true acceleration against tuned baselines and at larger scale.

From published optimizer acceleration claims, predict whether they hold against tuned baselines, at larger scale and at different batch sizes, and give a minimal falsifiable verification plan.

Anchor: **Fantastic Pretraining Optimizers** (arXiv:2509.02046).

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

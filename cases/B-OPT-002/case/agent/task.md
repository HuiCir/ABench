# B-OPT-002

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

## Task

Lion was found by program search over a frozen update grammar, screening candidates on a small proxy and then transferring the selected optimizer to language-model training.
Starting from the **frozen symbolic primitives** given in `interface/interface.md`, design and implement a **deterministically replayable program-search pipeline**,
and use it to find a new optimizer for pretraining a small decoder-only language model. Requirements:

1. **Effective**: under a common training setup, the selected optimizer is no worse than Lion and the best existing optimizers.
2. **Novel**: the selected update rule is behaviorally distinguishable from existing rules such as Lion and AdamW.
3. **Replayable**: record every candidate program, its proxy score and random seed; rerunning the search script reproduces the same candidate sequence.

Anchor: **Lion / Symbolic Discovery of Optimization Algorithms** (arXiv:2302.06675).

## Deliverables

1. The search pipeline and its output: `search.py`, `search_spec.yaml`, `search_log.jsonl`;
   the final rule `optimizer.py` and its hyperparameter declaration `hparam_space.yaml`, written against `interface/interface.md`.
   After submission the search is replayed and the rule is trained from scratch in another environment, so no trained parameters are needed.
2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design of the search space and search strategy,
   and the definition, algorithm, complexity and rationale of the selected rule. The report explains the design only; it does not describe experiments or numeric results.
3. `references.yaml`: every work cited in the report, in the format of `templates/references_template.yaml`. When you borrow a component from a work,
   state what was borrowed and where it is in the code.

## Rules

- Candidate programs may only use the frozen grammar; at most 256 candidates; the search seed is fixed at 17.
- Use only the interfaces provided in `interface/`; do not download data or models from the network.
- Development resources: one 24GB GPU, at most 8 GPU-hours in total during development.
- Submission is one-shot.

# ABench Case BI-PROT-001: Design a protein mutation-fitness prediction model (zero-shot or few-shot)

- Track: BI · Biology · domain
- Anchor: ProteinGym (arXiv: to be verified)
- Subset: main; difficulty: medium; resource tier: medium; status: **deferred**
- Task: Design a protein mutation-fitness prediction model (zero-shot or few-shot)
- Metrics: Spearman (averaged per assay); performance on hidden assays
- Proxy: FROZEN; plan: Frozen ESM-2 150M / 650M representations + lightweight head, no re-pretraining; 30 hidden DMS assays

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `licensed_biological_dataset_snapshot`
- `family_or_experiment_split`
- `domain_expert_review`

Until these inputs are frozen, only interfaces/contracts are added; no excessive environment setup or simulated substitutes.

## Files

- `case/anchor_card.yaml`: task anchor and acceptance boundary.
- `case/task_prompt.md`: the task given to the agent under test.
- `case/eval_protocol.yaml`: proxy ladder, budget, hidden evaluation and output constraints.
- `case/resources.yaml`: registry of data, models, software, hardware and private resources.
- `case/scoring_config.yaml`: metrics and thresholds pending calibration.
- `harness/README.md`: execution interface conventions.
- `scoring/README.md`: scoring input/output conventions.
- `example_submission/README.md`: example submission directory conventions.

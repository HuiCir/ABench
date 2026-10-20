# ABench Case LG-LONG-001: Design a classification architecture for long court decisions under fixed compute (hierarchical encoding, sparse attention, etc.)

- Track: LG · Law · domain
- Anchor: LexGLUE (arXiv: 2110.00976)
- Subset: main; difficulty: medium; resource tier: medium; status: **deferred**
- Task: Design a classification architecture for long court decisions under fixed compute (hierarchical encoding, sparse attention, etc.)
- Metrics: micro/macro-F1 (ECtHR and other subtasks); maximum processable length
- Proxy: SUBSET; plan: ECtHR subtasks, encoder ≤ 150M, maximum length 4k

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `licensed_legal_dataset_snapshot`
- `jurisdiction_and_time_split`
- `blind_legal_review_protocol`

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

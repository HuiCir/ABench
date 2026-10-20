# ABench Case LG-RSN-003: With a fixed base model, design a retrieval and reasoning pipeline for legal reasoning

- Track: LG · Law · domain
- Anchor: LegalBench (arXiv: 2308.11462)
- Subset: main; difficulty: medium; resource tier: medium; status: **deferred**
- Task: With a fixed base model, design a retrieval and reasoning pipeline for legal reasoning
- Metrics: Mean task accuracy; accuracy on unseen tasks
- Proxy: INFER; plan: ≤ 1.5B base, 20 LegalBench subtasks

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `licensed_legal_dataset_snapshot`
- `jurisdiction_and_time_split`
- `blind_legal_review_protocol`
- `fixed_checkpoint_and_runtime_stack`
- `specific_anchor_or_citation_verification`

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

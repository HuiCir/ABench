# ABench Case T-FM-001: Given a snapshot of the generative-modeling literature, improve an action-generation head

- Track: T · Cross-domain transfer · method
- Anchor: Flow Matching → action policy (arXiv: 2210.02747)
- Subset: main; difficulty: high; resource tier: high; status: **deferred**
- Task: Given a snapshot of the generative-modeling literature, improve an action-generation head
- Metrics: Embodied metrics + correctness of source claims
- Proxy: FROZEN, SUBSET; plan: Same reduced setting as E-ACT-001

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `source_case_contracts_and_cross_domain_holdouts`
- `domain_metric_calibration`
- `reference_runs`

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

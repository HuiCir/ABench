# ABench Case FN-TAB-003: Model design for tabular tasks such as credit risk

- Track: FN · Finance · domain
- Anchor: TabPFN (arXiv: 2207.01848)
- Subset: main; difficulty: medium; resource tier: low; status: **deferred**
- Task: Model design for tabular tasks such as credit risk
- Metrics: AUC; calibration error; out-of-time performance
- Proxy: SUBSET; plan: CPU only, 5 credit-risk datasets

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `licensed_time_forward_data_snapshot`
- `transaction_cost_and_survivorship_rules`
- `post_cutoff_hidden_window`
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

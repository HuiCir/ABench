# ABench Case U-LOOP-002: Search adaptive halting/exit rules so a looped model allocates test-time compute by sample difficulty

- Track: U · Non-mainstream paradigms · frontier
- Anchor: Recurrent Depth (Huginn) (arXiv: 2502.05171)
- Subset: primitive; difficulty: high; resource tier: medium; status: **scaffold**
- Task: Search adaptive halting/exit rules so a looped model allocates test-time compute by sample difficulty
- Metrics: Accuracy–test-time FLOPs Pareto (synthetic reasoning tasks)
- Proxy: SYN; plan: Accuracy–test-time-compute Pareto on synthetic reasoning tasks such as arithmetic and graph reachability

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case can enter the next implementation batch of the shared runner, but real scores still cannot be published on this basis.

Blocking inputs:

- `frozen_checkpoint_or_synthetic_task_suite`
- `long_context_and_compute_calibration`
- `reference_runs`
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

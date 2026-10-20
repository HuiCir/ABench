# ABench Case PH-WELL-002: A general surrogate model across multiple physical systems

- Track: PH · Physics · domain
- Anchor: The Well (arXiv: 2412.00568)
- Subset: main; difficulty: high; resource tier: high; status: **deferred**
- Task: A general surrogate model across multiple physical systems
- Metrics: VRMSE; error at unseen parameters
- Proxy: SUBSET; plan: Only 3 sub-datasets, downsampled to 64², total data < 200GB

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `PDEBench_or_simulator_snapshot`
- `numerical_solver_baselines`
- `resolution_and_parameter_holdouts`
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

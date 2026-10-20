# ABench Case PH-T7-004: Judge whether the speedup claims of a given neural PDE solver hold against numerical baselines at equal accuracy and on equal hardware

- Track: PH · Physics · domain
- Anchor: Weak baselines and reporting biases in ML for fluid PDEs (arXiv: 2407.07218)
- Subset: calibration; difficulty: medium; resource tier: low; status: **deferred**
- Task: Judge whether the speedup claims of a given neural PDE solver hold against numerical baselines at equal accuracy and on equal hardware
- Metrics: Judgment accuracy
- Proxy: JUDGE; plan: Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `PDEBench_or_simulator_snapshot`
- `numerical_solver_baselines`
- `resolution_and_parameter_holdouts`

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

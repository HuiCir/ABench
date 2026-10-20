# ABench Case U-LOOP-001: Design a language model with recurrent (weight-shared) depth that trades more loops for lower loss at the same parameter count

- Track: U · Non-mainstream paradigms · frontier
- Anchor: Universal Transformers (arXiv: 1807.03819)
- Subset: main; difficulty: high; resource tier: high; status: **scaffold**
- Task: Design a language model with recurrent (weight-shared) depth that trades more loops for lower loss at the same parameter count
- Metrics: Loss at equal parameters / equal FLOPs; length extrapolation; loop count–performance curve
- Proxy: MICRO, SYN; plan: Equal-parameter comparison at μ1–μ2 with ≤ 8 loops; only 1 hidden point at μ3

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

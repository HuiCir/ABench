# ABench Case U-TTT-008: Design a sequence layer whose hidden-state update is online learning

- Track: U · Non-mainstream paradigms · frontier
- Anchor: TTT layers (RNNs with test-time training) (arXiv: 2407.04620)
- Subset: main; difficulty: high; resource tier: high; status: **deferred**
- Task: Design a sequence layer whose hidden-state update is online learning
- Metrics: Long-context loss as a function of position; retrieval probes; throughput
- Proxy: SURG, SYN; plan: Replace some layers of SmolLM2-135M with TTT layers and continue training for 50M tokens; long-context recall probes up to 8k

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

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

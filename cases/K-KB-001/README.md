# ABench Case K-KB-001: Write correct and faster kernels for single operators

- Track: K · Systems and kernels · method
- Anchor: KernelBench (Level 1 subset) (arXiv: 2502.10517)
- Subset: main; difficulty: high; resource tier: medium; status: **deferred**
- Task: Write correct and faster kernels for single operators
- Metrics: fast_p; geometric-mean speedup (strong baseline)
- Proxy: STREAM; plan: 30-problem subset of KernelBench Level 1, timed on the reference card, four hidden input distributions

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `CUDA_toolchain_and_reference_gpu`
- `sandboxed_kernel_compilation`
- `hidden_input_distributions_and_strong_library_baseline`
- `signed_workload_and_wall_clock_measurement`

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

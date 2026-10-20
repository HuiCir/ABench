# ABench Case K-T7-004: Audit a batch of submitted kernels and find the samples that scored through weak baselines or narrow tests

- Track: K · Systems and kernels · method
- Anchor: KernelBench reward-hacking audit (to be verified)
- Subset: stress; difficulty: high; resource tier: low; status: **deferred**
- Task: Audit a batch of submitted kernels and find the samples that scored through weak baselines or narrow tests
- Metrics: Detection rate; false-positive rate
- Proxy: JUDGE; plan: Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `CUDA_toolchain_and_reference_gpu`
- `sandboxed_kernel_compilation`
- `hidden_input_distributions_and_strong_library_baseline`

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

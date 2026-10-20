# ABench Case X-CODE-008: Training optimization for an offline code-repair model

- Track: X · Training recipes / RL / PEFT · method
- Anchor: SWE-bench (arXiv: 2310.06770)
- Subset: main; difficulty: high; resource tier: medium; status: **scaffold**
- Task: On fixed small Python repositories, unit tests and a small code model, optimize the fine-tuning/training recipe so that patch correctness, regression safety, training cost and inference cost form an auditable Pareto front.
- Run boundary: fully offline; no browser, desktop, OSWorld, online APIs, external code search or robot/environment simulators. Code execution is limited to the fixed repositories and tests provided by the server.
- Metrics: visible and hidden test pass rate; regression failure rate; training GPU-h; peak memory; patch-generation latency; stability across repeated runs.
- Resources: `posttrain_24gb`, one 24GB GPU; development ≤ 4 GPU-h, per verification ≤ 2 GPU-h, reference precompute ≤ 25 GPU-h.

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract, offline harness interface and directory-level notes. There is no training, no code execution, no simulated test result, no simulated reference run and no fabricated leaderboard score. For formal runs, the benchmark operator injects signed fixed repositories, tests and model snapshots.

## Blockers and reservations

Must be frozen before formal activation:

- `offline_repo_bundle_and_test_manifest`
- `frozen_code_model_tokenizer_and_training_split`
- `hidden_patch_tests_and_regression_suite`
- `server_patch_sandbox_and_resource_measurement`

Submissions must not access the network, files outside the repository or hidden tests; the judge applies patches in an isolated directory and runs the fixed tests, and failures, timeouts and irreproducibility are written to structured records rather than interpolated into a score.

## Files

- `case/anchor_card.yaml`: task anchor, resources and acceptance boundary.
- `case/task_prompt.md`: the task given to the agent under test.
- `case/eval_protocol.yaml`: offline proxy ladder, budget, hidden evaluation and output constraints.
- `case/resources.yaml`: registry of fixed repositories, models, software, hardware and private resources.
- `case/scoring_config.yaml`: Pareto metrics and gate contract.
- `harness/README.md`: interface conventions for patch generation, application and test execution.
- `scoring/README.md`: scoring input/output conventions.
- `example_submission/README.md`: example submission directory conventions.

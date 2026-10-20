# ABench Case A-NORM-005: Normalization and residual-scaling scheme (calibration)

- Track: A · LLM architecture · method
- Anchor: RMSNorm (arXiv: 1910.07467)
- Subset: calibration; difficulty: low; resource tier: medium; status: **scaffold**
- Task: Normalization and residual-scaling scheme (calibration)
- Metrics: Loss; training stability (number of loss spikes)
- Proxy: MICRO; plan: μ1–μ2 ladder, count loss spikes

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case can enter the next implementation batch of the shared runner, but real scores still cannot be published on this basis.

Blocking inputs:

- `frozen_tokenizer_model_data_and_long_context_probe`
- `reference_runs_and_fixed_hardware_throughput`

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

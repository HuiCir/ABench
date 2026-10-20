# ABench Case A-HYB-001: Design a sequence mixer or a hybrid layer layout that keeps both language-modeling loss and retrieval ability at fixed FLOPs

- Track: A · LLM architecture · method
- Anchor: Mamba (arXiv: 2312.00752)
- Subset: main; difficulty: medium; resource tier: high; status: **scaffold**
- Task: Design a sequence mixer or a hybrid layer layout that keeps both language-modeling loss and retrieval ability at fixed FLOPs
- Metrics: Loss at fixed FLOPs; long-context retrieval probes; throughput–context curve
- Proxy: SYN, MICRO; plan: Synthetic probes such as MQAR recall + μ1–μ3 language modeling, training context ≤ 4k; the throughput–context curve is inference only, measured up to 16k

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

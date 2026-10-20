# ABench Case X-MUP-003: Hyperparameter-transfer scheme so that hyperparameters tuned on a small model remain optimal on a large one

- Track: X · Training recipes / RL / PEFT · method
- Anchor: μP / Tensor Programs V (arXiv: 2203.03466)
- Subset: main; difficulty: medium; resource tier: medium; status: **scaffold**
- Task: Hyperparameter-transfer scheme so that hyperparameters tuned on a small model remain optimal on a large one
- Metrics: Loss gap after transfer
- Proxy: MICRO; plan: Tune at μ1, transfer to μ2 / μ3 and measure the gap

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case can enter the next implementation batch of the shared runner, but real scores still cannot be published on this basis.

Blocking inputs:

- `frozen_base_model_and_task_split`
- `rollout_or_reward_environment_for_rl_cases`
- `hidden_reward_hacking_checks`

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

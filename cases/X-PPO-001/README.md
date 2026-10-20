# ABench Case X-PPO-001: Policy optimization algorithm for small environments (calibration)

- Track: X · Training recipes / RL / PEFT · method
- Anchor: PPO (arXiv: 1707.06347)
- Subset: calibration; difficulty: low; resource tier: low; status: **scaffold**
- Task: Policy optimization algorithm for small environments (calibration)
- Metrics: Return at a fixed sample count
- Proxy: SUBSET; plan: Small continuous-control environments, 5 seeds, fixed sample count

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

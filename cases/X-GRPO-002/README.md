# ABench Case X-GRPO-002: Post-training RL objective for verifiable rewards

- Track: X · Training recipes / RL / PEFT · method
- Anchor: DeepSeekMath (GRPO) (arXiv: 2402.03300)
- Subset: main; difficulty: medium; resource tier: high; status: **deferred**
- Task: Post-training RL objective for verifiable rewards
- Metrics: Math accuracy; reward-hacking check
- Proxy: SUBSET; plan: 0.5B instruct model, verifiable tasks such as Countdown, rollouts ≤ 512 tokens, 200 steps, 3 seeds

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

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

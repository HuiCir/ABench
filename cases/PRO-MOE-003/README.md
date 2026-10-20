# ABench Case PRO-MOE-003: MoE architecture pretraining: about 250M active and ≤ 3B total parameters, pretrained from scratch within 24 hours (on the order of 30B tokens)

- Track: PRO · Industrial-scale essentials · pro
- Anchor: DeepSeekMoE (arXiv: 2401.06066)
- Subset: main; difficulty: high; resource tier: pro; status: **deferred**
- Task: MoE architecture pretraining: about 250M active and ≤ 3B total parameters, pretrained from scratch within 24 hours (on the order of 30B tokens)
- Metrics: Validation loss and downstream evaluation at equal active FLOPs; load balance; MFU and token throughput
- Proxy: EVAL, AUDIT; plan: Evaluate validation loss and downstream tasks for about 2 H100-h + sampled resume; references are a DeepSeekMoE-style fine-grained recipe and an auxiliary-loss-free balancing recipe

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `4xH100_scheduler_and_network_sandbox`
- `hidden_tasks_and_post_cutoff_eval`
- `hash_chain_and_1h_resume_audit`
- `pro_hardware_quota_and_resume_audit`

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

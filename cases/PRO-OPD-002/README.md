# ABench Case PRO-OPD-002: On-policy distillation: teacher ≤ 32B, student ≤ 2B; design the distillation objective and sampling strategy for math and code reasoning

- Track: PRO · Industrial-scale essentials · pro
- Anchor: On-Policy Distillation of Language Models (GKD) (arXiv: 2306.13649)
- Subset: main; difficulty: high; resource tier: pro; status: **deferred**
- Task: On-policy distillation: teacher ≤ 32B, student ≤ 2B; design the distillation objective and sampling strategy for math and code reasoning
- Metrics: Student accuracy on new problems created after the cutoff; teacher and student tokens needed to reach the same accuracy; comparison with SFT+RL
- Proxy: EVAL, AUDIT; plan: Teacher inference and student training share the 4 GPUs; evaluation about 4 H100-h + sampled resume; references are one off-policy KD and one SFT+RL run

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `4xH100_scheduler_and_network_sandbox`
- `hidden_tasks_and_post_cutoff_eval`
- `hash_chain_and_1h_resume_audit`
- `pro_hardware_quota_and_resume_audit`
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

# ABench Case PRO-SPEC-006: Train a speculative-decoding draft model for an 8B-class target model

- Track: PRO · Industrial-scale essentials · pro
- Anchor: EAGLE-3 (arXiv: 2503.01840)
- Subset: main; difficulty: high; resource tier: pro; status: **deferred**
- Task: Train a speculative-decoding draft model for an 8B-class target model
- Metrics: End-to-end speedup on hidden workloads; acceptance length; losslessness check
- Proxy: EVAL, AUDIT; plan: Evaluate end-to-end speedup for about 2 H100-h + sampled resume; the reference is the official EAGLE-3 training recipe

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

# ABench Case A-T7-006: Given several published conclusions about hybrid architectures, judge which hold at larger scale or longer context, and design a minimal verification experiment

- Track: A · LLM architecture · method
- Anchor: Systematic analysis of hybrid linear attention (specific paper to be chosen) (arXiv: to be verified)
- Subset: calibration; difficulty: medium; resource tier: low; status: **deferred**
- Task: Given several published conclusions about hybrid architectures, judge which hold at larger scale or longer context, and design a minimal verification experiment
- Metrics: Judgment accuracy; information content of the verification experiment
- Proxy: JUDGE; plan: Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

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

# ABench Case U-DLM-004: Improve the noise process or training objective of a discrete diffusion language model to narrow the perplexity gap to autoregressive models

- Track: U · Non-mainstream paradigms · frontier
- Anchor: MDLM (masked diffusion language model) (arXiv: 2406.07524)
- Subset: main; difficulty: high; resource tier: high; status: **deferred**
- Task: Improve the noise process or training objective of a discrete diffusion language model to narrow the perplexity gap to autoregressive models
- Metrics: Perplexity upper bound at equal compute; gap to the AR baseline; generative perplexity vs NFE
- Proxy: MICRO; plan: μ2–μ3 scale; compare how the gap to an equal-compute autoregressive baseline changes with scale

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `frozen_checkpoint_or_synthetic_task_suite`
- `long_context_and_compute_calibration`
- `reference_runs`
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

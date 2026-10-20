# ABench Case U-DLM-006: Discrete diffusion training objective (calibration)

- Track: U · Non-mainstream paradigms · frontier
- Anchor: SEDD (score-entropy discrete diffusion) (arXiv: 2310.16834)
- Subset: calibration; difficulty: low; resource tier: medium; status: **scaffold**
- Task: Discrete diffusion training objective (calibration)
- Metrics: Perplexity upper bound
- Proxy: MICRO; plan: μ1–μ2 ladder

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case can enter the next implementation batch of the shared runner, but real scores still cannot be published on this basis.

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

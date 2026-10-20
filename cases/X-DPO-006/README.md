# ABench Case X-DPO-006: On fixed offline preference pairs, compare and improve DPO/IPO/KTO/ORPO-style objectives and sampling recipes so that preference accuracy, KL drift, length/reward hacking and training cost form a Pareto front

This case does not target SOTA on a single reward or preference accuracy; it is evaluated by the Pareto front over preference, KL drift, length bias and training cost.

- Corpus cutoff: 2026-07-01.
- Difficulty: **high**; resource rating: **medium**; budget tier: **medium**.
- Proxy: SURG, SUBSET; development about 4 GPU-h, per verification about 2 GPU-h, reference precompute about 25 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

Fixed small base and offline preference pairs; at most 5k training steps; a small public split is visible, and hidden counterfactual and safety probes run only server-side.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: quality metrics are maximized, cost metrics minimized, and constraint metrics must pass first; the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `frozen_base_and_reference_checkpoint_hash`
- `offline_preference_pairs_and_hidden_counterfactuals`
- `reference_runs_and_kl_length_evaluator`


Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `posttrain_24gb`; objective mode: `objective_mode: pareto`.

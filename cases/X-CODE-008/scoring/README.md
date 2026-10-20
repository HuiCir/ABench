# X-CODE-008 scoring contract

The scorer accepts only real run records produced by the benchmark operator and blind-review/reference results; it does not generate simulated results.

Required inputs:

- submission manifest, model/code/data/repository hashes and generated patches;
- server-side `run.jsonl` (with seeds, resources, time, patch status, test-failure categories and raw metrics);
- reference envelope and hidden evaluation;
- if a local discovery is claimed, the source ID, where it is used and the ablation record.

The scorer first applies legality, patch-boundary, budget, regression, safety and leakage gates, then outputs a machine-readable non-dominated set according to `case/scoring_config.yaml`. When real reference runs are missing it returns `pending_reference_runs` instead of substituting placeholder data.

# U-DLM-006 scoring contract

The scorer accepts only real run records produced by the benchmark operator and blind-review/reference results; it does not generate simulated results.

Required inputs:

- submission manifest and hashes of the submitted artifacts;
- server-side `run.jsonl` (with seeds, resources, time, failure status and raw metrics);
- reference envelope and hidden evaluation;
- if a local discovery is claimed, the source ID, where it is used and the ablation record.

The scorer first applies legality, budget, equivalence and leakage gates, then normalizes according to `case/scoring_config.yaml` and outputs a machine-readable report. When real reference runs are missing it returns `pending_reference_runs` instead of substituting placeholder data.

> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# DS-GBM-001 task statement

What you need to do: Design a tabular model within a CPU budget (tree model, neural network or hybrid)

Anchor method: XGBoost. This is a main-subset task in the DS track. Raw evaluation metrics: Normalized accuracy/AUC rank over multiple datasets; training time.

## Submission

- submission/manifest.yaml: method, dependencies, random seeds, resource declaration and sources.
- submission/run.py or an equivalent entry point: runs only through the interface defined in harness/README.md.
- submission/design.md: design motivation, differences from the anchor and strong baselines, known limitations.
- submission/metrics_request.yaml: allowed hyperparameter ranges and defaults, with no information about the hidden set.
- A reproducible model or pipeline implementation and configuration.

## Constraints

- The corpus cutoff is 2026-07-01; sources, code and data must be traceable.
- Do not read hidden data, labels, evaluator internal state or other submissions.
- Do not submit or cite simulated run results; the example directory only validates the format.
- The benchmark operator produces all real run results, resource usage and final scores.
- CPU only, 20 hidden OpenML datasets, each ≤ 50k rows.

## Evaluation focus

Primary metric: Normalized accuracy/AUC rank over multiple datasets; training time. Full metrics, splits, thresholds and fields still to be frozen are in case/eval_protocol.yaml and case/scoring_config.yaml.

## ABench unified experiment report package (`domain_pipeline_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `experiment_report.md`, `report_manifest.json`, `data_provenance.yaml` and `evidence.yaml` (`claim_evidence.jsonl` during migration); when external methods, models or data are claimed, also submit `references.yaml`/`sources.yaml` and `retrieval_log.jsonl`. The report must describe licenses and snapshots, data splits, leak prevention, the fitting scope of preprocessing, baselines, ablations, out-of-distribution generalization, metric units, server-side measurement, real results or `pending`, limitations and the reproduction command. This profile is exempt from `architecture_report.*`, but not from data provenance and measurement evidence; self-reported numbers in the report are not scored. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# GN-T7-003 task statement

What you need to do: Judge whether the gains of a given GNN are stable across splits and seeds

Anchor method: Pitfalls of Graph Neural Network Evaluation. This is a calibration-subset task in the GN track. Raw evaluation metrics: Judgment accuracy.

## Submission

- submission/manifest.yaml: method, dependencies, random seeds, resource declaration and sources.
- submission/run.py or an equivalent entry point: runs only through the interface defined in harness/README.md.
- submission/design.md: design motivation, differences from the anchor and strong baselines, known limitations.
- submission/metrics_request.yaml: allowed hyperparameter ranges and defaults, with no information about the hidden set.
- Judgment evidence package and design of a minimal verification experiment.

## Constraints

- The corpus cutoff is 2026-07-01; sources, code and data must be traceable.
- Do not read hidden data, labels, evaluator internal state or other submissions.
- Do not submit or cite simulated run results; the example directory only validates the format.
- The benchmark operator produces all real run results, resource usage and final scores.
- Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training.

## Evaluation focus

Primary metric: blind judgment accuracy with calibrated confidence. Full metrics, splits, thresholds and fields still to be frozen are in case/eval_protocol.yaml and case/scoring_config.yaml.

## ABench judgment report package (`judgment_review_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `judgment_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration), and keep a `retrieval_log.jsonl` for every external claim. The report must provide a claim table, supporting evidence, counterexamples/negative results, source cutoffs, evidence strength, uncertainty, conflict handling and the shortest review command; without real verification it is marked `pending` and a self-reported conclusion cannot replace hidden review. This profile is exempt from `architecture_report.*`. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


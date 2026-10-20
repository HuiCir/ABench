> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# IC-AUG-003 task statement

What you need to do: Search data-augmentation policies starting from image-transformation primitives

Anchor method: AutoAugment. This is a primitive-subset task in the IC track. Raw evaluation metrics: Accuracy gain; search cost.

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
- During search, candidate policies are evaluated with the CIFAR-10 speedrun recipe.

## Evaluation focus

Primary metric: Accuracy gain; search cost. Full metrics, splits, thresholds and fields still to be frozen are in case/eval_protocol.yaml and case/scoring_config.yaml.

## ABench unified design report package (`architecture_design_v1`)

Once this case enters `active_v1` or `proxy_ready`, in addition to the original deliverables of this task, `submission/` must also contain:

- `architecture_report.html`: the main report, offline, self-contained and directly readable;
- `architecture_report.md`: the diffable canonical source;
- `report_manifest.json`: index of case/profile, sections, file paths and hashes;
- `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration);
- when external papers, code, data or models are used, keep `retrieval_log.jsonl` recording queries, candidate screening, snapshots and where they are cited in the text.

The report must cover at least: problem and constraints, anchor/strong baselines and retrieval evidence, modules and data flow, input/output shapes, parameter/FLOPs/memory or KV budget, equations and invariants, code locations, runnable ablations, Pareto axes or quality thresholds, real results/`pending` status, limitations and the shortest reproduction command. Without real server-side artifacts only `pending` may be written; simulated numbers are forbidden, and self-reported numbers in the report are not scored. The full fields, HTML section anchors and offline resource constraints are in `spec/report_contract.md` and `spec/report_profiles.yaml`.

In the `contract_only` or `deferred` stage only the interfaces and templates above are frozen and no retrieval/citation budget is consumed; once the case is activated, submissions must use the same package format.


> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task C-T7-005 (original text given to the agent under test)

Corpus cutoff: **2026-07-01**. Anchor: **Analysis of speculative-decoding efficiency** (to be frozen).

## Goal

Judge whether speculative-decoding speedup claims hold under low acceptance rates or large-batch serving, and design a minimal falsifiable experiment.

## Evaluation metrics

- Judgment accuracy
- Speedup prediction error
- Falsifiability of the experimental design

## Delivery constraints

- Use only the resources and interfaces declared in the task card; do not read hidden inputs, final-evaluation results or the internal state of the evaluation service.
- Deliverables must be reproducible in the runtime described in `case/bench_contract.yaml`.
- Record configuration, code, data, runtime and output hashes; do not submit simulated curves or self-reported scores.
- Declare complex external dependencies through public interfaces, and run them once the benchmark operator freezes the resources.

## ABench judgment report package (`judgment_review_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `judgment_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration), and keep a `retrieval_log.jsonl` for every external claim. The report must provide a claim table, supporting evidence, counterexamples/negative results, source cutoffs, evidence strength, uncertainty, conflict handling and the shortest review command; without real verification it is marked `pending` and a self-reported conclusion cannot replace hidden review. This profile is exempt from `architecture_report.*`. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


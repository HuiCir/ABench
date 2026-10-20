> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task C-SPEC-001 (original text given to the agent under test)

Corpus cutoff: **2026-07-01**. Anchor: **Fast Inference via Speculative Decoding** (2211.17192).

## Goal

Design the draft model and verification strategy to increase end-to-end inference speed while keeping the output distribution unchanged.

## Evaluation metrics

- End-to-end speedup
- Draft-token acceptance rate
- Output-distribution consistency
- Speedup on OOD workloads

## Delivery constraints

- Use only the resources and interfaces declared in the task card; do not read hidden inputs, final-evaluation results or the internal state of the evaluation service.
- Deliverables must be reproducible in the runtime described in `case/bench_contract.yaml`.
- Record configuration, code, data, runtime and output hashes; do not submit simulated curves or self-reported scores.
- Declare complex external dependencies through public interfaces, and run them once the benchmark operator freezes the resources.

## ABench unified systems report package (`system_efficiency_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `system_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the system boundary, shapes/dtypes/layouts, correctness and equivalence checks, the end-to-end timing boundary, throughput/latency/memory measurement, baselines, runnable ablations, real results or `pending`, failure modes and the reproduction command. If the model topology changes, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


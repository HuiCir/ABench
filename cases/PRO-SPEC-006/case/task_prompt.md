> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task PRO-SPEC-006

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

Train a speculative-decoding draft model for an 8B-class target model

Anchor: **EAGLE-3** (arXiv: 2503.01840). This is a T6 case in the Industrial-scale essentials track; target metrics: End-to-end speedup on hidden workloads; acceptance length; losslessness check.

## Proxy and budget

- Proxy: EVAL, AUDIT
- Plan: Evaluate end-to-end speedup for about 2 H100-h + sampled resume; the reference is the official EAGLE-3 training recipe
- Development budget: 96 GPU-h; per verification: 6 GPU-h; reference precompute: 96 GPU-h.
- Proxy notes: evaluation of Pro deliverables; rerun by the benchmark operator; hash chain, sampled resume and audit logs

## Deliverables

The submission directory must contain executable method code, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated training curves, reference results or judge results; real reference runs and the hidden set are held by the benchmark operator. All random seeds, data splits, model/weight hashes and hardware timings must be written to the run record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, performs formal verification on hidden inputs/tasks, and checks resources, equivalence, leakage and failure states. This directory is a bench scaffold and provides no real scores.

## ABench unified systems report package (`system_efficiency_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `system_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the system boundary, shapes/dtypes/layouts, correctness and equivalence checks, the end-to-end timing boundary, throughput/latency/memory measurement, baselines, runnable ablations, real results or `pending`, failure modes and the reproduction command. If the model topology changes, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


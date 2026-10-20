> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task T-SPEC-003

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

Apply the draft-and-verify idea to few-step diffusion sampling

Anchor: **Speculative decoding → diffusion sampling** (arXiv: 2211.17192). This is a TX case in the Cross-domain transfer track; target metrics: FID at NFE + source claims.

## Proxy and budget

- Proxy: INFER
- Plan: Fixed pretrained diffusion weights (same as V-SMP-002)
- Development budget: 3 GPU-h; per verification: 1 GPU-h; reference precompute: 5 GPU-h.
- Proxy notes: inference only; weights fixed, end-to-end latency and equivalence measured server-side

## Deliverables

The submission directory must contain executable method code, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated training curves, reference results or judge results; real reference runs and the hidden set are held by the benchmark operator. All random seeds, data splits, model/weight hashes and hardware timings must be written to the run record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, performs formal verification on hidden inputs/tasks, and checks resources, equivalence, leakage and failure states. This directory is a bench scaffold and provides no real scores.

## ABench unified systems report package (`system_efficiency_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `system_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the system boundary, shapes/dtypes/layouts, correctness and equivalence checks, the end-to-end timing boundary, throughput/latency/memory measurement, baselines, runnable ablations, real results or `pending`, failure modes and the reproduction command. If the model topology changes, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


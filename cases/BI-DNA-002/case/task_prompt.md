> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task BI-DNA-002

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

Long-range genomic sequence model

Anchor: **HyenaDNA** (arXiv: 2306.15794). This is a T5 case in the Biology track; target metrics: Genomic benchmark accuracy; usable context length.

## Proxy and budget

- Proxy: SURG, SUBSET
- Plan: Continue training or fine-tune a small HyenaDNA model from HF on a subset of genomic benchmarks
- Development budget: 4 GPU-h; per verification: 2 GPU-h; reference precompute: 20 GPU-h.
- Proxy notes: frozen small-model surgery/short continued training; weights and preprocessing snapshots must be frozen; subset, fewer-fold or low-resolution proxy; fidelity calibration before launch

## Deliverables

The submission directory must contain executable method code, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated training curves, reference results or judge results; real reference runs and the hidden set are held by the benchmark operator. All random seeds, data splits, model/weight hashes and hardware timings must be written to the run record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, performs formal verification on hidden inputs/tasks, and checks resources, equivalence, leakage and failure states. This directory is a bench scaffold and provides no real scores.

## ABench unified experiment report package (`domain_pipeline_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `experiment_report.md`, `report_manifest.json`, `data_provenance.yaml` and `evidence.yaml` (`claim_evidence.jsonl` during migration); when external methods, models or data are claimed, also submit `references.yaml`/`sources.yaml` and `retrieval_log.jsonl`. The report must describe licenses and snapshots, data splits, leak prevention, the fitting scope of preprocessing, baselines, ablations, out-of-distribution generalization, metric units, server-side measurement, real results or `pending`, limitations and the reproduction command. This profile is exempt from `architecture_report.*`, but not from data provenance and measurement evidence; self-reported numbers in the report are not scored. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


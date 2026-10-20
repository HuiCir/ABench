> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task V-3DG-005

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

Sparse-view novel-view synthesis

Anchor: **3D Gaussian Splatting** (arXiv: 2308.04079). This is a T1 case in the Vision track; target metrics: PSNR / SSIM / LPIPS (hidden poses).

## Proxy and budget

- Proxy: SUBSET
- Plan: 6 small scenes, 3 input views, minutes of per-scene optimization, poses hidden
- Development budget: 3 GPU-h; per verification: 1 GPU-h; reference precompute: 10 GPU-h.
- Proxy notes: subset, fewer-fold or low-resolution proxy; fidelity calibration before launch

## Deliverables

The submission directory must contain executable method code, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated training curves, reference results or judge results; real reference runs and the hidden set are held by the benchmark operator. All random seeds, data splits, model/weight hashes and hardware timings must be written to the run record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, performs formal verification on hidden inputs/tasks, and checks resources, equivalence, leakage and failure states. This directory is a bench scaffold and provides no real scores.

## ABench unified design report package (`architecture_design_v1`)

Once this case enters `active_v1` or `proxy_ready`, in addition to the original deliverables of this task, `submission/` must also contain:

- `architecture_report.html`: the main report, offline, self-contained and directly readable;
- `architecture_report.md`: the diffable canonical source;
- `report_manifest.json`: index of case/profile, sections, file paths and hashes;
- `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration);
- when external papers, code, data or models are used, keep `retrieval_log.jsonl` recording queries, candidate screening, snapshots and where they are cited in the text.

The report must cover at least: problem and constraints, anchor/strong baselines and retrieval evidence, modules and data flow, input/output shapes, parameter/FLOPs/memory or KV budget, equations and invariants, code locations, runnable ablations, Pareto axes or quality thresholds, real results/`pending` status, limitations and the shortest reproduction command. Without real server-side artifacts only `pending` may be written; simulated numbers are forbidden, and self-reported numbers in the report are not scored. The full fields, HTML section anchors and offline resource constraints are in `spec/report_contract.md` and `spec/report_profiles.yaml`.

In the `contract_only` or `deferred` stage only the interfaces and templates above are frozen and no retrieval/citation budget is consumed; once the case is activated, submissions must use the same package format.


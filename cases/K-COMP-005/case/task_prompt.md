> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task K-COMP-005

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

In a fixed operator graph and single-GPU sandbox, design graph-capture, fusion, layout or compilation strategies so that end-to-end throughput, compile overhead, peak memory and numerical error form a better Pareto front

Anchor: **PyTorch 2 torch.compile / Inductor**. This case targets an end-to-end throughput–compile-time–memory–numerical-error Pareto front and does not accept speedups that hold only on a single public shape.

## Proxy and budget

- Proxy: STREAM
- Plan: 12 fixed operator graphs and 3 shape distributions; first-iteration and steady-state performance of eager, torch.compile and the submitted strategy measured on one 24GB GPU.
- Development budget: 2 GPU-h; per verification: 1 GPU-h; reference precompute: 10 GPU-h; difficulty: high; resource rating: medium.

## Deliverables

The submission directory must contain an executable method, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated curves, reference results, judge results or simulated scores. All random seeds, resource/model hashes, data splits, hardware timings and failure states must be written to the record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, reviews on hidden resources, and forms a Pareto front from the non-dominance of quality, resource and constraint metrics.

## ABench unified systems report package (`system_efficiency_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `system_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the system boundary, shapes/dtypes/layouts, correctness and equivalence checks, the end-to-end timing boundary, throughput/latency/memory measurement, baselines, runnable ablations, real results or `pending`, failure modes and the reproduction command. If the model topology changes, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


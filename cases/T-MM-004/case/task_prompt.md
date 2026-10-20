> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task T-MM-004

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

On frozen vision/text encoders and offline image–text pairs, design a lightweight cross-modal fusion or adapter that is Pareto-optimal over retrieval quality, trainable parameters, memory and inference latency

Anchor: **CLIP: Learning Transferable Visual Models From Natural Language Supervision**. This is a low-cost multimodal architecture case. It tests the quality–parameter–latency Pareto front of cross-modal adapters rather than chasing SOTA on a single dataset.

## Proxy and budget

- Proxy: FROZEN, SUBSET
- Plan: Freeze small vision/text encoders, use an offline thumbnail and caption bundle, and train only adapters/cross-attention/temperature; hidden compositional retrieval is evaluated server-side.
- Development budget: 3 GPU-h; per verification: 1.5 GPU-h; reference precompute: 15 GPU-h; difficulty: medium; resource rating: medium.

## Deliverables

The submission directory must contain an executable method, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated curves, reference results, judge results or simulated scores. All random seeds, resource/model hashes, data splits, hardware timings and failure states must be written to the record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, reviews on hidden resources, and forms a Pareto front from the non-dominance of quality, resource and constraint metrics.

## ABench unified design report package (`architecture_design_v1`)

Once this case enters `active_v1` or `proxy_ready`, in addition to the original deliverables of this task, `submission/` must also contain:

- `architecture_report.html`: the main report, offline, self-contained and directly readable;
- `architecture_report.md`: the diffable canonical source;
- `report_manifest.json`: index of case/profile, sections, file paths and hashes;
- `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration);
- when external papers, code, data or models are used, keep `retrieval_log.jsonl` recording queries, candidate screening, snapshots and where they are cited in the text.

The report must cover at least: problem and constraints, anchor/strong baselines and retrieval evidence, modules and data flow, input/output shapes, parameter/FLOPs/memory or KV budget, equations and invariants, code locations, runnable ablations, Pareto axes or quality thresholds, real results/`pending` status, limitations and the shortest reproduction command. Without real server-side artifacts only `pending` may be written; simulated numbers are forbidden, and self-reported numbers in the report are not scored. The full fields, HTML section anchors and offline resource constraints are in `spec/report_contract.md` and `spec/report_profiles.yaml`.

In the `contract_only` or `deferred` stage only the interfaces and templates above are frozen and no retrieval/citation budget is consumed; once the case is activated, submissions must use the same package format.


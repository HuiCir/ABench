> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task A-LCTX-006

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

Under fixed training and inference budgets, design a position encoding, state cache or context-compression scheme so that long-context recall, short-context loss, memory and throughput form a better Pareto front

Anchor: **YaRN: Efficient Context Window Extension of Large Language Models**. This case evaluates only reproducible long-context architecture/training changes without requiring a single SOTA point; final ranking is by Pareto dominance over quality, memory and throughput.

## Proxy and budget

- Proxy: SYN, MICRO, INFER
- Plan: Fixed offline data and a small decoder-only model; maximum training context 4k, hidden inference lengths 8k/16k/32k; all submissions use the same tokenizer, data and base checkpoint.
- Development budget: 4 GPU-h; per verification: 2 GPU-h; reference precompute: 20 GPU-h; difficulty: high; resource rating: medium.

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


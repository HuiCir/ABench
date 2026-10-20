> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task PRO-MOE-003

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

MoE architecture pretraining: about 250M active and ≤ 3B total parameters, pretrained from scratch within 24 hours (on the order of 30B tokens)

Anchor: **DeepSeekMoE** (arXiv: 2401.06066). This is a T4 case in the Industrial-scale essentials track; target metrics: Validation loss and downstream evaluation at equal active FLOPs; load balance; MFU and token throughput.

## Proxy and budget

- Proxy: EVAL, AUDIT
- Plan: Evaluate validation loss and downstream tasks for about 2 H100-h + sampled resume; references are a DeepSeekMoE-style fine-grained recipe and an auxiliary-loss-free balancing recipe
- Development budget: 96 GPU-h; per verification: 8 GPU-h; reference precompute: 192 GPU-h.
- Proxy notes: evaluation of Pro deliverables; rerun by the benchmark operator; hash chain, sampled resume and audit logs

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


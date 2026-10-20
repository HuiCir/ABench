> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task PRO-OPD-002

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

On-policy distillation: teacher ≤ 32B, student ≤ 2B; design the distillation objective and sampling strategy for math and code reasoning

Anchor: **On-Policy Distillation of Language Models (GKD)** (arXiv: 2306.13649). This is a T3 case in the Industrial-scale essentials track; target metrics: Student accuracy on new problems created after the cutoff; teacher and student tokens needed to reach the same accuracy; comparison with SFT+RL.

## Proxy and budget

- Proxy: EVAL, AUDIT
- Plan: Teacher inference and student training share the 4 GPUs; evaluation about 4 H100-h + sampled resume; references are one off-policy KD and one SFT+RL run
- Development budget: 96 GPU-h; per verification: 10 GPU-h; reference precompute: 192 GPU-h.
- Proxy notes: evaluation of Pro deliverables; rerun by the benchmark operator; hash chain, sampled resume and audit logs

## Deliverables

The submission directory must contain executable method code, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated training curves, reference results or judge results; real reference runs and the hidden set are held by the benchmark operator. All random seeds, data splits, model/weight hashes and hardware timings must be written to the run record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, performs formal verification on hidden inputs/tasks, and checks resources, equivalence, leakage and failure states. This directory is a bench scaffold and provides no real scores.

## ABench unified method report package (`post_training_method_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `method_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the data/training recipe, trainable parameters, baselines, hyperparameter-search bounds, forgetting/retention metrics, stability and cost ablations, real results or `pending`, limitations and the reproduction command. If this task adds a model topology or trainable module, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


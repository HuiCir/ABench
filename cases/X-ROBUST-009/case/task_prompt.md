> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task X-ROBUST-009

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

On a fixed small encoder and offline intent/safety data, design calibration, OOD-detection and selective-abstention training schemes that are Pareto-optimal over reliability, coverage, safety constraints and inference cost.

Anchor: **On Calibration of Modern Neural Networks**; related methods include OOD scores, selective prediction, temperature/vector scaling, lightweight abstention heads and OOD exposure. The task requires maintaining risk control under unseen shifts; public ID accuracy cannot stand in for reliability.

## Proxy and budget

- Proxy: FROZEN, SURG, SUBSET.
- Plan: Fixed ≤ 125M encoder, thresholds chosen on the public calibration split; the server replays on unseen topics, compositional near-OOD, length/wording shifts and safety-policy samples.
- Development budget: 3 GPU-h; per verification: 1.5 GPU-h; reference precompute: 15 GPU-h; difficulty: high; resource rating: medium.

## Deliverables

The submission directory must contain an executable method, manifest.json, source declarations, a deterministic run configuration and machine-readable result records. It must record the data split used to choose thresholds, random seeds, resource/configuration/model hashes, per-shift metrics and failure states. Do not submit or rely on fabricated curves, reference results, judge results or simulated scores.

## Evaluation boundary

Use only the offline resources declared in case/resources.yaml. case/eval_protocol.yaml and case/scoring_config.yaml are authoritative. The server first applies ID-quality and safety-constraint gates, then forms a Pareto front from the non-dominance of reliability, coverage, stability and real resource metrics.

## ABench unified method report package (`post_training_method_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `method_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the data/training recipe, trainable parameters, baselines, hyperparameter-search bounds, forgetting/retention metrics, stability and cost ablations, real results or `pending`, limitations and the reproduction command. If this task adds a model topology or trainable module, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


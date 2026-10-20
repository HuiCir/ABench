> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task X-DPO-006

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

On fixed offline preference pairs, compare and improve DPO/IPO/KTO/ORPO-style objectives and sampling recipes so that preference accuracy, KL drift, length/reward hacking and training cost form a Pareto front

Anchor: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**. This case does not target SOTA on a single reward or preference accuracy; it is evaluated by the Pareto front over preference, KL drift, length bias and training cost.

## Proxy and budget

- Proxy: SURG, SUBSET
- Plan: Fixed small base and offline preference pairs; at most 5k training steps; a small public split is visible, and hidden counterfactual and safety probes run only server-side.
- Development budget: 4 GPU-h; per verification: 2 GPU-h; reference precompute: 25 GPU-h; difficulty: high; resource rating: medium.

## Deliverables

The submission directory must contain an executable method, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated curves, reference results, judge results or simulated scores. All random seeds, resource/model hashes, data splits, hardware timings and failure states must be written to the record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, reviews on hidden resources, and forms a Pareto front from the non-dominance of quality, resource and constraint metrics.

## ABench unified method report package (`post_training_method_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `method_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the data/training recipe, trainable parameters, baselines, hyperparameter-search bounds, forgetting/retention metrics, stability and cost ablations, real results or `pending`, limitations and the reproduction command. If this task adds a model topology or trainable module, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


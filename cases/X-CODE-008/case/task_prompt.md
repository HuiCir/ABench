> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task X-CODE-008

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

On fixed offline small Python repositories, issue/function-level tasks, unit tests and a code model of ≤ 350M, design a training or fine-tuning recipe. You may change LoRA/QLoRA/full fine-tuning, data filtering and deduplication, sample mixing, loss, number of training steps and decoding configuration, but you may not use the network, external code search, browsers, desktop environments, OSWorld, online APIs or robot/environment simulators.

The goal is for generated patches to pass the tests on the fixed visible tasks and on hidden server-side tasks, while controlling training GPU-h, peak memory and patch-generation latency and staying stable across seeds. Final evaluation follows the constraints and Pareto front in `case/scoring_config.yaml`; no single SOTA point is required.

## Proxy and budget

- Proxy: SURG + SUBSET + INFER
- Plan: Fixed 125M–350M code checkpoint; signed offline repositories and tests; hidden API/defect-family holdout
- Development budget: 4 GPU-h; per verification: 2 GPU-h; reference precompute: 25 GPU-h; one 24GB GPU

## Deliverables

The submission directory must contain an executable training/inference entry point, `manifest.json`, source declarations, a deterministic configuration, generated patches and machine-readable result records. Patches may be written only to the isolated working directory assigned by the judge; the input repository, tests, labels and shared caches must not be modified. All random seeds, data/model/repository hashes, resources, failure states and test summaries must be written to the run record.

## Evaluation boundary

The server applies patches on an isolated copy and runs the fixed tests, checking syntax, regressions, timeouts, process/file access and network access. Formal evaluation includes unpublished repositories and tests; hidden inputs never appear in the submission directory or logs. This directory is a bench scaffold and provides no real scores.

## ABench unified method report package (`post_training_method_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `method_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the data/training recipe, trainable parameters, baselines, hyperparameter-search bounds, forgetting/retention metrics, stability and cost ablations, real results or `pending`, limitations and the reproduction command. If this task adds a model topology or trainable module, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.


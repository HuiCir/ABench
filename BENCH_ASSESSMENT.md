# ABench assessment and recommendations for further construction

## Conclusion

B-OPT-001 and B-OPT-002 have shown that the contract pattern of `case/`, `harness/`, `scoring/` and `example_submission/` works, but they are not yet a real benchmark that can be released. The curves, reference runs and judge outputs of B-OPT-001 are historical demo data; the search log of B-OPT-002 only validates search artifacts and cannot replace real training and server-side timing. Both cases therefore stay at `scaffold` and must not be marked `built`.

B-OPT-001 still has two key implementation inconsistencies: the protocol states that the gradient stream comes from real AdamW checkpoints, while `scoring/divergence.py` still generates a fixed synthetic stream; and SOAP, Sophia and Schedule-Free from the reference pool are not all wired into the behavioral probes. B-OPT-002 must bind the selected AST, training execution, resource limits and final model result on the server, to avoid "search log passes but candidate is invalid".

## Full flat-index results

The index currently has **98/98 cases at scaffold**. Every case has a uniform run schema, hash/provenance fields, metric registry, task prompt, evaluation protocol, resource registry, scoring entry point and example-submission notes; no simulated training, inference, reference or leaderboard results were generated for the new cases in this round.

Anchor readiness is **52 `deferred` and 46 `contract_only`**; no case has reached `reference_ready` or `hidden_ready`. **56** priority/new cases carry a structured resource profile. **11** cases state `objective_mode: pareto` explicitly in the index and anchor card; the remaining historical tasks keep their gate/judge or original protocol. Strict preflight gives `cases=98 missing=0 errors=0`; resource-profile preflight gives `pareto_cases=48 errors=0`, which includes Pareto configurations from some older protocols and must not be read as a finished real runner.

Under the resource constraints, this round only adds three kinds of content:

- `T-AUDIO-005`: freeze a ≤100M wav2vec 2.0/HuBERT encoder and train only CTC/adapter/pooling; Pareto over WER, out-of-speaker generalization, memory, real-time factor and GPU-h. Fixed 16 kHz, short utterances and a speaker-disjoint hidden split; no TTS, real-time streaming or large-scale speech pretraining.
- `X-DISTILL-010`: freeze a ≤350M teacher and a ≤125M student, compare hard-label, logit, feature and sequence KD; Pareto over hidden capability, forgetting, student throughput, memory and training cost. The teacher cache, tokenizer, initialization and capability-family split must be frozen by the server.
- `C-QNT-003` is narrowed to a low-bit quantization task comparing AWQ/GPTQ/SmoothQuant with an FP16 baseline, adding 2/3/4-bit, group size, low-batch decode, peak memory, p95 latency and a holdout of models/tasks outside the calibration set.

The existing `X-SFT-005`, `X-DPO-006`, `X-DATA-007`, `K-COMP-005`, `T-MM-004`, `X-CODE-008` and `X-ROBUST-009` are kept. `A-LCTX-006` keeps only its skeleton and is marked `deferred_out_of_scope_v1`, with no reference budget this round; no video or large repository-level benchmark is added, and `X-CODE-008` refers only to fixed small repositories.

## Main blockers for the two built cases and the new cases

1. **Real resources are not frozen**: tokenizer, model/teacher/student checkpoints, data splits, audio front end, quantization kernels, dependency images, licenses and SHA256 still have to be written into the server manifest.
2. **Reference runs have not been produced**: each must record seed, code commit, hardware signature, runtime, checkpoint/output hashes, loss/quality, duration, peak memory and failure status; the scorer rejects self-reported timings.
3. **Hidden evaluation is not wired in**: audio needs speaker/accent/noise splits, distillation needs capability-family/template/length splits, quantization needs hidden models/shapes/tasks and a numerical-equivalence check.
4. **Proxy fidelity is not calibrated**: every active case needs at least 5 known method pairs to rank, multi-seed bootstrap and a noise floor; below 80% it is downgraded, below 60% it becomes judge-based or is postponed.
5. **Server-side isolation is still a contract statement**: network, file system, processes, CUDA extensions, caches and time/memory collection must be enforced by the runner, not taken from submitter JSON.
6. **Entry points of older cases may still be broken**: 17 older cases still reference the not-yet-provided `paper_snapshot/seed_neighborhood.jsonl`; all older resources should first be resolved by a preflight that prints a list of what is missing.

## P1 statistics and scoring issues

- A Pareto front must pre-register the quality floor, handling of infeasible points, the hypervolume reference point, confidence intervals and tie-breaks; direction and unit of throughput, latency, memory and GPU-h must be fixed in the metric registry.
- Quantization, audio and distillation must not leak teacher cache, audio preprocessing or kernel compile time out of the cost metrics; teacher forward passes may be cached, but the cost of building the cache is recorded separately.
- Selective abstention, OOD, mechanism judgments and domain reviews need blind review, versioned judges or human agreement records.
- Search-type cases must separate "search quality" from "final model quality"; passing an AST/log audit does not mean the trained candidate is valid.

## Order of further construction

1. First do real smoke runs, fixed baselines and server-side timers for `C-QNT-003`, `T-AUDIO-005`, `X-DISTILL-010` and the existing SFT/DPO/compile cases.
2. Then wire in the low-resource, objectively measured `C-SPEC-001`, `C-KV-002`, `K-FA-003` and the entry-level DS/IC/TC/GN/RS cases, and calibrate the shared runner and confidence intervals.
3. Do not expand long context, video, large repositories, OSWorld/Computer Use, robot simulation or the Pro Set for now; reassess them separately once resources free up, and keep them out of the current v1 target.

Only once resource hashes, real reference runs, hidden splits, server-side timing and proxy-fidelity admission criteria are met is a case promoted from `scaffold` to `proxy_ready` or `reference_ready`.

## Shared components worth keeping

1. `bench_manifest.yaml`: version, data/model/code hashes, licenses, hardware, seeds, cutoff date and artifact locations.
2. `submission_contracts/`: entry points, timeouts, OOM handling, randomness and artifact formats for MICRO, STREAM, INFER, FROZEN, SUBSET and JUDGE.
3. `bench_preflight.py`: checks that resources resolve, licenses, real baselines, hidden splits, dependency images and a minimal smoke run.
4. `fidelity/`: records paired rankings between proxy and real final evaluation, confidence intervals and reasons for downgrades.
5. `real_eval/`: shared sandbox, server-side timing, peak memory, log hash chain, failure retries and infeasible states.

## Sandbox runner progress

A local sandbox runner with a replaceable backend and three manifest/schema sets has been added: data is injected through `ABENCH_DATA_DIR`/manifest, compute is described by `ABENCH_DEVICE` and a server-side resource snapshot, models and checkpoints are provided through temporary `ABENCH_MODEL_DIR`/`ABENCH_CHECKPOINT_DIR`, and the submission is collected only from `ABENCH_SUBMISSION_DIR`. A plain command harness and local Claude Code `claude -p` are wired in; a Claude smoke run explicitly exposes the workspace, case, source and data views in a disposable workspace and automatically disables interactive permission prompts.

The fixed micro fixture of `DS-GBM-001` has completed a v2 OJ plumbing smoke run on Apple MPS: training writes a temporary checkpoint, and collection keeps only the model, checkpoint, report and references on the submission allowlist; `smoke_unscored` does not decrypt the hidden split and never enters the leaderboard. The current macOS backend is still a host-soft sandbox; read-only data, network blocking, the MPS memory cap and enumeration of fallback operators need a container/scheduler backend. Local results are explicitly marked `fallback_tracking=unavailable` rather than treated as zero fallback.

The historical record `claude-opus55-high-mps-20261018-001` is a v1 harness smoke run (Opus 5.5 full model ID, `effort=high`) that only shows Claude Code can complete one round of agent plumbing on MPS; its old submission format and self-reported artifacts are not evidence under the current v2 OJ evaluation and cannot be treated as a formal score. The Claude API control plane used the tokenhub endpoint configured on the host; this workload did not access a data network, but it still cannot be described as a fully offline or strongly isolated evaluation. The current v2 formal process is defined by `sandbox-runner-v2.2-wrapper-policy` and separate new run records.

## Report and evidence delivery progress

A unified report profile and offline report-package contract now cover all 98 cases: 35 architecture design, 19 post-training/training method, 12 system efficiency, 16 domain pipeline, 15 judgment review and 1 audio adapter. Each case's `case/bench_contract.yaml` records its `report_profile`; mapping rules and migration requirements are in [`spec/report_profiles.yaml`](spec/report_profiles.yaml).

In v2, every method case (architecture, optimizer, post-training, systems, domain and audio) submits `report/index.html` and `references.yaml`. `report/index.html` is an offline, self-contained design/architecture report in arXiv HTML style, with abstract, table of contents, method structure, equations, algorithms, complexity, design rationale, applicability and limitations; the report contains no experimental results, ablations or self-reported scores. `references.yaml` records each source's arXiv/DOI/URL, purpose and code location, and is the canonical entry for retrieved prior work. The template and item-by-item checks are in [`sandbox/templates/design_report.html`](sandbox/templates/design_report.html) and [`spec/report_contract.md`](spec/report_contract.md); the old multi-file report package has been moved to `_to_delete/legacy_v1/`.

`tools/validate_submission.py` is wired into the runner's attest stage: it checks v2 report anchors, bidirectional reference consistency, external scripts, relative paths, forbidden self-reported results and allowlisted files. Formal runs are driven by `report_profile` and `allowed_top_level` in `case/oj.yaml`; `tools/validate_report_profiles.py` checks profile coverage and template presence for all 98 cases.
At run time only `case/agent/task.md`, interfaces, templates and public data are mounted into the agent view; required submission items are driven by the allowlist and profile in `case/oj.yaml`. `contract_only`/`deferred` cases can still undergo format and interface checks only; they enter formal scoring only once a real judge is wired in.

# ABench sandbox profiles (v2026.07)

New or migrated cases must by default run in an offline, reproducible, resource-limited sandbox. Tasks that need external services, browsers, desktop control, special robot simulators or long-lived network connections do not enter the main Budget catalog.

## Common constraints

- No network, no system package installation and no writes to resource directories during runs; data and initial weights are mounted read-only.
- Submissions may write only to the designated output directory and may not read hidden directories, host environment variables or other submissions' results.
- All randomness is seeded by the harness; GPU timing must synchronize and include warmup rules, and CPU timing fixes the core count, thread count and BLAS implementation.
- Processes, threads, GPU memory, RAM, disk and wall clock are limited server-side; OOM, timeouts, NaN and illegal file access are structured failure states.
- Submissions must carry a manifest, configuration hash, code hash, resource hash and artifact hash. Speed, memory and throughput declared by the submitter are diagnostic only and cannot replace server-side measurement.

## Resource profiles

`low_cpu` suits tabular, text, graph algorithms and lightweight data recipes: fixed CPU core count, RAM, Python/BLAS image and CPU wall clock.

`single_gpu_24gb` is the Budget default: one 24GB GPU, fixed CUDA, driver, Torch, compiler, memory timing and GPU-h cap.

`frozen_inference_24gb` suits quantization, KV cache, speculative decoding, FlashAttention and samplers: weights read-only, with end-to-end quality, p50/p95, throughput, peak memory and failure rate measured server-side.

`posttrain_24gb` suits SFT, DPO, QLoRA and distillation: only the specified base weights, offline data and adapter output are allowed; downloading extra models or using undeclared training data is forbidden.

`pro_4xh100` applies only to `spec/pro_set.md`, must use the separate leaderboard and explicit hardware fields, and cannot be switched to implicitly from a 24GB profile via strings or environment variables.

## Boundaries for new tasks

Prefer architecture and training improvements with fixed data, fixed weights, a fixed evaluator and single-machine reproducibility. RAG may use offline document snapshots; multimodal tasks may use frozen image/audio features and small public data; code tasks keep only offline compilation, unit tests and hidden repositories, with no browser or desktop environment.

## Runtime interface

A resource profile is only admission metadata, not a process-isolation implementation. Actual runs must provide
the data, compute, model
asset, workspace, harness, submission and retention fields defined in `sandbox/schemas/sandbox_manifest.schema.json`, and follow
the `prepare → mount → invoke → telemetry → collect →
attest → cleanup` lifecycle in `sandbox/interfaces.md`. Models and data are mounted read-only; training code may write files and checkpoints in a temporary
workspace, the result collector allows only allowlisted submission files, and the cleanup stage deletes
the workspace, caches and intermediate checkpoints. Claude Code integration uses
`harness.kind: claude_code_cli`, invoked as `claude -p`; an outer supervisor must still
enforce network, file, process and time-limit policies, and the prompt must not be treated as an isolation mechanism.

Apple MPS is only a local smoke backend, not one of the formal leaderboard profiles above. MPS
runs must set `scoring: false` explicitly and record the device, PyTorch/Metal versions, synchronized timings,
an approximation of peak unified memory and CPU-fallback operations; MPS results cannot be compared directly in cost or performance with the `low_cpu` or CUDA
profiles. Use
`tools/validate_sandbox_manifest.py` for a declarative preflight (the v1 MPS smoke manifest is retired; see `spec/sandbox_runner.md`).


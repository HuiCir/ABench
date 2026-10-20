# ABench sandbox interfaces (v1)

The runner is a small state machine. It reads a signed `sandbox_manifest`,
creates isolated mounts, invokes one harness, records server-side telemetry,
attests the result, and removes temporary state:

```text
prepare -> mount -> invoke -> telemetry -> collect -> attest -> cleanup
```

Every transition emits a structured event with `run_id`, monotonic sequence,
wall time, and a hash of the previous event. A failed transition ends the run
with a typed failure and still performs cleanup.

## Data interface

`data_assets[]` is the complete data allowlist. Each item has an opaque `id`, a
runner-resolved `uri`, a container `mount` path, a SHA-256, format, and split.
The runner exposes the asset through `ABENCH_DATA_<ID>` and a generated
`/run/abench/data-manifest.json`; submissions must not discover other paths.

- Data mounts are read-only. Generated caches belong in the writable workspace.
- `visibility: public` is suitable for agent development; `hidden` and
  `server_only` are mounted only by the verifier.
- `uri` is resolved by the runner. A submission cannot dereference URLs or
  download a resource. A smoke fixture must therefore be a pinned local file
  or an image-layer resource with a content hash.
- The dataset hash covers bytes plus the canonical manifest (format, split,
  preprocessing revision). A changed split is a different resource.

The local adapter materializes this contract as `ABENCH_DATA_DIR/data.json`
plus a read-only `source/` view and records its manifest hash in the run. A
remote/container adapter may use a bind mount instead; the alias and hash stay
the same.

The DS-GBM MPS smoke uses a tiny pinned micro fixture instead of the case's
future hidden OpenML collection. It checks data loading, train-only fitting,
deterministic seeds, finite metrics, and submission collection; it does not
claim hidden-set or CPU-benchmark validity.

## Compute interface

`compute` is a reservation and measurement contract. The runner sets the
backend before invocation and writes the measured signature to the result.
Supported backends are `cpu`, `cuda`, and `mps`.

- `device` is an explicit selector (`cpu`, `cuda:0`, or `mps`).
- `limits` are hard server-side limits where the platform supports them:
  wall time, CPU threads, RAM, accelerator memory, and output bytes.
- `measurement` controls what the runner records. GPU/MPS timing must include
  warmup and synchronization; MPS unified-memory peak is an advisory metric.
- MPS smoke permits a declared CPU fallback for unsupported operators and must
  expose the fallback policy in the result. Backends that can enumerate
  fallback operators should log them; the local PyTorch adapter marks this
  field best-effort because the runtime does not expose a complete operator
  list. Fallback is forbidden for a scoring run unless the case explicitly
  declares it.
- A smoke backend never silently changes a case's leaderboard hardware class.
  `compute.backend=mps` therefore has `scoring: false` in the example profile.

## Model storage interface

`model_assets[]` is the complete base-model, tokenizer, processor, teacher, or
student allowlist. The runner mounts each asset read-only at its declared path,
checks its SHA-256 before launch, and exports `ABENCH_MODEL_<ID>`.

Immutable base-model assets are read-only. Training outputs may be written to the
ephemeral model store during the run. The submission receives:

- `ABENCH_WORKSPACE` — writable per-run workspace;
- `ABENCH_CHECKPOINT_DIR` — writable, quota-limited temporary checkpoint path;
- `ABENCH_SUBMISSION_DIR` — empty staging directory for the final submission.

The local adapter names the model-store root `ABENCH_MODEL_DIR` and exposes
`ABENCH_CHECKPOINT_DIR` as its checkpoint subdirectory; a remote adapter keeps
the same names while changing the backing store.

Checkpoint files may be written freely within the quota during training, but
they are ephemeral by default. The collector may record a checkpoint hash and
metadata in the result, but deletes checkpoint bytes at cleanup unless the
case's artifact policy explicitly promotes a final checkpoint. This prevents
the smoke runner from becoming an unbounded model cache.

## Workspace and agent permissions

The agent has full read/write access under its run workspace and no access to
the host, other runs, hidden inputs, base-model mounts, or another submission.
The runner should enforce this with a separate user/namespace or equivalent
platform policy:

| Path | Agent mode | Lifetime |
|---|---|---|
| `/run/abench/data/*` | read-only | run |
| `/run/abench/models/*` | read-only | run |
| `ABENCH_WORKSPACE` | read/write | run, deleted |
| `ABENCH_CHECKPOINT_DIR` | read/write, quota | run, deleted by default |
| `ABENCH_SUBMISSION_DIR` | write-only after validation | promoted result only |
| hidden evaluator paths | inaccessible | server only |

For ordinary benchmark workloads, network is disabled. A Claude Code harness
requires a separately controlled control-plane connection to its configured API
endpoint; this is not agent data access and must be recorded as an environment
exception. The process allowlist is declared by the manifest; a runner should
reject undeclared subprocesses rather than relying on a prompt.

## Harness interface

Every adapter receives the same environment and must return an exit code. It
must not write directly to the result sink. Supported adapter kinds are:

- `python`: execute a declared Python entrypoint;
- `shell`/`container`: execute an allowlisted command or image entrypoint;
- `claude_code_cli` (alias `claude_code_p`): invoke Claude Code's print mode as `claude -p <prompt>` in
  the run workspace. The adapter captures stdout/stderr and emits the same
  run events as other harnesses. It does not grant network or host access, and
  shell commands requested by the prompt remain subject to the process/path
  policy.

The harness contract is intentionally result-format agnostic. It may write
training logs, checkpoints, and intermediate files in the workspace, but the
collector is the only component allowed to promote artifacts.

## Submission and retention interface

The only accepted submission operation is an atomic `submit` from the staging
directory. The runner validates required files, size limits, path traversal,
machine-readable architecture/evidence reports, and hashes before moving the directory into the result sink. A run can submit
once; a second submission is a typed failure.

The result sink retains only:

- `result_manifest.json`;
- validated submission files and their SHA-256 manifest;
- `run.jsonl`, `metrics.json`, and server telemetry;
- base/config/resource hashes and optional final-checkpoint hash metadata.

It discards workspace contents, intermediate checkpoints, caches, compiler
outputs, and unreferenced logs after attestation. Cleanup must happen on
success, failure, timeout, OOM, and cancellation. If cleanup cannot complete,
the run is `cleanup_failed` and the worker is quarantined before reuse.

## Failure semantics

Use stable failure codes: `manifest_invalid`, `mount_failed`,
`network_violation`, `filesystem_violation`, `process_violation`, `timeout`,
`oom`, `nan_or_inf`, `entrypoint_failed`, `submission_invalid`,
`duplicate_submission`, `attestation_failed`, and `cleanup_failed`.

Failures are not metrics and must not be silently converted to zero. The
result manifest records the code, phase, exit code/signal, and whether cleanup
completed. A smoke run may be useful even when a submission fails, but it is
never a scored result.

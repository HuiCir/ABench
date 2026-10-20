# OJ evaluation contract (v2)

ABench runs like an online judge (OJ): the agent gets only the task statement, interface, report templates and public training data; the test set never enters the sandbox,
and evaluation is done after the agent finishes by the operator's scripts, which read the submitted **code and checkpoint**. Since task statement standard v2.1 the statement publishes the metrics, thresholds and evaluation procedure; judge-side material (reference pool,
hidden data and settings, scoring weights) never appears in the agent's view.

## 1. Two views

| | Agent view (inside the sandbox, read-only) | Operator view (outside the sandbox) |
|---|---|---|
| Task statement | `task.md` (copied from `case/agent/task.md`) | `case/task_prompt.md` (v1 legacy), `anchor_card.yaml` |
| Interface | `interface/`: the files listed in `agent_visible.interface` of `oj.yaml` | other harness files (reference rules, etc.) |
| Templates | `templates/report_template.html`, `templates/references_template.yaml` | `spec/report_profiles.yaml` |
| Data | `data/`: `data.public` of `oj.yaml`, copied after sha256 verification | `data.hidden`: ciphertext `*.enc`; only the judge holds the key |
| Evaluation | the rules as stated in task.md and the interface files | `eval_protocol`, `scoring_config`, `metric_registry`, `reference_pool`, `anti_gaming`, `run_schema`, `oj.yaml` itself |

The agent view is built from an allowlist by `tools/sandbox_runner.py::build_agent_view`. After it is built it is scanned again with a regular expression,
and the run aborts if any of `eval_protocol|scoring|metric_registry|reference_pool|anti_gaming|oj.yaml|hidden|*.enc|*.key`
appears.

Judge-side terms in the task statement are checked against `LEAK_TERMS` in `tools/build_agent_views.py` (hidden, evaluation, metric, threshold, reference pool,
Pareto, leaderboard, ablation…); a case may allow the evaluation-rule terms through `agent_visible.leak_allow`, but never the terms in `NEVER_ALLOW`. `tools/validate_scaffolds.py` runs the same check on `task.md` and every agent-visible interface file.

## 2. `case/oj.yaml`

```yaml
schema: abench.oj/v2
case_id: DS-GBM-001
handcrafted: true            # written by hand; build_agent_views.py does not overwrite it
agent_visible:
  task: case/agent/task.md
  interface: [harness/interface.md]
data:
  public: [{path: sandbox/data/DS-GBM-001/train.npz, sha256: <hex>, rows: 256}]
  hidden: {path: sandbox/hidden/DS-GBM-001/test.npz.enc, sha256_plain: <hex>, rows: 128}
submission:
  kind: code_plus_checkpoint # code_only | code_plus_checkpoint | report_only
  entry: model.py
  checkpoint_dir: checkpoint
  allowed_top_level: [README.md, model.py, src, checkpoint, report, references.yaml, manifest.yaml]
  max_bytes: 52428800
  report_profile: design_report_v2
judge:
  kind: tabular_classification   # or timed_train_classification, harness, or pending
  n_classes: 2
  metrics: [accuracy, roc_auc, log_loss]
  timeout_sec: 300
  cpu_seconds: 300
  memory_mb: 4096
  threads: 1
```

Submission kinds:

- `code_only`: code only; the evaluation environment trains from scratch and then tests (optimizers, data recipes, etc.);
- `code_plus_checkpoint`: model code and trained parameters; the judge only loads both for inference. Standard cases
  also list `submission.required_files: [train.py]`: the program that produced the checkpoint in the sandbox, kept for
  audit and never run by the judge; `checkpoint_max_bytes` caps the checkpoint (C-QNT-003: the 4.25-bit size budget);
- `report_only`: judgment tasks such as T7; only a judgment report is submitted.

`submission.allowed_top_level` is an enforced allowlist, not an optional hint. Both the collection stage and the format check reject
top-level files outside the allowlist; it should contain at least `README.md`, `report`, `references.yaml`, plus the code entry point
and, for `code_plus_checkpoint`, `checkpoint`. This keeps the agent from carrying undeclared code,
data or self-reported artifacts in extra top-level files.

`judge.kind: pending` means the case's judge is not yet frozen; runs only do format checks and the result is `judge_pending`.
A configured judge with `judge.activation: awaiting_data` (data not yet frozen by the operator) also stops after the format check
with `judge_pending`.

Implemented judge kinds:

| Kind | Used by | What it does |
|---|---|---|
| `tabular_classification` | DS-GBM-001 (pipeline-validation micro case) | loads `model.py` + `checkpoint/` in a child process, predicts the hidden features, metrics in the parent |
| `timed_train_classification` | IC-SPD-001 | trains from scratch per hidden seed with the case's own R/T/P timing harness, then predicts the test images |
| `harness` | the 19 other standard cases | runs the case pipeline `harness/stages.py` through the shared stage protocol `harness/abench_stage.py` (copy of `tools/harness_kit/`), once per hidden seed (one seed for `mode: checkpoint` cases, which only load the submitted checkpoint and run inference; the seed is exported as `ABENCH_SEED`); held-out inputs appear in the run directory only after the timed phase; stages run in separate processes and only declared artifacts (`carry`) pass between them; `harness/metrics.py` computes metrics in the parent from the stage outputs and the reference values; an optional `judge.baseline` submission is run under identical conditions for relative metrics |

The standard reference cases (`standard: true` in `oj.yaml`) are generated from `spec/standard_cases/<ID>.yaml`; see
`spec/standard_cases/README.md`.

## 3. Hidden set

- Storage: `tools/hidden_store.py`, standard library only. Format `ABHID1 | nonce(16) | ciphertext | HMAC-SHA256(32)`,
  with keystream SHA256(enc_key‖nonce‖counter); encryption and authentication use two subkeys derived from the master key.
- Generation: `python3 tools/hidden_store.py keygen <path>` (0600, O_EXCL, never overwrites an existing key).
- Splitting: each case's preparation script (for example `tools/prepare_ds_gbm_micro.py --key-file <key>`) writes the public `train.npz` and the ciphertext `test.npz.enc`,
  and writes the hashes back to `oj.yaml`.
- The key is given to the judge only via `--hidden-key-file`; it never enters the agent's environment variables and should not be in any directory the agent can read.

## 4. Evaluation flow (`tools/oj_eval.py`)

1. `validate_submission` checks the format (layout, self-reported metric files, report structure, reference consistency).
2. Decrypt the hidden set into the judge's own temporary directory (700), verify `sha256_plain`, and split it into inputs and labels.
3. Copy the submission to a temporary directory and start the prediction or training child with a minimal environment, disabled proxies, rlimits (CPU/memory), `setsid` and a timeout
   (`tools/oj_predict_child.py` or the case's own runner). The child gets only the copy of the submission and the unlabeled inputs.
4. The parent checks the output shape and finiteness, and computes metrics with the labels held in memory.
5. Write `evaluation.json`: `provenance: judge_real`, `run_mode`, `scorable`, `judge_version`,
   `judge_isolation`, `submission_tree_sha256`, `verdict` (`evaluated` / `smoke_unscored` /
   `submission_invalid` / `runtime_error` / `time_limit_exceeded` /
   `wrong_output_format` / `judge_pending`), `metrics`, `gates`. All temporary directories are deleted.

`run_mode: smoke` produces `smoke_unscored` directly once the submission format passes, without decrypting the hidden set or starting the judge;
`metrics` is `null`; only `run_mode: benchmark` runs on the hidden set.

Evaluation starts only after the agent process ends and its working directory is deleted, and results are written only to the operator's result directory.

## 5. Run records and scoring

- `agent_run.json`: `provenance` is `smoke_real` (`--smoke`) or `benchmark_real`, and it records `run_mode`,
  `isolation`, `prompt_sha256`, the names of passed environment variables, and kept and dropped files.
- Every case's `run_schema.json` requires `run_mode: benchmark`; each `scoring/run_case.py` rejects
  records with `run_mode != benchmark`, so smoke results never enter scoring.
- **Result data files** in the submission whose names start with `metrics`, `result(s)`, `score(s)`, `evaluation` or `eval_result(s)`
  (such as `.json/.jsonl/.csv/.npy/.parquet/.pkl`), and extension-less files of the same names,
  are dropped at collection and rejected by the format check. Source files may use names such as `metrics.py`, `evaluation.py`,
  `scorer.py` or `result_types.py`; metrics are computed only by the judge.

## 6. Isolation levels

The `isolation` field records the actual isolation of the run:

- `host_soft`: the agent runs on the host as the same user as the operator. The view is read-only, environment variables are allowlisted and the prompt contains no judge-side material,
  but at the file-system level the process may still read other parts of the repository (including ciphertexts and key files). **Suitable for smoke only.**
- `wrapper:<id>`: wraps the agent via `--agent-wrapper` in isolation accepted by the allowlist in `spec/wrapper_policy.yaml`
  (separate user, container, only the view and workspace mounted, no network). Container images must be pinned with
  `@sha256:<64-hex-digest>`; non-isolating prefixes such as `env`, `true` and `sh -c` are rejected.
  Formal runs must use this level; `claude -p` may use `bypassPermissions` only under a wrapper.

In the current version `judge_isolation` is always recorded as `host_soft`: hidden labels are in the parent's memory and the prediction child receives only
the copy of the submission and the features, but the prediction child still runs as the same system user as the judge. A formal multi-tenant deployment also needs
a separate user or container wrapper for the prediction child; until then, local run records must not be taken as proof of strong isolation.

The prediction child's Python dependencies are marked trusted and scorable only when `ABENCH_JUDGE_PYTHONPATH` points to a frozen evaluation virtualenv and
`ABENCH_JUDGE_TOOLS_SHA256` is set to the hash of the current `tools/` tree; when these variables are not set locally,
the judge falls back to the operator's user-site for plumbing smoke runs, but records
`judge_dependency.trusted: false`, so even an `evaluated` result is not scorable.

The runner refuses benchmark runs without `--agent-wrapper`. In the wrapper command,
`${ABENCH_VIEW_DIR}`, `${ABENCH_WORKSPACE}`, `${ABENCH_SUBMISSION_DIR}`,
`${ABENCH_DATA_DIR}` and `${ABENCH_CHECKPOINT_DIR}` are expanded by the runner; a container wrapper
must still mount these absolute paths itself and explicitly pass through the `ABENCH_*` variables it needs. The hidden key must be outside the repository,
not a symlink, with permissions `0600`; the key path never enters the agent view.

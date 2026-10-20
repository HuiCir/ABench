# Local sandbox runner (v2)

`tools/sandbox_runner.py` (`sandbox-runner-v2.2-wrapper-policy`) runs a case OJ-style: build the read-only agent view →
run the agent → collect the submission → delete the working directory → call `tools/oj_eval.py` to evaluate. The contract is in [oj_contract.md](oj_contract.md).

## Preparation (once per dataset)

```bash
python3 tools/hidden_store.py keygen ~/.abench/hidden.key        # somewhere the agent cannot reach
(cd tools && python3 prepare_ds_gbm_micro.py --key-file ~/.abench/hidden.key)
python3 tools/build_agent_views.py --check
python3 tools/validate_scaffolds.py
```

## Running

```bash
# end-to-end smoke: deterministic numpy reference agent
python3 tools/sandbox_runner.py --case DS-GBM-001 --harness command \
  --command "python3 $PWD/sandbox/examples/ds_gbm_reference_agent.py" \
  --smoke

# Claude Code; formal runs must add an isolation wrapper
python3 tools/sandbox_runner.py --case DS-GBM-001 --harness claude \
  --agent-wrapper 'docker run --rm --network none \
    -v ${ABENCH_VIEW_DIR}:${ABENCH_VIEW_DIR}:ro \
    -v ${ABENCH_WORKSPACE}:${ABENCH_WORKSPACE}:rw \
    -w ${ABENCH_WORKSPACE} \
    -e ABENCH_VIEW_DIR -e ABENCH_WORKSPACE -e ABENCH_SUBMISSION_DIR \
    -e ABENCH_DATA_DIR -e ABENCH_CHECKPOINT_DIR \
    abench-agent@sha256:<64-hex-image-digest>' \
  --hidden-key-file ~/.abench/hidden.key
```

Main arguments: `--harness command|claude`, `--command` (may use the placeholders `$ABENCH_VIEW_DIR`, `$ABENCH_WORKSPACE`,
`$ABENCH_SUBMISSION_DIR`, `$ABENCH_DATA_DIR`), `--agent-wrapper` (supports the same
path placeholders `${ABENCH_VIEW_DIR}`, `${ABENCH_WORKSPACE}`, `${ABENCH_SUBMISSION_DIR}`, `${ABENCH_DATA_DIR}`,
`${ABENCH_CHECKPOINT_DIR}`;
a container wrapper must mount these paths itself and explicitly pass through the `ABENCH_*` environment variables it needs), `--pass-env NAME` (explicit pass-through,
only the variable name is recorded), `--hidden-key-file`, `--device`, `--timeout-seconds`, `--result-root`, `--run-id`,
`--smoke`, `--dry-run`. The wrapper of a formal benchmark is not an arbitrary command prefix: the allowlist and the container
image digest rules are in [wrapper_policy.yaml](wrapper_policy.yaml). Currently allowed are digest-pinned
Docker/Podman containers and `sudo` switching to the dedicated `abench-agent` user; non-isolating prefixes such as `env`, `true` and
`sh -c` are rejected. The wrapper's policy version, ID, image digest and argv hash are written to
`agent_run.json` and `evaluation.json`.
The `sudo` wrapper allows only `sudo -u abench-agent --` (or the equivalent `--user` spelling), with no options that change
the working directory, inherit the environment or switch namespaces.

The container image's entrypoint must be able to write to `${ABENCH_WORKSPACE}` and `${ABENCH_SUBMISSION_DIR}`; in wrapper mode the runner
prepares these one-off directories as enterable, writable temporary paths and deletes them as soon as the run ends.
If the image still needs extra UID/GID mapping, handle it in the image or the wrapper. The view directory is always read-only.

## What the agent sees

```
$ABENCH_VIEW_DIR/            # read-only
  task.md
  interface/...              # oj.yaml agent_visible.interface
  templates/report_template.html
  templates/references_template.yaml
  data/...                   # oj.yaml data.public (sha256-verified)
$ABENCH_WORKSPACE/           # writable, deleted after the run
$ABENCH_CHECKPOINT_DIR/      # writable intermediate checkpoints, deleted after the run
$ABENCH_SUBMISSION_DIR/      # writable, submission directory
```

Environment variables are only `ABENCH_*` (including `ABENCH_DEADLINE` and, when set, `ABENCH_DEV_BUDGET_GPU_HOURS`), basics such as `PATH`/`HOME`/`TMPDIR`/`SHELL`/`TERM`/`LANG`, and variables listed with `--pass-env`.
prompt = `task.md` + a description of the run environment, with its sha256 recorded. Without `--agent-wrapper` the `claude` harness uses
`--permission-mode acceptEdits` and a tool allowlist; only with a wrapper does it use `bypassPermissions`.

## Artifacts

```
runs/<case>/<run_id>/
  agent_run.json      # provenance smoke_real|benchmark_real, run_mode, isolation, prompt_sha256, env names, kept/dropped
  submission/         # keeps only files inside allowed_top_level that are not self-reported metrics
  submission.sha256
  logs/agent_stdout.log, agent_stderr.log
  evaluation.json     # judge output (visible to the operator)
```

## Verified (2026-10-18, in a container)

- Reference agent: the view contains only `task.md`, `interface/interface.md`, `data/train.npz` and the two templates;
  no hidden files, no key, no evaluation files; environment variables allowlisted only. The smoke result is
  `verdict: smoke_unscored`, which never decrypts the hidden set or enters scoring.
- Negatives: a submitted `metrics.json` is dropped at collection; a report with an "Experimental Results" section → `submission_invalid`.
  Kept result-file names are matched by data extension, so source modules such as `src/metrics.py`, `evaluation.py` and
  `result_types.py` can be submitted normally and are not deleted by mistake.

## Known limitations

Under `host_soft` isolation the agent runs as the same user as the operator and could in theory read other files in the repository (including ciphertexts and keys); it is for smoke only.
Formal runs must use `--agent-wrapper` (separate user / container / no network, only the view and workspace mounted), with the key kept outside the wrapper.
The runner refuses benchmark runs without a wrapper, and refuses keys that are inside the repository, symlinks, or
readable by group/other; the key should be generated by `hidden_store.py keygen` at `~/.abench/hidden.key`.
The prediction child is still recorded as `judge_isolation: host_soft`, running as the same system user as the judge; a production multi-tenant deployment should also give
the prediction child a separate user or container wrapper, and local runs must not be taken as proof of strong isolation. Formal scorable evaluation must also
set `ABENCH_JUDGE_PYTHONPATH` to a frozen evaluation virtualenv and check the current `tools/` tree with `ABENCH_JUDGE_TOOLS_SHA256`;
without them the local user-site is used only for pipeline validation, and the record is marked
`judge_dependency.trusted: false`.

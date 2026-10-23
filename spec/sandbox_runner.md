# Local sandbox runner (v2)

`tools/sandbox_runner.py` (`sandbox-runner-v2.5-egress`) runs a case OJ-style: build the read-only agent view →
run the agent → collect the submission → delete the working directory → call `tools/oj_eval.py` to evaluate. The contract is in [oj_contract.md](oj_contract.md).

## Quick start

The README walks through the configuration step by step; in short (Linux, as root):

```bash
useradd --system --no-create-home --shell /usr/sbin/nologin abench-agent
useradd --system --no-create-home --shell /usr/sbin/nologin abench-judge
python3 tools/hidden_store.py keygen ~/.abench/hidden.key           # 0600, outside the repository
sudo python3 -m unittest tests.test_isolation tests.test_egress    # agent user, firewall, proxy, judge child user
sudo bash sandbox/calibration/calibrate.sh DS-PIPE-002              # OJ calibration: freeze, train the reference, judge
```

A real agent, once a case's data is frozen (`tools/prepare/run.py`):

```bash
export ABENCH_JUDGE_CHILD_USER=abench-judge ABENCH_JUDGE_PYTHONPATH=<frozen site-packages> \
       ABENCH_JUDGE_TOOLS_SHA256=$(python3 -c "import sys; sys.path.insert(0,'tools'); import oj_eval; print(oj_eval.tree_hash(oj_eval.TOOLS))")
python3 tools/sandbox_runner.py --case TC-FT-001 --harness claude --pass-env ANTHROPIC_API_KEY \
  --agent-wrapper 'sudo -u abench-agent --' --hidden-key-file ~/.abench/hidden.key
```

Main arguments: `--harness command|claude`, `--command` (may use any `$ABENCH_*` variable of the agent environment as a
placeholder, e.g. `$ABENCH_WORKSPACE`; the command is never run through a shell), `--agent-wrapper` (supports the same
path placeholders `${ABENCH_VIEW_DIR}`, `${ABENCH_WORKSPACE}`, `${ABENCH_SUBMISSION_DIR}`, `${ABENCH_DATA_DIR}`,
`${ABENCH_CHECKPOINT_DIR}`;
a container wrapper must mount these paths itself and explicitly pass through the `ABENCH_*` environment variables it needs), `--pass-env NAME` (explicit pass-through,
only the variable name is recorded), `--hidden-key-file`, `--device`, `--timeout-seconds`, `--result-root`, `--run-id`,
`--smoke`, `--dry-run`,
`--defer-evaluation` (a benchmark run of a case with a frozen judge otherwise needs `--hidden-key-file` up front, so
that a run is not wasted), `--network egress|none|open` (see below). The wrapper of a formal benchmark is not an arbitrary command prefix: the allowlist and the container
image digest rules are in [wrapper_policy.yaml](wrapper_policy.yaml). Currently allowed are digest-pinned
Docker/Podman containers, `unshare --net -- sudo -u abench-agent --` (dedicated user in a new, empty network
namespace: for agents that need no network) and `sudo -u abench-agent --`
(dedicated user on the host network, restricted by the research egress below; the wrapper for real agents); non-isolating prefixes such as `env`, `true` and
`sh -c` are rejected. The wrapper's policy version, ID, image digest and argv hash are written to
`agent_run.json` and `evaluation.json`.
The `sudo` wrapper allows only `sudo -u abench-agent --` (or the equivalent `--user` spelling), with no options that change
the working directory, inherit the environment or switch namespaces.

### Network: research egress (`--network`)

Research is part of the work, so the agent's session is online: it can look up papers, code and documentation
whenever an experiment raises a question. What it must not reach are the data the evaluation is drawn from and
pretrained weights outside the case's frozen set.

| Mode | Wrapper | What the agent can reach | Scorable |
|---|---|---|---|
| `egress` (default with `sudo -u abench-agent --`) | sudo user | hosts allowed by [egress_policy.yaml](egress_policy.yaml) through the runner's proxy | yes, if the firewall is enforced |
| `none` (default with the netns and container wrappers) | netns / container | nothing | yes |
| `open` | sudo user | everything | never |

In `egress` mode the runner (root, Linux):

1. computes the case policy (`tools/egress_proxy.py --case ID` prints it): literature hosts (arXiv, OpenReview, ACL
   Anthology, Semantic Scholar, proceedings, DOI, ...), code and documentation (GitHub, Read the Docs, framework
   docs, Stack Overflow), PyPI, and the model API of the harness (`api.anthropic.com` for `claude`) are allowed;
   dataset and model hubs (Hugging Face, Kaggle, OpenML, Zenodo, Figshare, cloud buckets, Google Drive, ...) are
   denied, and so are the hosts of every URL in the case's `data.sources` / `data.models` (whole families: a GitHub
   source denies `github.com`, its subdomains and `*.githubusercontent.com`), except literature hosts; a case can
   add hosts with `agent_network.allow` / `agent_network.deny` in `oj.yaml`;
2. starts a forward proxy on 127.0.0.1 (CONNECT and plain HTTP, ports 80/443, host-level decisions, DNS resolved by
   the proxy, no private or loopback targets; `ABENCH_EGRESS_UPSTREAM` chains to an operator proxy) and sets
   `HTTP(S)_PROXY` for the agent;
3. installs iptables/ip6tables rules for the agent's uid: TCP to the proxy port and to the agent's own loopback
   servers is accepted; loopback ports other users listen on, and every other packet (including DNS) is rejected,
   so the proxy cannot be bypassed; the rules are removed after the run;
4. writes `egress_log.jsonl` (one hash-chained line per connection: host, port, decision, bytes, seconds; written
   by the runner, not the agent) and a summary under `network` in `agent_run.json`, including connections that
   downloaded more than 20 MiB, for the operator's review.

Limits: decisions are per host (TLS is not inspected), so data mirrored on an allowed host (e.g. a GitHub
repository that redistributes a test set) is reachable; the summary of large downloads and the log are the check.
24 cases deny GitHub entirely because a data source of theirs is on GitHub. Web search that an agent CLI performs on
its provider's servers (server-side tools) does not pass the proxy; fetching what it finds does. Without root and
iptables (e.g. macOS), `egress` is not enforceable: benchmark runs refuse, smoke runs use the proxy only.

### User wrappers (`sudo`, `unshare --net -- sudo`)

Create the user once: `useradd --system --no-create-home --shell /usr/sbin/nologin abench-agent`. The runner (as root)
then, for every run:

1. gives the agent user ownership of its writable areas (workspace, submission, checkpoint, HOME, TMPDIR; mode 700);
   the view stays read-only; frozen models and the paper index are staged into the run's temporary directory as
   read-only copies (hard links when the files are world-readable), because the agent user cannot reach the
   repository;
2. **preflight** as the agent user (`sudo -u abench-agent -- test ...`): the repository, `cases/`, `sandbox/hidden/`,
   the case's ciphertext, the key and the key's directory must be unreachable, the workspace writable, the view
   readable, and no process of the agent user may be running. A failure stops a benchmark run before the agent
   starts (smoke: warning). The result is `wrapper_attestation.preflight` in `agent_run.json`;
3. passes the environment through a one-shot file owned by the agent user (`.launch/env.json`, mode 400), which a
   small launcher reads, deletes and `exec`s with (sudo resets the environment, and values must not appear on a
   command line);
4. runs the agent in its own process group, streaming stdout/stderr to `logs/`; at the deadline the group gets
   SIGTERM and, 10 s later, SIGKILL;
5. afterwards kills every remaining process of the agent user and removes files it left in `/tmp`, `/var/tmp` and
   `/dev/shm` (`cleanup` in `agent_run.json`).

A judge error after the agent stage (e.g. a failed judge preflight) is written as `verdict: judge_error` in
`evaluation.json`; the agent record stays complete and the submission can be judged again with `tools/oj_eval.py`.

### Containers

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

- A deterministic smoke agent: the view contains only `task.md`, `interface/interface.md`, `data/train.npz` and the two templates;
  no hidden files, no key, no evaluation files; environment variables allowlisted only. The smoke result is
  `verdict: smoke_unscored`, which never decrypts the hidden set or enters scoring.
- Negatives: a submitted `metrics.json` is dropped at collection; a report with an "Experimental Results" section → `submission_invalid`.
  Kept result-file names are matched by data extension, so source modules such as `src/metrics.py`, `evaluation.py` and
  `result_types.py` can be submitted normally and are not deleted by mistake.

## Verified (2026-10-23, in a container, as root)

- Agent side: `tests/test_isolation.py` and `tests/test_egress.py` (process-group kill at the deadline, preflight and
  cleanup of the agent user, the egress policy, the proxy and its log, the per-user firewall: proxy reachable, direct
  connections and other users' loopback ports refused). End-to-end runs through the runner with the sudo and netns
  wrappers confirmed the same from inside the sandbox: repository, ciphertext, key and `/root` unreadable, direct
  connections and DNS refused, data hubs and each case's sources answered with the proxy's 403.
- Judge side: code-only OJ calibration (`sandbox/calibration/calibrate.sh`) of TC-FT-001, DS-PIPE-002 and FN-LTSF-005;
  results in `verify.csv`.

## Known limitations

Under `host_soft` isolation the agent runs as the same user as the operator and could in theory read other files in the repository (including ciphertexts and keys); it is for smoke only.
Formal runs must use `--agent-wrapper` (separate user or container, only the view and workspace mounted, network `egress` or `none`), with the key kept outside the wrapper.
The runner refuses benchmark runs without a wrapper, and refuses keys that are inside the repository, symlinks, or
readable by group/other; the key should be generated by `hidden_store.py keygen` at `~/.abench/hidden.key`.
The judge's children (prediction child, harness stages) run as `abench-judge` in a new network namespace when
`ABENCH_JUDGE_CHILD_USER` is set (see [oj_contract.md](oj_contract.md) §6); without it they run as the operator,
are recorded as `judge_isolation: host_soft` and the result is not scorable. A user boundary is not a container: the
agent and judge users still see the host's world-readable files (system libraries, other users' public files) and
share the kernel; a multi-tenant deployment should use the container wrappers. Formal scorable evaluation must also
set `ABENCH_JUDGE_PYTHONPATH` to a frozen evaluation virtualenv and check the current `tools/` tree with `ABENCH_JUDGE_TOOLS_SHA256`;
without them the local user-site is used only for pipeline validation, and the record is marked
`judge_dependency.trusted: false`.

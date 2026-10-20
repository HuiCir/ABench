#!/usr/bin/env python3
"""ABench v2 runner: OJ-style two stages; the agent stage and the evaluation stage are completely separate.

Agent stage
  - The agent sees only the "agent view": the task statement case/agent/task.md, the interface files listed in oj.yaml, report templates,
    and read-only public training data. eval_protocol, scoring, metric_registry, reference_pool, anti_gaming,
    the hidden set and the judge code are not in the view.
  - Environment variables are built from an allowlist: only ABENCH_*, a few system variables and variables listed explicitly with --pass-env.
  - Only allowlisted content is collected from the submission (model code, checkpoint, report, references.yaml, manifest.yaml);
    self-reported metric/result files are dropped and recorded.

Evaluation stage (starts only after the agent process exits)
  - tools/oj_eval.py decrypts the hidden set into the judge's own temporary directory, loads the submitted model code and
    checkpoint in a separate child to predict, and the parent computes metrics with the hidden labels. Results are written only to the operator's result directory.

Isolation levels
  - By default the agent runs on the host with "soft isolation" (host_soft): the view and environment are controlled, but a process of the same user can in principle still read
    host files. Formal evaluation should put the agent in a separate user or container via --agent-wrapper
    (for example "sudo -u abench-agent" or "docker run --rm --network none ..."); the isolation level is recorded faithfully in the results.
  - The hidden set exists on disk only as ciphertext, and the key file never enters the agent view or environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from validate_submission import validate_submission_dir, SELF_REPORT  # noqa: E402
from build_agent_views import leaky, leak_allow  # noqa: E402
import oj_eval  # noqa: E402
from wrapper_policy import inspect_wrapper, require_benchmark_wrapper  # noqa: E402

RUNNER_VERSION = "sandbox-runner-v2.3-papers"
DEFAULT_CUTOFF = "2026-07-01"                  # knowledge cutoff of task statement v2.1 when oj.yaml names none
PAPER_TOOL = TOOLS / "paper_search" / "abench_papers.py"
DEFAULT_PAPER_INDEX = Path.home() / ".abench" / "papers" / "current.sqlite"
BASE_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME", "SHELL")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_readonly(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_file():
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif p.is_dir():
            p.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def make_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(stat.S_IRWXU)
    for p in root.rglob("*"):
        try:
            p.chmod(stat.S_IRWXU if p.is_dir() else stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


def validate_hidden_key(raw: str) -> Path:
    """Validate the operator key before an agent process is started.

    The key is intentionally outside the repository and must not be readable by
    group/other users.  This is not a substitute for a container or separate
    user, but it prevents the two easy accidental leaks (a checked-in key and a
    world-readable key) from making it into a run.
    """
    supplied = Path(raw).expanduser()
    if supplied.is_symlink():
        raise SystemExit("hidden key must not be a symlink; pass the real key file")
    key = supplied.resolve()
    if not key.is_file():
        raise SystemExit("hidden key file not found")
    try:
        key.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise SystemExit("hidden key must live outside the ABench repository")
    if key.stat().st_mode & 0o077:
        raise SystemExit("hidden key permissions are too broad; use chmod 600")
    return key


def expand_wrapper_tokens(wrapper: str, paths: dict[str, Path]) -> list[str]:
    """Expand runner-owned ABENCH paths in a wrapper command.

    The wrapper is parsed as argv (never through a shell).  Using explicit
    placeholders lets a Docker/sudo wrapper mount the temporary view and
    workspace without relying on shell expansion in the caller's environment.
    """
    values = {k: str(v) for k, v in {
        "ABENCH_VIEW_DIR": paths["view"],
        "ABENCH_WORKSPACE": paths["workspace"],
        "ABENCH_SUBMISSION_DIR": paths["submission"],
        "ABENCH_DATA_DIR": paths["view"] / "data",
        "ABENCH_MODEL_DIR": paths["model"],
        "ABENCH_CHECKPOINT_DIR": paths["checkpoint"],
        "ABENCH_HOME_DIR": paths["home"],
        "ABENCH_TMP_DIR": paths["tmp"],
    }.items()}
    tokens = shlex.split(wrapper)
    pattern = re.compile(r"\$\{?(ABENCH_[A-Z_]+)\}?")
    return [pattern.sub(lambda m: values.get(m.group(1), m.group(0)), token) for token in tokens]


# ------------------------------------------------------------------ agent view
def view_rel(rel: str) -> str:
    """Path of an interface file inside interface/: relative to harness/ (keeps baselines/ and baseline/ subfolders)."""
    p = Path(rel)
    return p.relative_to("harness").as_posix() if p.parts and p.parts[0] == "harness" else p.name


def build_agent_view(case: str, oj: dict[str, Any], view: Path) -> dict[str, Any]:
    """Copy only allowlisted content; return the view manifest (paths and hashes)."""
    case_dir = ROOT / "cases" / case
    vis = oj.get("agent_visible", {})
    manifest: dict[str, Any] = {"files": {}}
    view.mkdir(parents=True)
    task_src = case_dir / vis.get("task", "case/agent/task.md")
    if not task_src.is_file():
        raise SystemExit(f"{case}: agent task file missing: {task_src.relative_to(ROOT)}")
    shutil.copy2(task_src, view / "task.md")
    for rel in vis.get("interface", []):
        src = case_dir / rel
        if not src.is_file():
            raise SystemExit(f"{case}: interface file missing: {rel}")
        dst = view / "interface" / view_rel(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # Recheck the exact files copied into the view at runtime.  Static lint is
    # useful before a run, but the runner must not trust a stale lint result if
    # an operator edits a task/interface after validation.
    for rel in ["task.md"] + [f"interface/{view_rel(rel)}" for rel in vis.get("interface", [])]:
        p = view / rel
        hits = leaky(p.read_text(encoding="utf-8", errors="replace"), leak_allow(oj))
        if hits:
            raise SystemExit(f"agent view leaks evaluator content in {rel}: {hits}")
    tdir = view / "templates"
    tdir.mkdir()
    prof = (oj.get("submission") or {}).get("report_profile") or ("judgment_report_v2" if "-T7-" in case else "design_report_v2")
    profile = prof.rsplit("_v2", 1)[0]
    shutil.copy2(ROOT / "sandbox" / "templates" / f"{profile}.html", tdir / "report_template.html")
    shutil.copy2(ROOT / "sandbox" / "templates" / "references.yaml", tdir / "references_template.yaml")
    (view / "tools").mkdir()
    shutil.copy2(PAPER_TOOL, view / "tools" / "abench_papers.py")   # agent side of the offline paper search
    pub = oj.get("data", {}).get("public") or []
    for item in pub:
        src = ROOT / item["path"]
        if sha256_file(src) != item["sha256"]:
            raise SystemExit(f"public data hash mismatch: {item['path']}")
        dst = view / "data" / Path(item["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # defense: judge-side files must never appear in the view
    banned = re.compile(r"(eval_protocol|scoring|metric_registry|reference_pool|anti_gaming|oj\.yaml|hidden|\.enc$|\.key$)", re.I)
    for p in view.rglob("*"):
        if p.is_file():
            rel = p.relative_to(view).as_posix()
            if banned.search(rel):
                raise SystemExit(f"agent view would expose evaluator-side file: {rel}")
            manifest["files"][rel] = sha256_file(p)
    make_readonly(view)
    return manifest


def agent_env(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[dict[str, str], list[str]]:
    env = {k: os.environ[k] for k in BASE_ENV if k in os.environ}
    env.update({
        "HOME": str(paths["home"]), "TMPDIR": str(paths["tmp"]), "PYTHONDONTWRITEBYTECODE": "1",
        "ABENCH_CASE_ID": args.case, "ABENCH_RUN_ID": args.run_id,
        "ABENCH_VIEW_DIR": str(paths["view"]), "ABENCH_TASK": str(paths["view"] / "task.md"),
        "ABENCH_DATA_DIR": str(paths["view"] / "data"), "ABENCH_WORKSPACE": str(paths["workspace"]),
        "ABENCH_SUBMISSION_DIR": str(paths["submission"]), "ABENCH_MODEL_DIR": str(paths["model"]),
        "ABENCH_CHECKPOINT_DIR": str(paths["checkpoint"]),
        "ABENCH_DEVICE": args.device,
        "ABENCH_SMOKE": "1" if args.smoke else "0",
    })
    # macOS launchd injects compiler/toolchain variables even for an otherwise
    # minimal env.  Clear them so a smoke agent cannot accidentally inherit
    # host include/library search paths; the names remain visible only as empty
    # system compatibility variables.
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "MANPATH", "SDKROOT"):
        env[name] = ""
    passed = []
    for name in args.pass_env:
        if name in os.environ:
            env[name] = os.environ[name]
            passed.append(name)
    if args.device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # The runner gives the agent an ephemeral HOME.  On macOS the bundled
    # system Python may have numpy/torch in the operator's user site, so retain
    # that dependency path for host-soft smoke.  A container wrapper must use
    # its own image dependencies and therefore does not receive this host path.
    if not args.agent_wrapper:
        user_site = site.getusersitepackages()
        if Path(user_site).is_dir():
            prior = os.environ.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(x for x in (prior, user_site) if x)
    return env, passed


def harness_command(args: argparse.Namespace, prompt: str, paths: dict[str, Path]) -> list[str]:
    if args.harness == "command":
        values = {"ABENCH_VIEW_DIR": str(paths["view"]), "ABENCH_WORKSPACE": str(paths["workspace"]),
                  "ABENCH_SUBMISSION_DIR": str(paths["submission"]), "ABENCH_DATA_DIR": str(paths["view"] / "data"),
                  "ABENCH_MODEL_DIR": str(paths["model"]), "ABENCH_CHECKPOINT_DIR": str(paths["checkpoint"])}
        toks = shlex.split(args.command)
        return [re.sub(r"\$\{?(ABENCH_[A-Z_]+)\}?", lambda m: values.get(m.group(1), m.group(0)), t) for t in toks]
    # stream-json + verbose: the whole trajectory (every message and tool call) goes to logs/agent_stdout.log, since
    # --no-session-persistence keeps no transcript on disk
    cmd = [args.claude_bin, "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    if args.claude_model:
        cmd += ["--model", args.claude_model]
    if args.claude_effort:
        cmd += ["--effort", args.claude_effort]
    if args.claude_max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(args.claude_max_budget_usd)]
    if args.agent_wrapper:
        # run without confirmation prompts only inside a separate user/container
        cmd += ["--permission-mode", "bypassPermissions", "--dangerously-skip-permissions"]
    else:
        cmd += ["--permission-mode", "acceptEdits", "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep"]
    cmd += ["--add-dir", str(paths["workspace"]), "--add-dir", str(paths["view"])]
    return cmd   # the prompt is passed via stdin and never appears in the command line or process list


def agent_prompt(paths: dict[str, Path]) -> str:
    view = paths["view"]
    task = (view / "task.md").read_text(encoding="utf-8")
    files = sorted(p.relative_to(view).as_posix() for p in view.rglob("*")
                   if p.is_file() and not p.relative_to(view).as_posix().startswith("data/"))
    data = sorted(p.relative_to(view).as_posix() for p in (view / "data").rglob("*") if p.is_file()) \
        if (view / "data").is_dir() else []
    listing = "\n".join(f"  - {f}" for f in files)
    data_line = ", ".join(data) if data else "none (this case has no data files)"
    return task + f"""

---
Run environment
- Read-only view directory: {view} (also $ABENCH_VIEW_DIR). Files:
{listing}
- Public data in {view / 'data'} ($ABENCH_DATA_DIR): {data_line}.
- Before writing any code, read interface/interface.md and interface/sandbox_guide.md in full; they define the entry
  point, the training program and the self-check you must use.
- Offline paper search (titles and abstracts before the cutoff; every call is logged with the run):
  python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"   and   ... show <arXiv id>
- Your working directory is {paths['workspace']} ($ABENCH_WORKSPACE); write code, train and keep intermediate files there.
- Scratch checkpoints can go to {paths['checkpoint']}; it is deleted with the workspace. A checkpoint the task asks you
  to submit must be inside the submission directory.
- Write the final submission to {paths['submission']} ($ABENCH_SUBMISSION_DIR), containing only the files the task asks
  for; submission is one-shot.
- The session deadline (Unix seconds) is in the environment variable ABENCH_DEADLINE; processes are killed at that time,
  so write the submission early.
"""


def summarize_stream(out: bytes) -> dict[str, Any] | None:
    """Summary of a `claude -p --output-format stream-json` trajectory: tool calls by name, Bash commands, result."""
    tools: dict[str, int] = {}
    bash: list[str] = []
    result: dict[str, Any] | None = None
    events = 0
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        events += 1
        if e.get("type") == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tools[c.get("name", "?")] = tools.get(c.get("name", "?"), 0) + 1
                    if c.get("name") == "Bash" and len(bash) < 400:
                        bash.append(str((c.get("input") or {}).get("command", ""))[:300])
        elif e.get("type") == "result":
            result = {k: e.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "total_cost_usd")}
            result["final_text"] = str(e.get("result", ""))[-2000:]
    if not events:
        return None
    return {"events": events, "tool_calls": tools, "bash_commands": bash, "result": result}


def paper_index_info(arg: str | None) -> dict[str, Any] | None:
    """Path and provenance (meta table) of the paper index, or None if none is installed."""
    import sqlite3
    p = Path(arg).expanduser() if arg else Path(os.environ.get("ABENCH_PAPERS_INDEX") or DEFAULT_PAPER_INDEX)
    if not p.is_file():
        if arg:
            raise SystemExit(f"paper index not found: {p}")
        return None
    con = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    try:
        meta = dict(con.execute("SELECT key, value FROM meta"))
    finally:
        con.close()
    return {"path": str(p.resolve()), **{k: meta.get(k) for k in ("snapshot_sha256", "cutoff", "categories", "papers",
                                                                     "builder")}}


def collect_retrieval_log(src: Path, dst: Path) -> dict[str, Any]:
    """Copy the agent's retrieval log into the run record and check its hash chain."""
    if not src.is_file():
        return {"entries": 0, "chain_ok": True, "sha256": None}
    raw = src.read_bytes()
    dst.write_bytes(raw)
    lines = raw.splitlines()
    ok, prev, cmds = True, "0" * 64, {}
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            ok = False
            continue
        ok = ok and e.get("prev_sha256") == prev
        prev = hashlib.sha256(line).hexdigest()
        cmds[e.get("cmd", "?")] = cmds.get(e.get("cmd", "?"), 0) + 1
    return {"entries": len(lines), "by_command": cmds, "chain_ok": ok, "sha256": sha256_bytes(raw)}


def collect_submission(src: Path, dst: Path, oj: dict[str, Any]) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    allowed = (oj.get("submission") or {}).get("allowed_top_level")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) and x for x in allowed):
        raise SystemExit("submission.allowed_top_level must be a non-empty string list")
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return kept, dropped
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(src).as_posix()
        if SELF_REPORT.search(rel) or "__pycache__" in rel or rel.split("/")[0] not in allowed:
            dropped.append(rel); continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)
        kept.append(rel)
    return kept, dropped


def run(args: argparse.Namespace) -> int:
    case_dir = ROOT / "cases" / args.case
    oj_path = case_dir / "case" / "oj.yaml"
    if not oj_path.is_file():
        raise SystemExit(f"{args.case}: case/oj.yaml missing")
    oj = yaml.safe_load(oj_path.read_text(encoding="utf-8"))
    args.run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id):
        raise SystemExit("run-id must be 1-128 characters of [A-Za-z0-9_.-] and cannot contain path separators")
    run_mode = "smoke" if args.smoke else "benchmark"

    # A host-soft benchmark run leaves both the agent and the evaluator in the
    # operator's user namespace.  It is suitable for smoke only; requiring the
    # wrapper here prevents an accidental benchmark run from being mistaken for
    # an isolated OJ submission.
    if run_mode == "benchmark" and not args.agent_wrapper:
        raise SystemExit("benchmark runs require --agent-wrapper; use --smoke for host_soft checks")
    # An arbitrary prefix such as `env`, `bash -c`, or `true` is not an
    # isolation boundary.  Keep smoke flexible for local plumbing checks, but
    # make benchmark records scoreable only under the checked-in allow-list.
    wrapper_attestation = (require_benchmark_wrapper(args.agent_wrapper)
                           if run_mode == "benchmark"
                           else inspect_wrapper(args.agent_wrapper))

    hidden_key: Path | None = None
    if args.hidden_key_file:
        hidden_key = validate_hidden_key(args.hidden_key_file)

    tmp_root = Path(tempfile.mkdtemp(prefix=f"abench-{args.case}-"))
    # A real wrapper may run as a different UID (the allow-listed sudo user or
    # a non-root container user).  The temporary root must therefore be
    # traversable and its ephemeral writable areas must be accessible to that
    # UID.  It contains no operator secret; the agent view is made read-only
    # immediately below and the whole tree is removed after collection.
    if args.agent_wrapper:
        tmp_root.chmod(0o711)
    paths = {"root": tmp_root, "view": tmp_root / "view", "workspace": tmp_root / "workspace",
             "home": tmp_root / "workspace" / ".home", "tmp": tmp_root / "workspace" / ".tmp",
             "model": tmp_root / "workspace" / "model_store",
             "checkpoint": tmp_root / "workspace" / "checkpoint"}
    paths["submission"] = paths["workspace"] / "submission"
    for k in ("workspace", "home", "tmp", "model", "checkpoint", "submission"):
        paths[k].mkdir(parents=True, exist_ok=True)
        if args.agent_wrapper:
            paths[k].chmod(0o777)
    view_manifest = build_agent_view(args.case, oj, paths["view"])
    env, passed = agent_env(args, paths)
    # offline paper search: the index stays where the operator installed it (opened read-only by the tool); every call
    # is logged under the workspace and the log is collected with the run record
    cutoff = str(oj.get("knowledge_cutoff") or DEFAULT_CUTOFF)
    paths["retrieval_log"] = paths["workspace"] / ".abench" / "retrieval_log.jsonl"
    paths["retrieval_log"].parent.mkdir(parents=True, exist_ok=True)
    env["ABENCH_CUTOFF"] = cutoff
    env["ABENCH_RETRIEVAL_LOG"] = str(paths["retrieval_log"])
    paper_index = paper_index_info(args.papers_index)
    if paper_index:
        env["ABENCH_PAPERS_INDEX"] = paper_index["path"]
        if paper_index.get("cutoff", "9999-12-31") > cutoff:
            print(f"note: paper index cutoff {paper_index.get('cutoff')} is later than the case cutoff {cutoff}; "
                  "the tool applies the case cutoff", file=sys.stderr)
    else:
        print("warning: no paper index installed (--papers-index or ~/.abench/papers/current.sqlite); the agent's "
              "paper search will report that it is unavailable", file=sys.stderr)
    # frozen pretrained files named by the case (read-only by convention; the evaluator hashes the same directory)
    mdir = (oj.get("data") or {}).get("models_dir")
    if mdir:
        mpath = (ROOT / mdir) if not Path(mdir).is_absolute() else Path(mdir)
        if args.models_dir:
            mpath = Path(args.models_dir).expanduser()
        if not mpath.is_dir():
            raise SystemExit(f"{args.case}: models directory missing: {mpath} (install the frozen files or pass --models-dir)")
        env["ABENCH_MODELS_DIR"] = str(mpath.resolve())
    # task statement standard v2.1: the agent can read the session deadline and development budget (see each case's sandbox guide)
    env["ABENCH_DEADLINE"] = str(int(time.time() + args.timeout_seconds))
    budget = (oj.get("agent_budget") or {})
    if budget.get("gpu_hours") is not None:
        env["ABENCH_DEV_BUDGET_GPU_HOURS"] = str(budget["gpu_hours"])
    prompt = agent_prompt(paths)
    cmd = harness_command(args, prompt, paths)
    if args.agent_wrapper:
        wrapper = expand_wrapper_tokens(args.agent_wrapper, paths)
        if not wrapper:
            raise SystemExit("--agent-wrapper must contain a command")
        cmd = wrapper + cmd
    isolation = ("wrapper:" + str(wrapper_attestation.get("wrapper_id"))
                 if args.agent_wrapper else "host_soft")

    result_dir = Path(args.result_root or ROOT / "runs") / args.case / args.run_id
    # Run records are append-only evidence.  Reusing a run id would silently
    # rewrite agent/evaluation JSON and invalidate hashes in downstream review.
    if not args.dry_run and result_dir.exists() and any(result_dir.iterdir()):
        raise SystemExit(f"run record already exists and is immutable: {result_dir}")
    plan = {"runner_version": RUNNER_VERSION, "case_id": args.case, "run_id": args.run_id, "run_mode": run_mode,
            "harness": args.harness, "isolation": isolation, "wrapper_attestation": wrapper_attestation,
            "agent_view": view_manifest, "knowledge_cutoff": cutoff, "paper_index": paper_index,
            "prompt_sha256": sha256_bytes(prompt.encode()), "passed_env_names": passed,
            "agent_env_names": sorted(env),
            "argv": cmd}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        make_writable(paths["view"]); shutil.rmtree(tmp_root, ignore_errors=True)
        return 0

    # ------------------------------ agent stage
    t0, c0 = time.perf_counter(), os.times()
    try:
        r = subprocess.run(cmd, cwd=paths["workspace"], env=env, input=prompt.encode(), capture_output=True,
                           timeout=args.timeout_seconds)
        status, code, out, err = ("succeeded" if r.returncode == 0 else "failed"), r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        status, code, out, err = "timeout", None, e.stdout or b"", e.stderr or b""
    c1 = os.times()
    agent_wall = round(time.perf_counter() - t0, 3)
    agent_cpu = round((c1.children_user - c0.children_user) + (c1.children_system - c0.children_system), 3)

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "logs").mkdir(exist_ok=True)
    (result_dir / "logs" / "agent_stdout.log").write_bytes(out)
    (result_dir / "logs" / "agent_stderr.log").write_bytes(err)
    kept, dropped = collect_submission(paths["submission"], result_dir / "submission", oj)
    retrieval = collect_retrieval_log(paths["retrieval_log"], result_dir / "retrieval_log.jsonl")
    sub_hash = oj_eval.tree_hash(result_dir / "submission")
    (result_dir / "submission.sha256").write_text(
        "".join(f"{sha256_file(result_dir / 'submission' / k)}  submission/{k}\n" for k in kept), encoding="utf-8")

    agent_record = {
        "schema_version": "2.0", "case_id": args.case, "run_id": args.run_id, "run_mode": run_mode,
        "provenance": "smoke_real" if args.smoke else "benchmark_real", "runner_version": RUNNER_VERSION,
        "harness": args.harness, "isolation": isolation, "wrapper_attestation": wrapper_attestation,
        "status": status, "exit_code": code,
        "agent_wall_seconds": agent_wall, "agent_child_cpu_seconds": agent_cpu,
        "started_at": now(), "host": {"platform": platform.platform(), "python": platform.python_version()},
        "agent_view": view_manifest, "prompt_sha256": plan["prompt_sha256"],
        "passed_env_names": passed, "agent_env_names": sorted(env), "argv": plan["argv"],
        "submission": {"kept": kept, "dropped_self_reported_or_disallowed": dropped, "tree_sha256": sub_hash},
        "knowledge_cutoff": cutoff, "paper_index": paper_index, "retrieval_log": retrieval,
        "stdout_sha256": sha256_bytes(out), "stderr_sha256": sha256_bytes(err),
        "trajectory": summarize_stream(out) if args.harness == "claude" else None,
    }
    (result_dir / "agent_run.json").write_text(json.dumps(agent_record, ensure_ascii=False, indent=2), encoding="utf-8")

    # clean up the agent's temporary directories (checkpoints were copied with the submission; intermediate workspace files are not kept)
    make_writable(paths["view"])
    shutil.rmtree(tmp_root, ignore_errors=True)

    # ------------------------------ evaluation stage
    evaluation = oj_eval.evaluate(args.case, result_dir / "submission",
                                  hidden_key, run_mode) \
        if (args.smoke or args.hidden_key_file or (oj.get("judge") or {}).get("kind") in (None, "pending")) else \
        {"verdict": "not_evaluated", "note": "--hidden-key-file not provided"}
    evaluation["agent_isolation"] = wrapper_attestation
    # oj_eval can only attest judge-side execution.  A result is scoreable
    # only when the agent wrapper itself passed the allow-list as well.
    if run_mode == "benchmark" and not wrapper_attestation.get("scoreable"):
        evaluation["scorable"] = False
        evaluation["scoreability_reason"] = wrapper_attestation.get("reason", "wrapper is not scoreable")
    evaluation["agent_run_sha256"] = sha256_file(result_dir / "agent_run.json")
    (result_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"run_id": args.run_id, "result_dir": str(result_dir), "agent_status": status,
               "isolation": isolation, "kept": kept, "dropped": dropped, "verdict": evaluation.get("verdict"),
               "metrics": evaluation.get("metrics")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "succeeded" and evaluation.get("verdict") in ("evaluated", "judge_pending", "smoke_unscored") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="ABench v2 OJ-style runner")
    ap.add_argument("--case", required=True)
    ap.add_argument("--harness", choices=["command", "claude"], required=True)
    ap.add_argument("--command", help="command harness: the command to run; $ABENCH_* placeholders allowed")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--claude-model")
    ap.add_argument("--claude-effort")
    ap.add_argument("--claude-max-budget-usd", type=float)
    ap.add_argument("--agent-wrapper", help="command prefix that puts the agent in a separate user or container")
    ap.add_argument("--pass-env", action="append", default=[], help="name of a host environment variable passed explicitly to the agent; repeatable")
    ap.add_argument("--hidden-key-file", help="hidden-set key; used only in the evaluation stage")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout-seconds", type=float, default=3600)
    ap.add_argument("--result-root")
    ap.add_argument("--run-id")
    ap.add_argument("--smoke", action="store_true", help="mark as smoke: records provenance=smoke_real and never enters formal scoring")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--papers-index", help="SQLite index from tools/paper_search/build_paper_index.py "
                                           "(default: $ABENCH_PAPERS_INDEX or ~/.abench/papers/current.sqlite)")
    ap.add_argument("--models-dir", help="frozen pretrained files for cases that declare data.models_dir "
                                         "(default: the path in oj.yaml)")
    a = ap.parse_args()
    if a.harness == "command" and not a.command:
        ap.error("--command is required for --harness command")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())

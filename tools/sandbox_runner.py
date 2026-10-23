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
    host files. Smoke only. Benchmark runs need an allow-listed --agent-wrapper (spec/wrapper_policy.yaml):
    "unshare --net -- sudo -u abench-agent --" (dedicated user, no network), "sudo -u abench-agent --" (dedicated
    user, host network: for agent CLIs that call a model API) or a digest-pinned "docker/podman run --network none".
  - User wrappers: before the agent starts, the runner checks as the agent user that the repository, the hidden
    set and the key cannot be read and that no process of that user is running; the environment reaches the agent
    through a one-shot file owned by the agent user (sudo resets the environment); afterwards every process of the
    agent user is killed and files it left in /tmp, /var/tmp and /dev/shm are removed (all recorded).
  - Network (--network): research is part of the work, so the session is online by default ("egress": every
    connection goes through the runner's logging proxy, tools/egress_proxy.py, under spec/egress_policy.yaml —
    literature, code, documentation and packages allowed; dataset/model hubs and the case's own data sources denied —
    and a per-user firewall rejects everything else, so the proxy cannot be bypassed). "none" (netns or container
    wrappers) has no network; "open" is unrestricted and never scorable.
  - The agent runs in its own process group; at the deadline the group gets SIGTERM, then SIGKILL. stdout/stderr
    stream to the run's log files.
  - The hidden set exists on disk only as ciphertext, and the key file never enters the agent view or environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pwd
import signal
import threading
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
from leak_check import leaky, leak_allow  # noqa: E402
import oj_eval  # noqa: E402
import egress_proxy  # noqa: E402
from wrapper_policy import inspect_wrapper, require_benchmark_wrapper  # noqa: E402

RUNNER_VERSION = "sandbox-runner-v2.5-egress"
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


def harness_command(args: argparse.Namespace, prompt: str, paths: dict[str, Path],
                    env: dict[str, str] | None = None) -> list[str]:
    if args.harness == "command":
        # $ABENCH_* placeholders: every ABENCH_* variable of the agent environment (the command is never run by a shell)
        values = {k: v for k, v in (env or {}).items() if k.startswith("ABENCH_")}
        values.update({"ABENCH_VIEW_DIR": str(paths["view"]), "ABENCH_WORKSPACE": str(paths["workspace"]),
                       "ABENCH_SUBMISSION_DIR": str(paths["submission"]), "ABENCH_DATA_DIR": str(paths["view"] / "data"),
                       "ABENCH_MODEL_DIR": str(paths["model"]), "ABENCH_CHECKPOINT_DIR": str(paths["checkpoint"])})
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


NETWORK_TEXT = {
    "egress": "- Network: online through the sandbox proxy (already set in HTTP_PROXY / HTTPS_PROXY). Papers, code,\n"
              "  documentation and package sites are reachable at any point of your work: look things up whenever\n"
              "  your experiments raise a question. Dataset and model hubs and this case's own data sources are\n"
              "  blocked; data and weights come only from $ABENCH_DATA_DIR (and $ABENCH_MODELS_DIR). Every\n"
              "  connection is logged with the run.",
    "none": "- Network: none in this run; use the offline paper search below.",
    "open": "- Network: unrestricted (unscored run).",
}


def agent_prompt(paths: dict[str, Path], network: str = "none") -> str:
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
{NETWORK_TEXT[network]}
- Offline paper search (arXiv titles and abstracts, fast; every call is logged with the run):
  python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"   and   ... show <arXiv id>
- Your working directory is {paths['workspace']} ($ABENCH_WORKSPACE); write code, train and keep intermediate files there.
- Scratch checkpoints can go to {paths['checkpoint']}; it is deleted with the workspace. A checkpoint the task asks you
  to submit must be inside the submission directory.
- Write the final submission to {paths['submission']} ($ABENCH_SUBMISSION_DIR), containing only the files the task asks
  for; submission is one-shot.
- The session deadline (Unix seconds) is in the environment variable ABENCH_DEADLINE; processes are killed at that time,
  so write the submission early.
"""


class GpuMeter:
    """Samples nvidia-smi while the agent runs and integrates GPU-busy time (a GPU counts as busy in an interval when
    its utilisation is above 5% or its memory use is 256 MiB above the level seen at start). Without nvidia-smi
    (CPU-only or Apple MPS hosts) the record says "unmetered"."""

    def __init__(self, interval: float = 5.0):
        import threading
        self.interval, self.busy, self.samples, self.gpus = interval, 0.0, 0, 0
        self.base: list[float] = []
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> list[tuple[float, float]]:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        rows = []
        for line in r.stdout.strip().splitlines():
            u, m = (float(x) for x in line.split(","))
            rows.append((u, m))
        return rows

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                rows = self._read()
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            self.samples += 1
            for i, (u, m) in enumerate(rows):
                base = self.base[i] if i < len(self.base) else 0.0
                if u > 5 or m > base + 256:
                    self.busy += self.interval

    def start(self) -> "GpuMeter":
        if self.available:
            try:
                rows = self._read()
                self.base, self.gpus = [m for _, m in rows], len(rows)
            except (OSError, ValueError, subprocess.SubprocessError):
                self.available = False
        if self.available:
            self._t.start()
        return self

    def stop(self) -> dict[str, Any]:
        if not self.available:
            return {"metered": False, "reason": "nvidia-smi not available (CPU-only or MPS host)"}
        self._stop.set()
        self._t.join(timeout=self.interval + 5)
        return {"metered": True, "gpus": self.gpus, "interval_seconds": self.interval, "samples": self.samples,
                "gpu_busy_seconds": round(self.busy, 1), "gpu_busy_hours": round(self.busy / 3600, 3)}


def budget_check(budget: dict[str, Any], gpu: dict[str, Any], agent_cpu: float) -> dict[str, Any]:
    """Compare metered use with oj.yaml agent_budget (gpu_hours, cpu_core_hours); over budget makes a run unscorable."""
    out: dict[str, Any] = {"budget": budget, "over_budget": False, "checks": []}
    if budget.get("gpu_hours") is not None:
        if gpu.get("metered"):
            used = gpu["gpu_busy_hours"]
            ok = used <= float(budget["gpu_hours"]) * 1.05            # 5% tolerance for sampling granularity
            out["checks"].append({"resource": "gpu_hours", "used": used, "limit": budget["gpu_hours"], "ok": ok})
            out["over_budget"] |= not ok
        else:
            out["checks"].append({"resource": "gpu_hours", "used": None, "limit": budget["gpu_hours"], "ok": None,
                                  "note": "unmetered on this host"})
    if budget.get("cpu_core_hours") is not None:
        used = round(agent_cpu / 3600, 3)
        ok = used <= float(budget["cpu_core_hours"]) * 1.05
        out["checks"].append({"resource": "cpu_core_hours", "used": used, "limit": budget["cpu_core_hours"], "ok": ok,
                              "note": "CPU time of the agent's reaped child processes, including the agent CLI itself"})
        out["over_budget"] |= not ok
    return out


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


def filtered_paper_index(src: str, exclude: list, dst: Path) -> dict[str, Any]:
    """A copy of the paper index without the given arXiv ids (judgment cases: the later papers that settle the
    verdicts). The list itself never reaches the sandbox; the copy is what the agent's tool opens."""
    import sqlite3
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as a, sqlite3.connect(dst) as b:
        a.backup(b)
    con = sqlite3.connect(dst)
    try:
        removed = 0
        for pid in exclude:
            row = con.execute("SELECT rowid, title, authors, abstract FROM papers WHERE id = ?", (str(pid),)).fetchone()
            if row is None:
                continue
            con.execute("INSERT INTO papers_fts(papers_fts, rowid, title, authors, abstract) VALUES('delete', ?, ?, ?, ?)", row)
            con.execute("DELETE FROM papers WHERE rowid = ?", (row[0],))
            removed += 1
        con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('case_filter', ?)", (f"{removed} papers withheld",))
        con.commit()
    finally:
        con.close()
    dst.chmod(0o444)
    return {"path": str(dst), "withheld": removed}


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


# ------------------------------------------------------------------ isolation helpers (user wrappers)
def _as_user(user: str, argv: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-n", "-u", user, "--"] + argv, capture_output=True, timeout=timeout)


def user_processes(user: str) -> list[str]:
    r = subprocess.run(["pgrep", "-u", user, "-a"], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


def isolation_preflight(user: str, paths: dict[str, Path], key: Path | None, oj: dict[str, Any]) -> dict[str, Any]:
    """Checks, as the agent user, that evaluator-side files are out of reach and the workspace is writable.

    Returns the record; "ok" is False if any forbidden path is readable or the user already runs processes."""
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return {"ok": False, "reason": f"agent user {user!r} does not exist"}
    forbidden = [ROOT, ROOT / "index.yaml", ROOT / "cases", ROOT / "sandbox" / "hidden"]
    hid = (oj.get("data") or {}).get("hidden") or {}
    if hid.get("path"):
        forbidden.append(ROOT / hid["path"])
    if key is not None:
        forbidden += [key, key.parent]
    checks, ok = [], True
    for f in forbidden:
        flag = "-x" if f.is_dir() else "-r"
        r = _as_user(user, ["test", flag, str(f)])
        reachable = r.returncode == 0
        checks.append({"path": str(f), "test": flag, "reachable": reachable})
        ok &= not reachable
    w = _as_user(user, ["test", "-w", str(paths["workspace"])]).returncode == 0
    v = _as_user(user, ["test", "-r", str(paths["view"] / "task.md")]).returncode == 0
    procs = user_processes(user)
    rec = {"user": user, "uid": pw.pw_uid, "forbidden_paths": checks, "workspace_writable": w, "view_readable": v,
           "running_processes_before": len(procs)}
    reasons = [f"{c['path']} is reachable by {user}" for c in checks if c["reachable"]]
    if not w or not v:
        reasons.append("workspace not writable or view not readable by the agent user")
    if procs:
        reasons.append(f"{len(procs)} process(es) of {user} already running")
    rec["ok"] = ok and w and v and not procs
    if reasons:
        rec["reason"] = "; ".join(reasons)
    return rec


def stage_tree(src: Path, dst: Path) -> None:
    """Make a read-only copy of src at dst that another user can read: files are hard-linked when they are already
    world-readable (no data copied, the original's mode is left alone), copied otherwise."""
    def link_or_copy(a: str, b: str) -> None:
        if os.stat(a).st_mode & stat.S_IROTH:
            try:
                os.link(a, b)
                return
            except OSError:
                pass
        shutil.copy2(a, b)
        os.chmod(b, 0o444)
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=link_or_copy)
        for d, _, _ in os.walk(dst):
            os.chmod(d, 0o555)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(str(src), str(dst))


def write_launcher(root: Path, user: str, env: dict[str, str], cmd: list[str], cwd: Path) -> list[str]:
    """sudo resets the environment, so the agent's environment travels in a one-shot file readable only by the
    agent user; launch.py reads and deletes it, then execs the harness command. Returns the argv after the wrapper."""
    d = root / ".launch"
    d.mkdir(mode=0o711)
    pw = pwd.getpwnam(user)
    envf = d / "env.json"
    fd = os.open(envf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as f:
        json.dump({"env": env, "cmd": cmd, "cwd": str(cwd)}, f)
    os.chown(envf, pw.pw_uid, pw.pw_gid)
    (d / "launch.py").write_text(
        "import json, os, sys\n"
        "p = sys.argv[1]\n"
        "with open(p) as f:\n"
        "    spec = json.load(f)\n"
        "os.unlink(p)\n"
        "os.chdir(spec['cwd'])\n"
        "os.execvpe(spec['cmd'][0], spec['cmd'], spec['env'])\n", encoding="utf-8")
    (d / "launch.py").chmod(0o444)
    os.chown(d, pw.pw_uid, pw.pw_gid)            # so that launch.py can delete the file
    d.chmod(0o700)
    return [sys.executable, "-I", "-S", str(d / "launch.py"), str(envf)]


def write_launcher_env(root: Path, env: dict[str, str]) -> None:
    """Replace the environment in the launcher file (root may write it: it owns the file system)."""
    envf = root / ".launch" / "env.json"
    spec = json.loads(envf.read_text())
    spec["env"] = env
    st = envf.stat()
    envf.chmod(0o600)
    envf.write_text(json.dumps(spec))
    envf.chmod(0o400)
    os.chown(envf, st.st_uid, st.st_gid)


def cleanup_user(user: str, keep: Path) -> dict[str, Any]:
    """Kill every remaining process of the agent user and remove files it left in shared temporary directories."""
    procs = user_processes(user)
    if procs:
        subprocess.run(["pkill", "-KILL", "-u", user], capture_output=True)
        for _ in range(50):
            if not user_processes(user):
                break
            time.sleep(0.1)
    left: list[str] = []
    for top in ("/tmp", "/var/tmp", "/dev/shm"):
        if not os.path.isdir(top):
            continue
        r = subprocess.run(["find", top, "-xdev", "-path", str(keep), "-prune", "-o", "-user", user, "-prune", "-print"],
                           capture_output=True, text=True)
        left += [x for x in r.stdout.splitlines() if x]
    for x in left:
        if os.path.isdir(x) and not os.path.islink(x):
            shutil.rmtree(x, ignore_errors=True)
        else:
            try:
                os.unlink(x)
            except OSError:
                pass
    return {"stray_processes_killed": len(procs), "stray_processes": procs[:20],
            "processes_after": len(user_processes(user)), "stray_files_removed": left[:50],
            "stray_files_count": len(left)}


def run_agent(cmd: list[str], cwd: Path, env: dict[str, str] | None, prompt: bytes, timeout: float,
              out_path: Path, err_path: Path, grace: float = 10.0) -> tuple[str, int | None]:
    """Run the agent in its own process group with stdout/stderr streamed to files. At the deadline the group
    gets SIGTERM and, after `grace` seconds, SIGKILL."""
    with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=out_f, stderr=err_f,
                                start_new_session=True)

        def feed() -> None:
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        threading.Thread(target=feed, daemon=True).start()
        try:
            code = proc.wait(timeout=timeout)
            status = "succeeded" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
            for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 30)):
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        try:                                     # anything left in the group (background jobs of the agent)
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return status, code


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
    judge_kind = (oj.get("judge") or {}).get("kind")
    if run_mode == "benchmark" and judge_kind not in (None, "pending") and hidden_key is None and not args.defer_evaluation:
        # fail before the agent spends its budget, not after
        raise SystemExit("benchmark runs of a case with a frozen judge need --hidden-key-file "
                         "(or --defer-evaluation to judge later with tools/oj_eval.py)")
    agent_user = wrapper_attestation.get("user") if wrapper_attestation.get("kind") in ("user", "user_netns") else None
    wkind = wrapper_attestation.get("kind")
    network = args.network or {"user": "egress", "user_netns": "none", "container": "none"}.get(wkind, "open")
    if network == "egress" and wkind != "user":
        raise SystemExit("--network egress needs the sudo user wrapper (sudo -u abench-agent --): the proxy and the "
                         "per-user firewall work on the host network")
    if network == "none" and wkind not in ("user_netns", "container"):
        raise SystemExit("--network none needs the netns wrapper (unshare --net -- sudo -u abench-agent --) or a "
                         "container with --network none")
    if network in ("egress", "open") and wkind in ("user_netns", "container"):
        raise SystemExit(f"--network {network} is not possible with the {wkind} wrapper (it has no network)")
    firewall_ok, firewall_why = (egress_proxy.Firewall.available() if network == "egress" else (False, ""))
    if network == "egress" and not firewall_ok and run_mode == "benchmark":
        raise SystemExit(f"--network egress cannot be enforced on this host ({firewall_why}); benchmark runs need "
                         "the per-user firewall (Linux, root, iptables)")
    egress_policy = egress_proxy.policy_for_case(oj, args.harness) if network == "egress" else None

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
    try:
        agent_pw = pwd.getpwnam(agent_user) if agent_user else None
    except KeyError:
        raise SystemExit(f"agent user {agent_user!r} does not exist (create it: useradd --system --no-create-home {agent_user})")
    for k in ("workspace", "home", "tmp", "model", "checkpoint", "submission"):
        paths[k].mkdir(parents=True, exist_ok=True)
        if agent_pw:                             # the agent user owns its writable areas; nobody else can write them
            os.chown(paths[k], agent_pw.pw_uid, agent_pw.pw_gid)
            paths[k].chmod(0o700)
        elif args.agent_wrapper:                 # container: the image's user is unknown here
            paths[k].chmod(0o777)
    view_manifest = build_agent_view(args.case, oj, paths["view"])
    env, passed = agent_env(args, paths)
    # offline paper search: the index stays where the operator installed it (opened read-only by the tool); every call
    # is logged under the workspace and the log is collected with the run record
    cutoff = oj.get("knowledge_cutoff")              # none by default: literature of any date may be used
    paths["retrieval_log"] = paths["workspace"] / ".abench" / "retrieval_log.jsonl"
    paths["retrieval_log"].parent.mkdir(parents=True, exist_ok=True)
    if agent_pw:
        os.chown(paths["retrieval_log"].parent, agent_pw.pw_uid, agent_pw.pw_gid)
    if cutoff:
        env["ABENCH_CUTOFF"] = str(cutoff)
    env["ABENCH_RETRIEVAL_LOG"] = str(paths["retrieval_log"])
    paper_index = paper_index_info(args.papers_index)
    exclude = (oj.get("agent_visible") or {}).get("paper_exclude") or []
    if paper_index and exclude:
        f = filtered_paper_index(paper_index["path"], exclude, paths["root"] / "papers" / "papers_filtered.sqlite")
        paper_index = {**paper_index, "path": f["path"], "withheld": f["withheld"]}
    if paper_index and agent_user and not paper_index["path"].startswith(str(paths["root"]) + os.sep):
        stage_tree(Path(paper_index["path"]), paths["root"] / "papers" / "papers.sqlite")
        paper_index = {**paper_index, "path": str(paths["root"] / "papers" / "papers.sqlite")}
    if paper_index:
        env["ABENCH_PAPERS_INDEX"] = paper_index["path"]
        if cutoff and paper_index.get("cutoff", "9999-12-31") > str(cutoff):
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
        if agent_user:                           # the agent user cannot reach the repository: give it a read-only copy
            stage_tree(mpath.resolve(), paths["root"] / "models")
            mpath = paths["root"] / "models"
        env["ABENCH_MODELS_DIR"] = str(mpath.resolve())
    # task statement standard v2.1: the agent can read the session deadline and development budget (see each case's sandbox guide)
    env["ABENCH_DEADLINE"] = str(int(time.time() + args.timeout_seconds))
    budget = (oj.get("agent_budget") or {})
    if budget.get("gpu_hours") is not None:
        env["ABENCH_DEV_BUDGET_GPU_HOURS"] = str(budget["gpu_hours"])
    prompt = agent_prompt(paths, network)
    cmd = harness_command(args, prompt, paths, env)
    preflight = None
    if agent_user:
        preflight = isolation_preflight(agent_user, paths, hidden_key, oj)
        wrapper_attestation = {**wrapper_attestation, "preflight": preflight}
        if not preflight["ok"]:
            msg = f"isolation preflight failed: {preflight.get('reason')}"
            if run_mode == "benchmark":
                make_writable(paths["view"]); shutil.rmtree(tmp_root, ignore_errors=True)
                raise SystemExit(msg)
            print("warning: " + msg, file=sys.stderr)
    if args.agent_wrapper:
        wrapper = expand_wrapper_tokens(args.agent_wrapper, paths)
        if not wrapper:
            raise SystemExit("--agent-wrapper must contain a command")
        if agent_user and not args.dry_run:
            cmd = wrapper + write_launcher(paths["root"], agent_user, env, cmd, paths["workspace"])
        else:
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
            "network": network, "egress_policy": egress_policy,
            "agent_view": view_manifest, "knowledge_cutoff": cutoff, "paper_index": paper_index,
            "prompt_sha256": sha256_bytes(prompt.encode()), "passed_env_names": passed,
            "agent_env_names": sorted(env),
            "argv": cmd}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        make_writable(paths["view"]); shutil.rmtree(tmp_root, ignore_errors=True)
        return 0

    # ------------------------------ agent stage
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "logs").mkdir(exist_ok=True)
    out_log, err_log = result_dir / "logs" / "agent_stdout.log", result_dir / "logs" / "agent_stderr.log"
    net_rec: dict[str, Any] = {"mode": network}
    proxy = firewall = None
    if network == "egress":
        proxy = egress_proxy.EgressProxy(egress_policy, result_dir / "egress_log.jsonl",
                                         upstream=os.environ.get("ABENCH_EGRESS_UPSTREAM")).start()
        purl = f"http://127.0.0.1:{proxy.port}"
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[k] = purl
        env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
        net_rec.update(policy_version=egress_policy["policy_version"], policy_sha256=egress_policy["policy_sha256"],
                       case_deny=egress_policy["case_deny"], proxy_port=proxy.port,
                       upstream=bool(os.environ.get("ABENCH_EGRESS_UPSTREAM")))
        if firewall_ok:
            firewall = egress_proxy.Firewall(agent_pw.pw_uid, proxy.port)
            try:
                net_rec["firewall"] = firewall.install()
            except RuntimeError as e:
                firewall.remove()
                proxy.stop()
                raise SystemExit(f"egress firewall could not be installed: {e}")
        else:
            net_rec["firewall"] = {"enforced": False, "reason": firewall_why}
        if agent_user:                           # the launcher file was written before the proxy existed
            write_launcher_env(paths["root"], env)
    started_at = now()
    t0, c0 = time.perf_counter(), os.times()
    meter = GpuMeter().start()
    try:
        # a user wrapper gets its environment from the launcher file; sudo itself runs with the runner's minimal env
        status, code = run_agent(cmd, paths["workspace"],
                                 ({k: env[k] for k in BASE_ENV if k in env} if agent_user else env),
                                 prompt.encode(), args.timeout_seconds, out_log, err_log)
    finally:
        if firewall is not None:
            firewall.remove()
        if proxy is not None:
            net_rec["summary"] = proxy.stop()
    gpu_use = meter.stop()
    c1 = os.times()
    agent_wall = round(time.perf_counter() - t0, 3)
    agent_cpu = round((c1.children_user - c0.children_user) + (c1.children_system - c0.children_system), 3)
    cleanup = cleanup_user(agent_user, tmp_root) if agent_user else None
    out, err = out_log.read_bytes(), err_log.read_bytes()
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
        "started_at": started_at, "finished_at": now(), "network": net_rec, "cleanup": cleanup, "host": {"platform": platform.platform(), "python": platform.python_version()},
        "agent_view": view_manifest, "prompt_sha256": plan["prompt_sha256"],
        "passed_env_names": passed, "agent_env_names": sorted(env), "argv": plan["argv"],
        "submission": {"kept": kept, "dropped_self_reported_or_disallowed": dropped, "tree_sha256": sub_hash},
        "knowledge_cutoff": cutoff, "paper_index": paper_index, "retrieval_log": retrieval,
        "stdout_sha256": sha256_bytes(out), "stderr_sha256": sha256_bytes(err),
        "trajectory": summarize_stream(out) if args.harness == "claude" else None,
        "gpu_use": gpu_use, "budget": budget_check(oj.get("agent_budget") or {}, gpu_use, agent_cpu),
    }
    (result_dir / "agent_run.json").write_text(json.dumps(agent_record, ensure_ascii=False, indent=2), encoding="utf-8")

    # clean up the agent's temporary directories (checkpoints were copied with the submission; intermediate workspace files are not kept)
    make_writable(paths["view"])
    shutil.rmtree(tmp_root, ignore_errors=True)

    # ------------------------------ evaluation stage
    if args.smoke or hidden_key is not None or judge_kind in (None, "pending"):
        try:
            evaluation = oj_eval.evaluate(args.case, result_dir / "submission", hidden_key, run_mode)
        except SystemExit as e:              # the record of the agent stage stays complete; the judge error is recorded
            evaluation = {"schema_version": "2.0", "case_id": args.case, "verdict": "judge_error", "metrics": None,
                          "scorable": False, "error": str(e.code if e.code is not None else e)[:2000],
                          "note": "rerun the judge with tools/oj_eval.py on the collected submission after fixing"}
    else:
        evaluation = {"verdict": "not_evaluated", "note": "--defer-evaluation: judge later with tools/oj_eval.py"}
    evaluation["agent_isolation"] = wrapper_attestation
    # oj_eval can only attest judge-side execution.  A result is scoreable
    # only when the agent wrapper itself passed the allow-list as well.
    if run_mode == "benchmark" and not wrapper_attestation.get("scoreable"):
        evaluation["scorable"] = False
        evaluation["scoreability_reason"] = wrapper_attestation.get("reason", "wrapper is not scoreable")
    if run_mode == "benchmark" and not (network == "none" or (network == "egress" and
                                                              (net_rec.get("firewall") or {}).get("enforced"))):
        evaluation["scorable"] = False
        evaluation["scoreability_reason"] = f"agent network {network!r} was not restricted to the egress policy"
    if agent_record["budget"]["over_budget"]:
        evaluation["scorable"] = False
        evaluation["scoreability_reason"] = "the agent exceeded the development budget: " + json.dumps(
            [c for c in agent_record["budget"]["checks"] if c.get("ok") is False])
    evaluation["agent_run_sha256"] = sha256_file(result_dir / "agent_run.json")
    (result_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"run_id": args.run_id, "result_dir": str(result_dir), "agent_status": status,
               "isolation": isolation, "kept": kept, "dropped": dropped, "verdict": evaluation.get("verdict"),
               "metrics": evaluation.get("metrics")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "succeeded" and evaluation.get("verdict") in ("evaluated", "judge_pending", "smoke_unscored",
                                                                       "not_evaluated") else 1


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
    ap.add_argument("--network", choices=["egress", "none", "open"],
                    help="agent network: egress (default with the sudo wrapper: logged proxy + firewall, "
                         "spec/egress_policy.yaml), none (default with the netns and container wrappers), "
                         "open (unrestricted; never scorable)")
    ap.add_argument("--defer-evaluation", action="store_true",
                    help="benchmark run without the key: collect the submission now, judge later with tools/oj_eval.py")
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

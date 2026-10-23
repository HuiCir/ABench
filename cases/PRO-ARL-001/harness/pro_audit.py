"""Pro Set process integrity (spec/pro_set.md section 2): supervised training runs, a hash chain of registered
checkpoints, and the sampled-resume check (evaluator code, agent-visible; identical copy in every Pro audit case).

How a registered run works (in the sandbox; the session's audit directory is written only by this supervisor, which
the sandbox runner starts outside the agent's user):

  python3 pro_audit.py run --name NAME --code CODE_DIR -- torchrun ... train.py ...

  - the command runs from CODE_DIR (the directory you will submit), so it must refer to its files by relative paths
    and to data, models and outputs through the environment variables of the case (ABENCH_DATA_DIR,
    ABENCH_MODELS_DIR, ABENCH_CKPT_DIR): the evaluator re-runs it from the submitted directory;

  - the command, the tree hash of the code directory and of the starting weights, and the start time open the run
    in the chain (ABENCH_AUDIT_DIR/chain.jsonl);
  - the training program reads ABENCH_CKPT_DIR and writes resumable checkpoints there as sub-directories named
    after the step (for example step_001200/), each with the deliverable weights in model/ (the case's model
    format), anything else needed to resume (optimiser, data position, random state) beside it, and a file named DONE
    written last; the supervisor hashes every completed
    checkpoint and appends it to the chain; two registrations may be at most REGISTER_EVERY (30 minutes) apart, and
    the training program must therefore save at least that often;
  - the program prints lines "ABENCH step=<int> <name>=<float> ..." (for example loss=..., reward=...); the
    supervisor stores them with wall-clock times in ABENCH_AUDIT_DIR/<NAME>/metrics.jsonl, and GPU utilisation and
    memory every 60 s in telemetry.jsonl;
  - a checkpoint directory must not change after DONE appears (changes are detected at the end of the run).

Delivery: the submitted checkpoint/ must be byte-identical to the model/ directory of a registered checkpoint (same
weight and config files), and the chain must lead from the allowlisted starting weights to it.

Sampled resume (by the evaluator): one registered checkpoint of the delivering run, chosen with a secret seed among
those followed by at least RESUME_MIN_STEPS more logged steps, is resumed with the run's own command and
ABENCH_RESUME_FROM=<checkpoint dir> for at most RESUME_SECONDS; the resumed metric must match the logged one over the
overlapping steps (trajectory_match).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REGISTER_EVERY, GAP_SLACK = 1800.0, 300.0
RESUME_SECONDS, RESUME_MIN_STEPS = 3600.0, 20
LINE = re.compile(r"^ABENCH\s+step=(\d+)((?:\s+[A-Za-z_][\w.]*=[-+0-9.eEinfa]+)+)\s*$")


# ----------------------------------------------------------------------------------------------- hashing
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def file_hashes(d: Path) -> dict:
    d = Path(d)
    return {p.relative_to(d).as_posix(): sha256_file(p) for p in sorted(d.rglob("*"))
            if p.is_file() and not p.is_symlink() and p.name != "DONE" and "__pycache__" not in p.parts}


def tree_hash(files: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(k.encode() + b"\0" + bytes.fromhex(files[k]))
    return h.hexdigest()


CODE_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".json", ".toml")
NOT_CODE = ("checkpoint/", "checkpoints/", "report/", "audit/", "data/", "outputs/")


def code_hash(d: Path) -> str:
    """Tree hash of the program files of a code directory (what the submission must reproduce)."""
    files = {k: v for k, v in file_hashes(d).items() if k.endswith(CODE_SUFFIXES) and not k.startswith(NOT_CODE)}
    return tree_hash(files)


def weights_hashes(d: Path) -> dict:
    """Hashes of the tensor files of a model directory (what must match between a delivery and a registration)."""
    return {k: v for k, v in file_hashes(d).items()
            if k.endswith((".safetensors", ".bin", ".pt", ".pth")) or k.endswith("config.json")}


# ----------------------------------------------------------------------------------------------- chain
class Chain:
    """Append-only JSONL hash chain; every entry carries the hash of the previous one."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def entries(self) -> list:
        if not self.path.is_file():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def append(self, kind: str, **fields) -> dict:
        with self.lock:
            ents = self.entries()
            prev = ents[-1]["hash"] if ents else "0" * 64
            e = {"i": len(ents), "time": time.time(), "kind": kind, **fields, "prev": prev}
            e["hash"] = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, sort_keys=True) + "\n")
            return e


def verify_chain(entries: list) -> list:
    """Problems with a chain: broken links or hashes, time going backwards, unregistered gaps in a run."""
    errs, prev, last_t = [], "0" * 64, -1.0
    last_reg: dict = {}
    for k, e in enumerate(entries):
        body = {x: y for x, y in e.items() if x != "hash"}
        if e.get("i") != k or e.get("prev") != prev:
            errs.append(f"entry {k}: broken link")
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e.get("hash"):
            errs.append(f"entry {k}: hash mismatch")
        if e["time"] < last_t:
            errs.append(f"entry {k}: time goes backwards")
        run = e.get("run")
        if e["kind"] in ("checkpoint", "end") and run in last_reg and e["time"] - last_reg[run] > REGISTER_EVERY + GAP_SLACK:
            errs.append(f"entry {k}: {e['time'] - last_reg[run]:.0f} s without a registration in run {run}")
        if e["kind"] in ("start", "checkpoint"):
            last_reg[run] = e["time"]
        if e["kind"] == "violation":
            errs.append(f"entry {k}: {e.get('what')}")
        prev, last_t = e["hash"], e["time"]
    return errs


def model_files(files: dict) -> dict:
    """The deliverable part of a registered checkpoint: weight and config files under model/, without the prefix."""
    sub = {k[len("model/"):]: v for k, v in files.items() if k.startswith("model/")}
    return {k: v for k, v in sub.items() if k.endswith((".safetensors", ".bin", ".pt", ".pth")) or k.endswith("config.json")}


def trace(entries: list, delivered: dict, start_hash: str | None) -> dict:
    """Find the registered checkpoint whose model/ is identical to the delivered files; check its run started from
    the allowlisted weights (or resumed from a registered checkpoint that traces back to them)."""
    want = tree_hash(delivered) if delivered else None
    hit = next((e for e in entries if e["kind"] == "checkpoint" and want and tree_hash(model_files(e["files"])) == want),
               None)
    if hit is None:
        return {"ok": False, "why": "delivered files match no registered checkpoint"}
    runs = {e["run"]: e for e in entries if e["kind"] == "start"}
    seen, run = set(), hit["run"]
    while run not in seen:
        seen.add(run)
        st = runs.get(run)
        if st is None:
            return {"ok": False, "why": f"run {run} has no start entry"}
        if st.get("resume_from"):
            parent = next((e for e in entries if e["kind"] == "checkpoint" and e["hash"] == st["resume_from"]), None)
            if parent is None:
                return {"ok": False, "why": f"run {run} resumed from an unregistered checkpoint"}
            run = parent["run"]
            continue
        ok = st.get("start_weights") == start_hash
        return {"ok": ok, "why": "" if ok else f"run {run} did not start from the allowlisted weights",
                "checkpoint": hit, "run": hit["run"]}
    return {"ok": False, "why": "cycle in resume chain"}


# ----------------------------------------------------------------------------------------------- metrics lines
def parse_line(line: str):
    m = LINE.match(line.strip())
    if not m:
        return None
    out = {"step": int(m.group(1))}
    for kv in m.group(2).split():
        k, v = kv.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            return None
    return out


def trajectory_match(orig: list, resumed: list, key: str, rel: float = 0.05, n_sigma: float = 3.0) -> dict:
    """Compare a logged metric of the original run and of the resume over their common steps.

    Both are averaged over the common steps; the difference must be within max(rel x |original mean|,
    n_sigma x the standard error of the difference estimated from step-to-step noise of the original)."""
    a = {int(r["step"]): r[key] for r in orig if key in r}
    b = {int(r["step"]): r[key] for r in resumed if key in r}
    common = sorted(set(a) & set(b))
    if len(common) < RESUME_MIN_STEPS // 2:
        return {"ok": False, "why": f"only {len(common)} common logged steps", "n": len(common)}
    import numpy as np
    x, y = np.array([a[s] for s in common]), np.array([b[s] for s in common])
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return {"ok": False, "why": "non-finite values", "n": len(common)}
    noise = np.std(np.diff(x)) / np.sqrt(2) if len(x) > 2 else 0.0
    se = noise * np.sqrt(2.0 / len(x))
    d = float(y.mean() - x.mean())
    tol = max(rel * abs(float(x.mean())), n_sigma * float(se))
    return {"ok": bool(abs(d) <= tol), "diff": d, "tolerance": tol, "n": len(common), "key": key}


def pick_resume_point(entries: list, run: str, metrics: list, u: float) -> dict | None:
    """The registered checkpoint of `run` to resume from, chosen by u in [0, 1) among those followed by at least
    RESUME_MIN_STEPS logged steps."""
    last = max((int(r["step"]) for r in metrics), default=-1)
    cands = [e for e in entries if e["kind"] == "checkpoint" and e["run"] == run and e.get("step") is not None
             and int(e["step"]) + RESUME_MIN_STEPS <= last]
    return cands[int(u * len(cands)) % len(cands)] if cands else None


# ----------------------------------------------------------------------------------------------- supervisor
def _gpu_sample() -> list:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return [[float(v) for v in ln.split(",")] for ln in r.stdout.strip().splitlines() if ln.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def supervise(name: str, cmd: list, audit: Path, ckpt: Path, code: Path, start_weights: Path | None,
              resume_from: str | None = None, resume_dir: str | None = None, max_seconds: float | None = None,
              env_extra: dict | None = None) -> int:
    """Run cmd, register checkpoints, store metrics and telemetry. Returns the exit code."""
    audit, ckpt = Path(audit), Path(ckpt)
    (audit / name).mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    chain = Chain(audit / "chain.jsonl")
    chain.append("start", run=name, cmd=cmd, code=code_hash(code),
                 start_weights=tree_hash(weights_hashes(start_weights)) if start_weights else None,
                 resume_from=resume_from)
    registered: dict = {}
    stop = threading.Event()

    def scan(final: bool = False):
        for d in sorted(p for p in ckpt.iterdir() if p.is_dir()):
            if (d / "DONE").is_file() and d.name not in registered:
                files = file_hashes(d)
                m = re.search(r"(\d+)$", d.name)
                registered[d.name] = files
                chain.append("checkpoint", run=name, dir=d.name, step=int(m.group(1)) if m else None, files=files)
            elif final and d.name in registered and file_hashes(d) != registered[d.name]:
                chain.append("violation", run=name, what=f"checkpoint {d.name} changed after registration")

    def watcher():
        tel = open(audit / name / "telemetry.jsonl", "a", encoding="utf-8")
        while not stop.wait(60):
            scan()
            tel.write(json.dumps({"time": time.time(), "gpus": _gpu_sample()}) + "\n")
            tel.flush()
        tel.close()

    env = dict(os.environ, ABENCH_CKPT_DIR=str(ckpt), **(env_extra or {}))
    if resume_dir:
        env["ABENCH_RESUME_FROM"] = str(resume_dir)
    th = threading.Thread(target=watcher, daemon=True)
    th.start()
    t0 = time.time()
    with open(audit / name / "stdout.txt", "w", encoding="utf-8") as so, \
            open(audit / name / "metrics.jsonl", "a", encoding="utf-8") as mf:
        p = subprocess.Popen(cmd, env=env, cwd=str(code), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)                    # run from the code directory, so the command is portable
        for line in p.stdout:
            so.write(line)
            r = parse_line(line)
            if r is not None:
                r["time"] = time.time()
                mf.write(json.dumps(r) + "\n")
                mf.flush()
            if max_seconds and time.time() - t0 > max_seconds:
                p.terminate()
                break
        try:
            code_ = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            code_ = p.wait()
    stop.set()
    th.join(5)
    scan(final=True)
    chain.append("end", run=name, returncode=code_, seconds=time.time() - t0)
    return code_


def read_metrics(audit: Path, run: str) -> list:
    p = Path(audit) / run / "metrics.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def main() -> int:
    ap = argparse.ArgumentParser(description="Pro Set audited training runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a training command under the supervisor")
    r.add_argument("--name", required=True)
    r.add_argument("--code", default=".")
    r.add_argument("--resume-from", help="hash of a registered checkpoint entry this run continues from")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    v = sub.add_parser("verify", help="check the chain and the traceability of a delivered checkpoint")
    v.add_argument("--deliver", required=True)
    a = ap.parse_args()
    audit = Path(os.environ.get("ABENCH_AUDIT_DIR", "audit"))
    if a.cmd == "run":
        rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
        start = os.environ.get("ABENCH_START_WEIGHTS")
        ck = Path(os.environ.get("ABENCH_CKPT_ROOT", "checkpoints")) / a.name
        resume_dir = None
        if a.resume_from:
            e = next((x for x in Chain(audit / "chain.jsonl").entries() if x.get("hash") == a.resume_from), None)
            if e is None:
                raise SystemExit("--resume-from must be the hash of a registered checkpoint entry")
            resume_dir = str(Path(os.environ.get("ABENCH_CKPT_ROOT", "checkpoints")) / e["run"] / e["dir"])
        return supervise(a.name, rest, audit, ck, Path(a.code), Path(start) if start else None,
                         resume_from=a.resume_from, resume_dir=resume_dir)
    ents = Chain(audit / "chain.jsonl").entries()
    errs = verify_chain(ents)
    start = os.environ.get("ABENCH_START_WEIGHTS")
    t = trace(ents, weights_hashes(Path(a.deliver)), tree_hash(weights_hashes(Path(start))) if start else "")
    print(json.dumps({"chain_errors": errs, "trace": {k: v for k, v in t.items() if k != "checkpoint"}}, indent=2))
    return 0 if not errs and t["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

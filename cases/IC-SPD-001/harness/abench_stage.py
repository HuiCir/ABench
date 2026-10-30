#!/usr/bin/env python3
"""ABench stage runner, shared by every standard case (identical copy in each case's harness/ directory).

The judge (outside the sandbox) and selfcheck.py (inside the sandbox) both run a case through this file, so the
protocol you test against is exactly the protocol you are evaluated with.

A case defines its stages in harness/stages.py. One run of a stage is one fresh Python process:

  1. setup(ctx)            untimed: load public data, create the CUDA context, etc. (case code, same for everyone)
  2. handshake byte R      the parent's clock starts
  3. timed(ctx, state)     imports your submission and does the timed work (training, loading a checkpoint, ...)
  4. handshake byte T      the parent's clock stops; the timed phase is over
  5. the parent now copies the held-out inputs (if this stage has any) into ctx.post_dir and sends byte G;
     they do not exist anywhere in the run directory before T, so the timed phase cannot read them
  6. finish(ctx, state)    produces the outputs on the held-out inputs (prediction, sampling, rollouts, ...)
  7. handshake byte P      must arrive within the stage's finish_cap seconds after G
  8. outputs are written to <work>/<stage>.out.npz and <work>/<stage>.meta.json and the process exits

The parent requires exactly R, T, P in this order, checks that the process exits within finish_cap + tail_slack
seconds of G (step 8 counts against finish_cap, unless the stage sets slack_after_p: then it may also use tail_slack,
as IC-SPD-001's timing contract allows), and gives every run empty HOME / TMPDIR / compiler cache directories and no network proxy.
Labels and reference values never enter the child process: they stay with the parent (the judge or selfcheck.py).

The judge runs every stage in a fresh, offline Docker container. Selfcheck uses the same stdio protocol locally.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KIT_VERSION = "abench-stage-v1.2"
THIS = Path(__file__).resolve()


# --------------------------------------------------------------------------- shared definitions
@dataclass
class Stage:
    """One stage of a case pipeline. setup/finish may be None."""
    timed: Callable[["Context", Any], Any]
    setup: Callable[["Context"], Any] | None = None
    finish: Callable[["Context", Any], dict | None] | None = None


@dataclass
class Context:
    stage: str
    submission: Path
    work: Path
    public: Path | None
    models: Path | None
    seed: int
    device: str
    params: dict = field(default_factory=dict)
    post_dir: Path | None = None
    pre_dir: Path | None = None          # held-out inputs given before the timed phase (PIPELINE held_out_pre)
    meta: dict = field(default_factory=dict)

    def sync(self) -> None:
        """Wait for queued GPU work so that the handshake reflects finished work."""
        torch = sys.modules.get("torch")
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def import_submission(self, entry: str, required: tuple[str, ...] = ()):
        """Import a module from the submission directory (counted in the timed phase when called there)."""
        sub = str(self.submission)
        if sub not in sys.path:
            sys.path.insert(0, sub)
        path = self.submission / entry
        spec = importlib.util.spec_from_file_location("abench_submission_" + Path(entry).stem, path)
        if spec is None or spec.loader is None or not path.is_file():
            raise SystemExit(f"cannot import {entry}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        missing = [n for n in required if not callable(getattr(mod, n, None))]
        if missing:
            raise SystemExit(f"{entry} must define {', '.join(missing)}")
        return mod


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def to_numpy(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------- child side
def _child(a: argparse.Namespace) -> int:
    # stdout is exclusively the protocol stream; submission prints cannot corrupt it.
    # Keep --fd visible for the protocol-forgery regression test and older callers.
    fd = os.dup(a.fd if a.fd is not None else sys.stdout.fileno())
    if "--fd" in sys.argv:
        sys.argv[sys.argv.index("--fd") + 1] = str(fd)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    for target, name in ((1, "_stdout.txt"), (2, "_stderr.txt")):
        logfd = os.open(Path(a.work) / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(logfd, target)
        os.close(logfd)
    import numpy as np
    sys.path.insert(0, str(THIS.parent))
    sys.path.insert(0, str(Path(a.stages).resolve().parent))      # the case's own helper modules
    stages = load_module(Path(a.stages), "abench_case_stages")
    stage: Stage = stages.STAGES[a.stage]
    ctx = Context(stage=a.stage, submission=Path(a.submission), work=Path(a.work),
                  public=Path(a.public) if a.public else None, models=Path(a.models) if a.models else None,
                  seed=a.seed, device=a.device, params=json.loads(a.params or "{}"),
                  post_dir=Path(a.work) / "post", pre_dir=Path(a.work) / "pre")
    state = stage.setup(ctx) if stage.setup else None
    try:
        import torch  # noqa: F401  (created here so that context creation is never timed)
        if a.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.init()
            torch.zeros(1, device=a.device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    child_r = time.perf_counter()
    os.write(fd, b"R")                                   # ---- clock starts
    state = stage.timed(ctx, state)
    ctx.sync()
    child_t = time.perf_counter()
    os.write(fd, b"T")                                   # ---- clock stops
    go = sys.stdin.buffer.read(1)                        # parent materialises held-out inputs, then sends G
    if go != b"G":
        raise SystemExit("protocol: expected G from parent")
    outputs = (stage.finish(ctx, state) if stage.finish else None) or {}
    ctx.sync()
    os.write(fd, b"P")                                   # ---- outputs ready
    arrays = {k: to_numpy(v) for k, v in outputs.items()}
    for k, v in arrays.items():
        if v.dtype == object:
            raise SystemExit(f"output {k!r} has dtype object; return numeric or string arrays")
    np.savez(Path(a.work) / f"{a.stage}.out.npz", **arrays)
    torch = sys.modules.get("torch")
    meta = dict(ctx.meta)
    meta["child_timing"] = {"r_monotonic": child_r, "t_monotonic": child_t,
                            "timed_seconds": child_t - child_r}
    meta["torch"] = getattr(torch, "__version__", None)
    if torch is not None and a.device.startswith("cuda") and torch.cuda.is_available():
        meta["device_name"] = torch.cuda.get_device_name()
        meta["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    (Path(a.work) / f"{a.stage}.meta.json").write_text(json.dumps(meta, default=str)[:20000], encoding="utf-8")
    os.close(fd)
    return 0


# --------------------------------------------------------------------------- parent side
def _read_regular(path: Path, tail: int | None = None) -> bytes | None:
    """Contents (or only the last tail bytes) of a file the child wrote, or None: never follows a symbolic link (a child must not make the parent
    read, and copy into a run record, a file the child itself cannot read)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        import stat as _st
        if not _st.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb") as f:
            fd = -1
            if tail is not None:
                f.seek(max(0, os.fstat(f.fileno()).st_size - tail))
            return f.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _fresh_dir(d: Path) -> None:
    """Remove whatever is at d (directory, file or symbolic link) and create an empty directory owned by the parent."""
    if d.is_symlink() or d.is_file():
        d.unlink()
    elif d.exists():
        shutil.rmtree(d)
    d.mkdir(mode=0o755)


def _env(workdir: Path, threads: int, extra: dict[str, str] | None) -> dict[str, str]:
    caches = {k: workdir / "_env" / k.lower() for k in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR",
                                                        "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "HF_HOME")}
    for p in caches.values():
        p.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "CUDA_CACHE_DISABLE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "no_proxy": "*", "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
           "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"}
    env.update({k: str(v) for k, v in caches.items()})
    for k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK"):
        if os.environ.get(k) is not None:
            env[k] = os.environ[k]
    env.update(extra or {})
    return env


class LocalStageSession:
    """Local selfcheck transport; the Docker transport implements the same small interface."""
    def __init__(self, cmd, work, env):
        self.proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self.protocol_fd = self.proc.stdout.fileno()
        self.work = work

    def stage_post(self, files):
        post = self.work / "post"
        _fresh_dir(post)
        for name, src in files.items():
            shutil.copyfile(src, post / name)

    def collect(self):
        return {}

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc.wait()


class LocalStageLauncher:
    def start(self, *, cmd, work, env, **kwargs):
        return LocalStageSession(cmd, work, env)


def run_stage(stages_path: Path, stage: str, submission: Path, work: Path, *, public: Path | None = None,
              models: Path | None = None, post_files: dict[str, Path] | None = None,
              pre_files: dict[str, Path] | None = None, seed: int = 0,
              device: str = "cuda", params: dict | None = None, timeout: float = 600.0, finish_cap: float = 60.0,
              tail_slack: float = 5.0, threads: int = 8, extra_env: dict[str, str] | None = None,
              python: str = sys.executable, launcher=None, artifacts=(), slack_after_p: bool = False) -> dict[str, Any]:
    """Run one stage in a fresh process and time it with the parent's clock.

    post_files: name -> source path, copied into <work>/post/<name> only after T has been received.
    pre_files: name -> source path, copied into <work>/pre/<name> before the process starts (inputs the timed phase
    needs, e.g. the training split of a held-out dataset in cases where the evaluator trains); removed afterwards.
    slack_after_p: writing the outputs after P may use tail_slack beyond finish_cap (otherwise P and the outputs
    both have to come within finish_cap of G).
    Returns a record with status, timed_seconds, finish_seconds, phase_seconds (wall time of starting, delivering the
    held-out inputs, running and collecting the stage; never part of a metric) and, if ok, the paths of the outputs.
    """
    work.mkdir(parents=True, exist_ok=True)
    post = work / "post"
    if post.exists() or post.is_symlink():
        post.unlink() if post.is_symlink() else shutil.rmtree(post)
    pre = work / "pre"
    if pre.exists() or pre.is_symlink():
        pre.unlink() if pre.is_symlink() else shutil.rmtree(pre)
    if pre_files:
        pre.mkdir(parents=True)
        for name, src in pre_files.items():
            shutil.copyfile(src, pre / name)
    cmd = [python, str(THIS), "--child", "--stages", str(stages_path), "--stage", stage, "--submission",
           str(submission), "--work", str(work), "--seed", str(seed), "--device", device, "--fd", "1",
           "--params", json.dumps(params or {})]
    if public is not None:
        cmd += ["--public", str(public)]
    if models is not None:
        cmd += ["--models", str(models)]
    env = _env(work, threads, extra_env)
    env["ABENCH_SEED"] = str(seed)
    t_start = time.perf_counter()
    session = (launcher or LocalStageLauncher()).start(
        cmd=cmd, work=work, env=env, stages_path=stages_path,
        submission=submission, public=public, models=models, artifacts=artifacts)
    proc, r = session.proc, session.protocol_fd
    t_spawn = time.perf_counter()  # staging/upload is outside the stage execution cap
    phases = {"start": round(t_spawn - t_start, 4)}
    marks: dict[bytes, float] = {}
    seq = b""
    status, deadline = "ok", t_spawn + timeout
    t_go = None
    try:
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                status = "time_limit_exceeded"
                break
            if t_go is not None:                          # finish phase: enforce the cap with the parent clock
                cap = finish_cap + (tail_slack if slack_after_p and b"P" in marks else 0.)
                left = min(left, t_go + cap - time.perf_counter())
                if left <= 0:
                    status = "finish_time_exceeded"
                    break
            ready, _, _ = select.select([r], [], [], left)
            if not ready:
                continue
            chunk = os.read(r, 16)
            now = time.perf_counter()
            if not chunk:
                break
            for b in chunk:
                seq += bytes([b])
                marks.setdefault(bytes([b]), now)
            if b"P" in marks and t_go is None:
                # Honest children block on G after T. Batched TP cannot skip that barrier.
                status = "protocol_violation"
                break
            if seq == b"RT" and t_go is None:
                t_post = time.perf_counter()
                session.stage_post(post_files or {})
                phases["post_inputs"] = round(time.perf_counter() - t_post, 4)
                try:
                    proc.stdin.write(b"G")
                    proc.stdin.flush()
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                t_go = time.perf_counter()
            elif seq not in (b"R", b"RT", b"RTP"):
                status = "protocol_violation"
                break
        if status == "ok":
            limit = (t_go + finish_cap + tail_slack) if t_go is not None else deadline
            try:
                proc.wait(timeout=max(0.5, min(deadline, limit) - time.perf_counter()))
            except subprocess.TimeoutExpired:
                status = "finish_time_exceeded" if t_go is not None else "time_limit_exceeded"
    finally:
        interrupted = sys.exc_info()[0] is not None
        try:
            session.close()
            t_exit = time.perf_counter()
            transport_record = session.collect()
            phases.update(run=round(t_exit - t_spawn, 4), collect=round(time.perf_counter() - t_exit, 4))
        except BaseException:
            interrupted = True
            raise
        finally:
            if interrupted:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream:
                        stream.close()
        for d in (post, pre):                             # held-out inputs never outlive the run
            if d.is_symlink():
                d.unlink()
            elif d.exists():
                shutil.rmtree(d, ignore_errors=True)
    stdout = _read_regular(work / "_stdout.txt", tail=1500) or b""     # a chatty stage cannot fill the parent's memory
    stderr = _read_regular(work / "_stderr.txt", tail=3000) or b""
    if not stderr and proc.stderr is not None:
        stderr = proc.stderr.read()
    proc.stdout.close()
    proc.stderr.close()
    try:
        proc.stdin.close()
    except (BrokenPipeError, ValueError):
        pass
    rec: dict[str, Any] = {"stage": stage, "seed": seed, "exit_code": proc.returncode,
                           "stdout_tail": stdout[-1500:].decode("utf-8", "replace"),
                           "stderr_tail": stderr[-3000:].decode("utf-8", "replace")}
    rec.update(transport_record)
    rec["phase_seconds"] = phases
    if status == "ok" and proc.returncode != 0:
        status = "runtime_error"
    if status == "ok" and seq != b"RTP":
        status = "protocol_violation"
    if seq and status == "protocol_violation":
        rec["handshake"] = seq.decode("latin-1")
    if b"R" in marks and b"T" in marks:
        rec["timed_seconds"] = round(marks[b"T"] - marks[b"R"], 4)
    if t_go is not None and b"P" in marks:
        rec["finish_seconds"] = round(marks[b"P"] - t_go, 4)
        rec["tail_seconds"] = round(t_exit - t_go, 4)
    rec["status"] = status
    if status == "ok":
        out = work / f"{stage}.out.npz"
        if not out.is_file() or out.is_symlink():
            rec["status"] = "protocol_violation"
            rec["detail"] = "outputs are not a regular file"
            return rec
        rec["outputs"] = str(out)
        raw = _read_regular(work / f"{stage}.meta.json")
        if raw is not None:
            try:
                rec["meta"] = json.loads(raw.decode("utf-8"))   # self-reported, for reference only
                if isinstance(rec["meta"], dict) and "child_timing" in rec["meta"]:
                    rec["child_timing"] = rec["meta"]["child_timing"]
            except ValueError:
                rec["meta"] = None
    return rec


def _check_npz(path: Path, *, max_bytes: int = 8 * 1024**3, max_members: int = 100000) -> None:
    """Bound expanded output bytes before NumPy reads untrusted ZIP members.

    NumPy allocates an array from the shape in an NPY header before it reads the payload, so every member NumPy
    would parse as an array (any member starting with the NPY magic, whatever its name) must declare no more
    bytes than the member actually holds."""
    import math
    from numpy.lib import format as npy
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise ValueError("output NPZ exceeds member limit")
        if sum(member.file_size for member in members) > max_bytes:
            raise ValueError("output NPZ exceeds uncompressed byte limit")
        for member in members:
            with archive.open(member) as f:
                if f.read(len(npy.MAGIC_PREFIX)) != npy.MAGIC_PREFIX:
                    continue                            # NumPy returns such members as bytes, bounded above
            with archive.open(member) as f:
                version = npy.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = npy.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, dtype = npy.read_array_header_2_0(f)
                else:
                    raise ValueError(f"output member {member.filename!r} uses unsupported NPY version {version}")
                if dtype.hasobject:
                    raise ValueError(f"output member {member.filename!r} has dtype object")
                if any(n < 0 for n in shape):
                    raise ValueError(f"output member {member.filename!r} has a negative dimension")
                count = math.prod(shape)
                if count > max_bytes or f.tell() + count * dtype.itemsize > member.file_size:
                    raise ValueError(f"output member {member.filename!r} declares more data than it contains")


def run_pipeline(stages_mod, stages_path: Path, submission: Path, work: Path, *, public: Path | None,
                 models: Path | None, held_out: Path | None, seed: int, device: str,
                 extra_env: dict[str, str] | None = None, scale: float = 1.0,
                 held_pre: Path | None = None, judge_models: Path | None = None,
                 launcher=None) -> dict[str, Any]:
    """Run all stages in stages_mod.PIPELINE for one seed. held_out is the npz of held-out inputs (no labels).

    Returns {"status", "stages": [records], "outputs": {stage: {key: array}},
             "seconds": {stage: timed_seconds, stage + ".finish": finish_seconds}}.
    scale multiplies the timeouts (selfcheck --quick uses a smaller value on smaller data; the judge uses 1.0).
    held_pre is the npz of held-out inputs a stage receives before its timed phase (stages with held_out_pre: true).
    PIPELINE entries: stage, timeout, finish_cap, tail_slack, slack_after_p (bool), threads, held_out (bool), held_out_pre (bool), models (bool, default True:
    pass the frozen model directory), judge_models (bool: pass the evaluator-only model directory instead, e.g. a
    reward model; such a stage must not import the submission), carry ({previous stage: [file or directory names]}).
    """
    import numpy as np
    recs, outs, secs = [], {}, {}
    for spec in stages_mod.PIPELINE:
        name = spec["stage"]
        # every stage gets its own directory; only the files a stage declares in "carry" move between stages
        sdir = work / name
        if sdir.exists():
            shutil.rmtree(sdir)
        sdir.mkdir(parents=True)
        for src_stage, names in (spec.get("carry") or {}).items():
            for n in names:
                src = work / src_stage / n
                # symbolic links are copied as links, never followed: a stage must not carry a file it cannot read
                if src.is_symlink():
                    os.symlink(os.readlink(src), sdir / n)
                elif src.is_dir():
                    shutil.copytree(src, sdir / n, symlinks=True)
                elif src.is_file():
                    shutil.copyfile(src, sdir / n, follow_symlinks=False)
                else:
                    rec = {"stage": name, "seed": seed, "status": "missing_artifact",
                           "detail": f"stage {src_stage!r} did not produce {n!r}"}
                    recs.append(rec)
                    return {"status": rec["status"], "failed_stage": name, "stages": recs, "outputs": outs, "seconds": secs}
        post = {"held_out.npz": held_out} if spec.get("held_out") and held_out is not None else None
        rec = run_stage(stages_path, name, submission, sdir, public=public,
                        models=(judge_models if spec.get("judge_models") else
                                models if spec.get("models", True) else None), post_files=post,
                        pre_files={"held_pre.npz": held_pre} if spec.get("held_out_pre") and held_pre is not None else None,
                        seed=seed, device=device, params=getattr(stages_mod, "PARAMS", {}),
                        timeout=float(spec["timeout"]) * scale, finish_cap=float(spec.get("finish_cap", 60)),
                        tail_slack=float(spec.get("tail_slack", 5)), threads=int(spec.get("threads", 8)),
                        slack_after_p=bool(spec.get("slack_after_p")), extra_env=extra_env, launcher=launcher,
                        artifacts=sorted({artifact for other in stages_mod.PIPELINE
                                          for artifact in (other.get("carry") or {}).get(name, [])}))
        recs.append(rec)
        if rec["status"] != "ok":
            return {"status": rec["status"], "failed_stage": name, "stages": recs, "outputs": outs, "seconds": secs}
        try:
            _check_npz(Path(rec["outputs"]))
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            rec.update(status="protocol_violation", detail=str(exc)[:500])
            rec.pop("outputs", None)
            return {"status": rec["status"], "failed_stage": name, "stages": recs, "outputs": outs, "seconds": secs}
        with np.load(rec["outputs"], allow_pickle=False) as z:
            outs[name] = {k: z[k] for k in z.files}
        secs[name] = rec["timed_seconds"]
        secs[name + ".finish"] = rec.get("finish_seconds", 0.0)
        rec.pop("outputs", None)
    return {"status": "ok", "stages": recs, "outputs": outs, "seconds": secs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--stages")
    ap.add_argument("--stage")
    ap.add_argument("--submission")
    ap.add_argument("--work")
    ap.add_argument("--public")
    ap.add_argument("--models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fd", type=int)
    ap.add_argument("--params")
    a = ap.parse_args()
    if not a.child:
        ap.error("this file is a library; run selfcheck.py instead")
    return _child(a)


if __name__ == "__main__":
    sys.exit(main())

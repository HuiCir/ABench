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
seconds of G, and gives every run empty HOME / TMPDIR / compiler cache directories and no network proxy.
Labels and reference values never enter the child process: they stay with the parent (the judge or selfcheck.py).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KIT_VERSION = "abench-stage-v1"
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
    import numpy as np
    sys.path.insert(0, str(THIS.parent))
    sys.path.insert(0, str(Path(a.stages).resolve().parent))      # the case's own helper modules
    stages = load_module(Path(a.stages), "abench_case_stages")
    stage: Stage = stages.STAGES[a.stage]
    ctx = Context(stage=a.stage, submission=Path(a.submission), work=Path(a.work),
                  public=Path(a.public) if a.public else None, models=Path(a.models) if a.models else None,
                  seed=a.seed, device=a.device, params=json.loads(a.params or "{}"),
                  post_dir=Path(a.work) / "post")
    fd = a.fd
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
    os.write(fd, b"R")                                   # ---- clock starts
    state = stage.timed(ctx, state)
    ctx.sync()
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
    meta["torch"] = getattr(torch, "__version__", None)
    if torch is not None and a.device.startswith("cuda") and torch.cuda.is_available():
        meta["device_name"] = torch.cuda.get_device_name()
        meta["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    (Path(a.work) / f"{a.stage}.meta.json").write_text(json.dumps(meta, default=str)[:20000], encoding="utf-8")
    os.close(fd)
    return 0


# --------------------------------------------------------------------------- parent side
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


def run_stage(stages_path: Path, stage: str, submission: Path, work: Path, *, public: Path | None = None,
              models: Path | None = None, post_files: dict[str, Path] | None = None, seed: int = 0,
              device: str = "cuda", params: dict | None = None, timeout: float = 600.0, finish_cap: float = 60.0,
              tail_slack: float = 5.0, threads: int = 8, extra_env: dict[str, str] | None = None,
              python: str = sys.executable) -> dict[str, Any]:
    """Run one stage in a fresh process and time it with the parent's clock.

    post_files: name -> source path, copied into <work>/post/<name> only after T has been received.
    Returns a record with status, timed_seconds, finish_seconds and, if ok, the paths of the outputs.
    """
    work.mkdir(parents=True, exist_ok=True)
    post = work / "post"
    if post.exists():
        shutil.rmtree(post)
    r, w = os.pipe()
    cmd = [python, str(THIS), "--child", "--stages", str(stages_path), "--stage", stage, "--submission",
           str(submission), "--work", str(work), "--seed", str(seed), "--device", device, "--fd", str(w),
           "--params", json.dumps(params or {})]
    if public is not None:
        cmd += ["--public", str(public)]
    if models is not None:
        cmd += ["--models", str(models)]
    t_spawn = time.perf_counter()
    # stdout/stderr go to files (a pipe could fill up and block a chatty child while the parent waits on the fd)
    out_f, err_f = open(work / "_stdout.txt", "wb"), open(work / "_stderr.txt", "wb")
    env = _env(work, threads, extra_env)
    env["ABENCH_SEED"] = str(seed)                       # run seed, for submissions that only load and infer
    proc = subprocess.Popen(cmd, cwd=work, env=env, pass_fds=(w,),
                            stdin=subprocess.PIPE, stdout=out_f, stderr=err_f, start_new_session=True)
    os.close(w)
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
                left = min(left, t_go + finish_cap - time.perf_counter())
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
            if seq == b"RT" and t_go is None:
                post.mkdir(parents=True, exist_ok=True)
                for name, src in (post_files or {}).items():
                    shutil.copyfile(src, post / name)
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
        os.close(r)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        try:                                              # also kill anything the stage left behind
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if post.exists():
            shutil.rmtree(post, ignore_errors=True)       # held-out inputs never outlive the run
    t_exit = time.perf_counter()
    out_f.close()
    err_f.close()
    try:
        proc.stdin.close()
    except (BrokenPipeError, ValueError):
        pass
    stdout = (work / "_stdout.txt").read_bytes()
    stderr = (work / "_stderr.txt").read_bytes()
    rec: dict[str, Any] = {"stage": stage, "seed": seed, "exit_code": proc.returncode,
                           "stdout_tail": stdout[-1500:].decode("utf-8", "replace"),
                           "stderr_tail": stderr[-3000:].decode("utf-8", "replace")}
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
        rec["outputs"] = str(work / f"{stage}.out.npz")
        meta = work / f"{stage}.meta.json"
        if meta.is_file():
            try:
                rec["meta"] = json.loads(meta.read_text(encoding="utf-8"))   # self-reported, for reference only
            except ValueError:
                rec["meta"] = None
    return rec


def run_pipeline(stages_mod, stages_path: Path, submission: Path, work: Path, *, public: Path | None,
                 models: Path | None, held_out: Path | None, seed: int, device: str,
                 extra_env: dict[str, str] | None = None, scale: float = 1.0) -> dict[str, Any]:
    """Run all stages in stages_mod.PIPELINE for one seed. held_out is the npz of held-out inputs (no labels).

    Returns {"status", "stages": [records], "outputs": {stage: {key: array}},
             "seconds": {stage: timed_seconds, stage + ".finish": finish_seconds}}.
    scale multiplies the timeouts (selfcheck --quick uses a smaller value on smaller data; the judge uses 1.0).
    PIPELINE entries: stage, timeout, finish_cap, tail_slack, threads, held_out (bool), models (bool, default True:
    pass the frozen model directory), carry ({previous stage: [file or directory names]}).
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
                if src.is_dir():
                    shutil.copytree(src, sdir / n)
                elif src.is_file():
                    shutil.copyfile(src, sdir / n)
                else:
                    rec = {"stage": name, "seed": seed, "status": "missing_artifact",
                           "detail": f"stage {src_stage!r} did not produce {n!r}"}
                    recs.append(rec)
                    return {"status": rec["status"], "failed_stage": name, "stages": recs, "outputs": outs, "seconds": secs}
        post = {"held_out.npz": held_out} if spec.get("held_out") and held_out is not None else None
        rec = run_stage(stages_path, name, submission, sdir, public=public,
                        models=models if spec.get("models", True) else None, post_files=post,
                        seed=seed, device=device, params=getattr(stages_mod, "PARAMS", {}),
                        timeout=float(spec["timeout"]) * scale, finish_cap=float(spec.get("finish_cap", 60)),
                        tail_slack=float(spec.get("tail_slack", 5)), threads=int(spec.get("threads", 8)),
                        extra_env=extra_env)
        recs.append(rec)
        if rec["status"] != "ok":
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

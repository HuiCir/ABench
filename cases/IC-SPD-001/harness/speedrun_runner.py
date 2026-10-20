#!/usr/bin/env python3
"""IC-SPD-001 timing parent process: starts speedrun_harness.py and times with the parent's own clock.

The judge (outside the sandbox) and selfcheck.py (inside the sandbox) both call run_once() here, with exactly the same timing rules:

  - every run is a fresh process; HOME, TMPDIR and the torch.compile / Triton / CUDA cache directories point to
    empty directories owned by this run alone, so compiled results or any caches cannot be reused across runs;
  - training time = interval between receiving handshake byte R and receiving T (parent clock);
  - inference time = interval from T to P, which must not exceed predict_cap seconds;
  - the interval from T to process exit must not exceed predict_cap + tail_slack seconds, so training cannot be moved after T;
  - the handshake must be exactly the three bytes R, T, P in the right order.
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HARNESS = Path(__file__).resolve().parent / "speedrun_harness.py"


def _env(workdir: Path, threads: int, extra: dict[str, str] | None) -> dict[str, str]:
    caches = {k: workdir / k.lower() for k in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR",
                                              "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")}
    for p in caches.values():
        p.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads), "CUDA_CACHE_DISABLE": "1",
           "no_proxy": "*", "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
           "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"}
    env.update({k: str(v) for k, v in caches.items()})
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env.update(extra or {})
    return env


def run_once(submission: Path, train_npz: Path, x_test_npy: Path, seed: int, workdir: Path, *,
             entry: str = "method.py", device: str = "cuda", timeout: float = 600.0,
             predict_cap: float = 5.0, tail_slack: float = 3.0, threads: int = 8,
             extra_env: dict[str, str] | None = None, python: str = sys.executable) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "pred.npy"
    r, w = os.pipe()
    cmd = [python, str(HARNESS), "--submission", str(submission), "--entry", entry, "--train", str(train_npz),
           "--x-test", str(x_test_npy), "--out", str(out), "--seed", str(seed), "--device", device, "--fd", str(w)]
    t_spawn = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=workdir, env=_env(workdir, threads, extra_env), pass_fds=(w,),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    os.close(w)
    marks: list[tuple[bytes, float]] = []
    status, deadline = "ok", t_spawn + timeout
    try:
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                status = "time_limit_exceeded"
                break
            ready, _, _ = select.select([r], [], [], left)
            if not ready:
                continue
            chunk = os.read(r, 16)
            now = time.perf_counter()
            if not chunk:
                break
            marks.extend((bytes([b]), now) for b in chunk)
        try:
            proc.wait(timeout=max(1.0, deadline - time.perf_counter()) if status == "ok" else 1.0)
        except subprocess.TimeoutExpired:
            status = "time_limit_exceeded"
    finally:
        os.close(r)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    t_exit = time.perf_counter()
    stdout, stderr = proc.stdout.read(), proc.stderr.read()
    rec: dict[str, Any] = {"seed": seed, "exit_code": proc.returncode,
                           "stdout_tail": stdout[-1500:].decode("utf-8", "replace"),
                           "stderr_tail": stderr[-3000:].decode("utf-8", "replace")}
    seq = b"".join(m[0] for m in marks)
    if status == "ok" and proc.returncode != 0:
        status = "runtime_error"
    if status == "ok" and seq != b"RTP":
        status = "protocol_violation"
        rec["handshake"] = seq.decode("latin-1")
    if status == "ok":
        t = {m[0]: m[1] for m in marks}
        rec["train_seconds"] = round(t[b"T"] - t[b"R"], 4)
        rec["predict_seconds"] = round(t[b"P"] - t[b"T"], 4)
        rec["tail_seconds"] = round(t_exit - t[b"T"], 4)
        if rec["predict_seconds"] > predict_cap or rec["tail_seconds"] > predict_cap + tail_slack:
            status = "predict_time_exceeded"
    rec["status"] = status
    if status == "ok":
        rec["pred_path"] = str(out)
        info = Path(str(out) + ".info.json")
        if info.is_file():
            rec["child_info"] = info.read_text(encoding="utf-8")[:500]   # self-reported by the child, for reference only
    return rec

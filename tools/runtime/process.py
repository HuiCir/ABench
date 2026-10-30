"""Process-group deadlines and allocation-scoped resource metering."""
from __future__ import annotations
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

class GpuMeter:
    """Samples nvidia-smi while the agent runs and integrates GPU-busy time (a GPU counts as busy in an interval when
    its utilisation is above 5% or its memory use is 256 MiB above the level seen at start). Without nvidia-smi
    (CPU-only or Apple MPS hosts) the record says "unmetered"."""

    def __init__(self, interval: float = 5.0, gpu_ids=None):
        import threading
        self.gpu_ids = gpu_ids
        self.interval, self.busy, self.samples, self.failed, self.gpus = interval, 0.0, 0, 0, 0
        self.base: list[float] = []
        self.available = shutil.which("nvidia-smi") is not None and (gpu_ids is None or bool(gpu_ids))
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> list[tuple[float, float]]:
        ids = ["--id=" + (self.gpu_ids if isinstance(self.gpu_ids, str) else ",".join(map(str,self.gpu_ids)))] if self.gpu_ids else []
        r = subprocess.run(["nvidia-smi", *ids, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        rows = []
        for line in r.stdout.strip().splitlines():
            u, m = (float(x) for x in line.split(","))
            rows.append((u, m))
        expected = len(self.gpu_ids.split(",") if isinstance(self.gpu_ids, str) else self.gpu_ids or []) or len(rows)
        if r.returncode or not rows or len(rows) != expected:
            raise ValueError("nvidia-smi did not report every allocated GPU")
        return rows

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                rows = self._read()
            except (OSError, ValueError, subprocess.SubprocessError):
                self.failed += 1
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
        try:                                     # the meter must still work at the end of the session
            self._read()
            final_ok = True
        except (OSError, ValueError, subprocess.SubprocessError):
            final_ok = False
        attempts = self.samples + self.failed
        coverage = self.samples / attempts if attempts else 1.0
        record = {"metered": final_ok and coverage >= 0.95, "gpus": self.gpus, "interval_seconds": self.interval,
                  "samples": self.samples, "failed_samples": self.failed, "coverage": round(coverage, 4),
                  "gpu_busy_seconds": round(self.busy, 1), "gpu_busy_hours": round(self.busy / 3600, 3)}
        if not record["metered"]:
            record["reason"] = "nvidia-smi sampling failed or did not cover the session"
        return record


def budget_check(budget: dict[str, Any], gpu: dict[str, Any], agent_cpu: float | None) -> dict[str, Any]:
    """Compare metered use with oj.yaml agent_budget (gpu_hours, h100_hours, cpu_core_hours).

    over_budget is True when a metered resource exceeds its limit, None when a budgeted resource could not be
    metered (the budget is not attested), and False only when every budgeted resource was metered within limit.
    Either non-False value makes a run unscorable."""
    out: dict[str, Any] = {"budget": budget, "over_budget": False, "verified": True, "checks": []}
    for key in ("gpu_hours", "h100_hours"):                  # GPU busy hours summed over the allocated GPUs
        if budget.get(key) is None:
            continue
        if gpu.get("metered"):
            used = gpu["gpu_busy_hours"]
            ok = used <= float(budget[key]) * 1.05               # 5% tolerance for sampling granularity
            out["checks"].append({"resource": key, "used": used, "limit": budget[key], "ok": ok})
        else:
            out["checks"].append({"resource": key, "used": None, "limit": budget[key], "ok": None,
                                  "note": gpu.get("reason") or "unmetered on this host"})
    if budget.get("cpu_core_hours") is not None:
        if agent_cpu is None:
            out["checks"].append({"resource": "cpu_core_hours", "used": None, "limit": budget["cpu_core_hours"],
                                  "ok": None, "note": "container CPU accounting unavailable"})
        else:
            used = round(agent_cpu / 3600, 3)
            ok = used <= float(budget["cpu_core_hours"]) * 1.05
            out["checks"].append({"resource": "cpu_core_hours", "used": used, "limit": budget["cpu_core_hours"],
                                  "ok": ok, "note": "CPU time of the agent container (or reaped host children in smoke)"})
    results = [c["ok"] for c in out["checks"]]
    out["verified"] = None not in results
    out["over_budget"] = True if False in results else (None if None in results else False)
    return out


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

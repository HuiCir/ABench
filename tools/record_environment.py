"""Record or check the software and hardware environment of an ABench profile image (spec/env/profiles.yaml).

  python3 tools/record_environment.py --profile gpu-torch --out spec/env/lock/gpu-torch@2026.10.json   # inside the image
  python3 tools/record_environment.py --check spec/env/lock/gpu-torch@2026.10.json                     # before judging

The lock holds the Python version, every installed distribution and its version, and, when present, the versions of
torch / CUDA / cuDNN / NCCL, the GPU names and driver (nvidia-smi) and the CPU model. --check compares the software part
(Python, distributions, torch build) and the GPU model, and exits 1 on any difference; it does not compare the driver
patch level or the CPU model, which are reported only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

SCHEMA = "abench.env_lock/v1"


def _torch() -> dict | None:
    try:
        import torch
    except Exception:
        return None
    t = {"version": torch.__version__, "cuda": torch.version.cuda, "git": getattr(torch.version, "git_version", None)}
    try:
        t["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        t["cudnn"] = None
    try:
        t["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        t["nccl"] = None
    return t


def _gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return rows


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def snapshot(profile: str | None = None) -> dict:
    dists = {}
    for d in metadata.distributions():
        name = (d.metadata.get("Name") or "").lower()
        if name:
            dists[name] = d.version
    dists = dict(sorted(dists.items()))
    soft = {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "distributions": dists, "torch": _torch()}
    digest = hashlib.sha256(json.dumps(soft, sort_keys=True).encode()).hexdigest()
    return {"schema": SCHEMA, "profile": profile, "software": soft, "software_sha256": digest,
            "hardware": {"gpus": _gpus(), "cpu": _cpu(), "machine": platform.machine(), "system": platform.system()}}


def compare(lock: dict, now: dict) -> list[str]:
    diffs = []
    a, b = lock["software"], now["software"]
    for k in ("python", "implementation"):
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: lock {a.get(k)} != now {b.get(k)}")
    da, db = a.get("distributions", {}), b.get("distributions", {})
    for name in sorted(set(da) | set(db)):
        if da.get(name) != db.get(name):
            diffs.append(f"package {name}: lock {da.get(name)} != now {db.get(name)}")
    if (a.get("torch") or {}) != (b.get("torch") or {}):
        diffs.append(f"torch build: lock {a.get('torch')} != now {b.get('torch')}")
    ga = sorted(g["name"] for g in lock["hardware"].get("gpus", []))
    gb = sorted(g["name"] for g in now["hardware"].get("gpus", []))
    if ga != gb:
        diffs.append(f"GPUs: lock {ga} != now {gb}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile")
    ap.add_argument("--out")
    ap.add_argument("--check")
    a = ap.parse_args()
    now = snapshot(a.profile)
    if a.check:
        lock = json.loads(Path(a.check).read_text())
        diffs = compare(lock, now)
        print(json.dumps({"lock": a.check, "profile": lock.get("profile"), "matches": not diffs, "differences": diffs},
                         indent=2))
        return 1 if diffs else 0
    text = json.dumps(now, indent=2, sort_keys=False) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"wrote {a.out} (software_sha256 {now['software_sha256'][:16]}...)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

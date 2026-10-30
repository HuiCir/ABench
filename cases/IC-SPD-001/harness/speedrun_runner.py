#!/usr/bin/env python3
"""Compatibility entry point for IC-SPD-001; delegates to the shared stage protocol."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "speedrun_harness.py"
sys.path.insert(0, str(HERE))
from abench_stage import run_stage


def run_once(submission, train_npz, x_test_npy, seed, workdir, *, entry="method.py", device="cuda",
             timeout=600., predict_cap=5., tail_slack=3., threads=8, extra_env=None, python=sys.executable):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    held = workdir.parent / (workdir.name + ".inputs.npz")
    np.savez(held, x=np.load(x_test_npy, allow_pickle=False))
    try:
        r = run_stage(HERE / "stages.py", "train", Path(submission), workdir,
                      public=Path(train_npz).parent, pre_files={"held_pre.npz": held},
                      seed=seed, device=device, params={"entry": entry}, timeout=timeout,
                      finish_cap=predict_cap, tail_slack=tail_slack, slack_after_p=True, threads=threads,
                      extra_env=extra_env, python=python)
    finally:
        held.unlink(missing_ok=True)
    if r["status"] == "finish_time_exceeded":
        r["status"] = "predict_time_exceeded"
    for old, new in (("timed_seconds", "train_seconds"), ("finish_seconds", "predict_seconds")):
        if old in r:
            r[new] = r[old]
    if r["status"] == "ok":
        out = workdir / "pred.npy"
        with np.load(r.pop("outputs"), allow_pickle=False) as z:
            np.save(out, z["pred"], allow_pickle=False)
        r["pred_path"] = str(out)
    return r

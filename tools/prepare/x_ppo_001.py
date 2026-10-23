"""X-PPO-001: no download. The held-out set is 20 initial states per task, drawn from each task's reset distribution
(cases/X-PPO-001/harness/control_envs.py, gymnasium 1.0.0 dynamics, MIT) with a secret seed."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from prepare.common import Frozen

HARNESS = Path(__file__).resolve().parents[2] / "cases" / "X-PPO-001" / "harness"
SECRET_SEED_ENV = "ABENCH_X_PPO_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
SOURCES = [{"name": "gymnasium classic control (re-implemented)", "generator": "cases/X-PPO-001/harness/control_envs.py",
            "url": "https://github.com/Farama-Foundation/Gymnasium/tree/9938990977eb5e85ea4523ecf5cb2678c1708ba4",
            "license": "MIT"}]


def _stages():
    sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("x_ppo_stages", HARNESS / "stages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (the seed of the held-out initial states)")
    return Frozen(public={}, hidden=_stages().held_out_states(seed),
                  sources_sha256={"generator": "control_envs.py", "seed": "secret (operator)"})


def synthetic(n: int) -> Frozen:
    return Frozen(public={}, hidden=_stages().held_out_states(7, max(2, min(n, 20))), sources_sha256={"synthetic": f"n{n}"})

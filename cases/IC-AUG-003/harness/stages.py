"""IC-AUG-003 pipeline: the evaluator trains the fixed CIFAR-10 recipe (recipe.py) with your augmentation policy and
measures test accuracy (evaluator and self-check use this same file).

One stage, "train" (imports policy.py after checking that it uses only the primitives of aug_ops.py):
  setup   load the training images to the GPU (untimed)
  timed   check and import policy.py; train the fixed recipe for 40 epochs, augmenting the whole training set with
          policy.augment at the start of every epoch
  finish  class probabilities for the CIFAR-10 test images and the CIFAR-10.1 images (with mirror averaging)
The evaluator runs the flip-and-translate baseline (baselines/flipcrop) through the same stage on the same seeds.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

PARAMS = {"epochs": 40, "allowed_imports": ["math", "typing", "__future__"]}
HELD_OUT_KEYS = ["x_test", "x_101"]
PIPELINE = [{"stage": "train", "timeout": 900, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
FORBIDDEN_NAMES = {"torch", "numpy", "np", "os", "sys", "open", "exec", "eval", "compile", "__import__", "globals",
                   "locals", "getattr", "setattr", "vars", "importlib", "subprocess"}


def policy_violations(path: Path) -> list[str]:
    """policy.py may import only math/typing and may not name torch, numpy, files or reflection; tensor work goes
    through the aug_ops functions it is given."""
    out = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"import {a.name}" for a in node.names if a.name.split(".")[0] not in PARAMS["allowed_imports"]]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in PARAMS["allowed_imports"]:
                out.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            out.append(f"attribute {node.attr}")
    return sorted(set(out))


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    import torch
    from recipe import to01
    with np.load(ctx.public / "train.npz", allow_pickle=False) as z:
        x, y = z["x"], z["y"]
        keep = np.ones(len(y), bool)
        if "exclude" in z.files:                       # self-check: the dev images are held out of training
            keep[z["exclude"]] = False
    return {"x": to01(x[keep], ctx.device), "y": torch.from_numpy(y[keep].astype(np.int64)).to(ctx.device)}


def _timed(ctx: Context, st):
    import aug_ops
    from recipe import train
    bad = policy_violations(ctx.submission / "policy.py")
    if bad:
        raise SystemExit(f"policy.py uses more than the aug_ops primitives: {bad}")
    mod = ctx.import_submission("policy.py", required=("augment",))
    epochs = int(_overrides(ctx).get("epochs", PARAMS["epochs"]))
    model, mean, std = train(st["x"], st["y"], mod.augment, aug_ops, ctx.seed, ctx.device, epochs)
    return {"model": model, "mean": mean, "std": std}


def _finish(ctx: Context, st):
    from recipe import predict, to01
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for k in ("x_test", "x_101"):
            if k in z.files:
                out[f"proba_{k[2:]}"] = predict(st["model"], st["mean"], st["std"], to01(z[k], ctx.device)).cpu().numpy()
    return out


STAGES = {"train": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the 5,000 images in dev_idx are held out of training and play the test role
    (no CIFAR-10.1 counterpart); --quick trains 8 epochs."""
    with np.load(public / "train.npz", allow_pickle=False) as z:
        x, y, dev = z["x"], z["y"], z["dev_idx"]
    pub = out / "public"
    pub.mkdir(parents=True, exist_ok=True)
    np.savez(pub / "train.npz", x=x, y=y, exclude=dev)
    if quick:
        (pub / "overrides.json").write_text(json.dumps({"epochs": 8}))
    np.savez(out / "held_out.npz", x_test=x[dev])
    return {"public": pub, "held_out": out / "held_out.npz", "reference": {"y_test": y[dev]}, "time_scale": 1.0}

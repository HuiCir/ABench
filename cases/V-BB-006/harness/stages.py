"""V-BB-006 pipeline: the evaluator trains your vision-transformer backbone at two compute budgets on Imagenette with a fixed
recipe and measures test accuracy (evaluator and self-check use this same file).

For each variant v in VARIANTS ("small", "large"), one stage "train_<v>" (imports model.py):
  setup   load the training images (untimed)
  timed   build(v, num_classes=10) with the seed; train with RECIPE (img_train.py:
          AdamW, warm-up and cosine decay, label smoothing, clipping, random resized crops and flips at 160 pixels)
  finish  forward FLOPs per image at 160 pixels (FlopCounterMode) against the variant's budget; the test images arrive
          (held_out.npz) and their logits are returned
The evaluator runs the ViT baseline (baselines/vit: ViT-Ti/16 and ViT-S/16) through the same stages and seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import img_train

RES = 160
VARIANTS = {"small": 1.4e9, "large": 4.8e9}              # forward FLOPs per 160 x 160 image
RECIPE = {"resolution": RES, "epochs": 30, "batch": 128, "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.05,
          "warmup_epochs": 5, "label_smoothing": 0.1, "grad_clip": 1.0}
PARAMS = {"recipe": RECIPE, "flops_budgets": VARIANTS, "classes": 10}
HELD_OUT_KEYS = ["x_test"]
PIPELINE = [{"stage": f"train_{v}", "timeout": 2400 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False} for v in VARIANTS]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    with np.load(ctx.public / "train.npz", allow_pickle=False) as z:
        return {"x": z["x"], "y": z["y"], "ov": _overrides(ctx)}


def _make_timed(variant: str):
    def _timed(ctx: Context, st):
        import torch
        build = ctx.import_submission("model.py", required=("build",)).build
        torch.manual_seed(ctx.seed)
        model = build(variant, num_classes=10).to(ctx.device).to(memory_format=torch.channels_last)
        recipe = dict(RECIPE, **st["ov"].get("recipe", {}))
        res = img_train.train(model, st["x"], st["y"], recipe, ctx.seed, ctx.device)
        return {"model": model, "res": res}
    return _timed


def _make_finish(variant: str):
    def _finish(ctx: Context, st):
        model, res = st["model"], st["res"]
        flops = img_train.forward_flops(model, RES, ctx.device)
        with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
            logits = img_train.predict(model, z["x_test"], RES, ctx.device)
        return {"logits": logits, "flops": np.asarray([flops]), "flops_ratio": np.asarray([flops / VARIANTS[variant]]),
                "diverged": np.asarray([float(res["diverged"])]), "curve": res["curve"],
                "params": np.asarray([float(sum(p.numel() for p in model.parameters()))])}
    return _finish


STAGES = {f"train_{v}": Stage(setup=_setup, timed=_make_timed(v), finish=_make_finish(v)) for v in VARIANTS}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: 1,500 training images (150 per class) play the test role; --quick trains 2 epochs."""
    with np.load(public / "train.npz", allow_pickle=False) as z:
        x, y = z["x"], z["y"]
    rng = np.random.default_rng(2015)
    test = np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(150, int((y == c).sum()) // 3), replace=False)
                           for c in range(10)])
    keep = np.setdiff1d(np.arange(len(y)), test)
    (out / "public").mkdir(parents=True, exist_ok=True)
    np.savez(out / "public" / "train.npz", x=x[keep], y=y[keep])
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"recipe": {"epochs": 2, "warmup_epochs": 0.4}}))
    np.savez(out / "held_out.npz", x_test=x[test])
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {"y_test": y[test]},
            "time_scale": 1.0}

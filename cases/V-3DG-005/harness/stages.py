"""V-3DG-005 pipeline: for each held-out scene your method is fitted to 24 training views within a time budget and
renders the test cameras (evaluator and self-check use this same file).

Stages "scene0" ... "scene5" (imports method.py), one per held-out Mip-NeRF 360 scene:
  setup   the scene's training views (images, K, w2c) arrive in ctx.pre_dir (held_pre.npz) (untimed)
  timed   state = fit(views, budget_seconds, device), at most BUDGET seconds
  finish  the test cameras arrive (held_out.npz); render(state, cameras) -> uint8 images             (≤ 300 s)
The evaluator runs plain 3D Gaussian splatting (baselines/gs3d) through the same stages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import nvs
from abench_stage import Context, Stage

N_SCENES, BUDGET = 6, 600.0
SCENES = [f"scene{i}" for i in range(N_SCENES)]
PARAMS = {"budget_seconds": BUDGET, "n_train_views": nvs.N_TRAIN, "test_every": nvs.HOLD_EVERY, "resolution": "images_8"}
HELD_OUT_PRE_KEYS = [f"{s}_train_{k}" for s in SCENES for k in ("images", "K", "w2c")]
HELD_OUT_KEYS = [f"{s}_test_{k}" for s in SCENES for k in ("K", "w2c", "hw")]
PIPELINE = [{"stage": s, "timeout": int(BUDGET) + 60 + 300, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out_pre": True, "held_out": True, "models": False} for s in SCENES]


def _budget(ctx: Context) -> float:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return float(json.loads(p.read_text()).get("budget_seconds", BUDGET)) if p and p.is_file() else BUDGET


def _setup(ctx: Context):
    with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
        views = {k: z[f"{ctx.stage}_train_{k}"] for k in ("images", "K", "w2c")}
    return {"views": views, "budget": _budget(ctx)}


def _timed(ctx: Context, st):
    mod = ctx.import_submission("method.py", required=("fit", "render"))
    return {"mod": mod, "state": mod.fit(st["views"], st["budget"], ctx.device)}


def _finish(ctx: Context, st):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        cams = {"K": z[f"{ctx.stage}_test_K"], "w2c": z[f"{ctx.stage}_test_w2c"]}
        h, w = (int(v) for v in z[f"{ctx.stage}_test_hw"])
    cams["height"], cams["width"] = h, w
    img = np.asarray(st["mod"].render(st["state"], cams))
    if img.shape != (len(cams["K"]), h, w, 3):
        raise SystemExit(f"render() returned {img.shape}, expected {(len(cams['K']), h, w, 3)}")
    return {"images": np.clip(img, 0, 255).astype(np.uint8)}


STAGES = {s: Stage(setup=_setup, timed=_timed, finish=_finish) for s in SCENES}


def scene_arrays(sc: dict, name: str, train, test) -> tuple:
    """(pre inputs, post inputs, reference) of one scene from {"images", "K", "w2c"}."""
    pre = {f"{name}_train_images": sc["images"][train], f"{name}_train_K": sc["K"][train],
           f"{name}_train_w2c": sc["w2c"][train]}
    post = {f"{name}_test_K": sc["K"][test], f"{name}_test_w2c": sc["w2c"][test],
            f"{name}_test_hw": np.asarray(sc["images"].shape[1:3], np.int64)}
    return pre, post, {f"{name}_test_images": sc["images"][test]}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the three public scenes, each used twice (test views at offsets 0 and 4 of every 8) as scene0-5.
    --quick: a 60 s budget per scene."""
    names = sorted(p.stem for p in public.glob("dev_*.npz"))
    pre, post, ref = {}, {}, {}
    for i, s in enumerate(SCENES):
        with np.load(public / f"{names[i % len(names)]}.npz", allow_pickle=False) as z:
            sc = {k: z[k] for k in ("images", "K", "w2c")}
        tr, te = nvs.split(len(sc["images"]), offset=0 if i < len(names) else 4)
        a, b, c = scene_arrays(sc, s, tr, te)
        pre.update(a)
        post.update(b)
        ref.update(c)
    pub = out / "public"
    pub.mkdir(parents=True, exist_ok=True)
    if quick:
        (pub / "overrides.json").write_text(json.dumps({"budget_seconds": 60}))
    np.savez(out / "held_pre.npz", **pre)
    np.savez(out / "held_out.npz", **post)
    return {"public": pub, "held_out": out / "held_out.npz", "held_pre": out / "held_pre.npz", "reference": ref,
            "time_scale": 1.0}

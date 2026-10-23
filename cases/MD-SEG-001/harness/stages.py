"""MD-SEG-001 pipeline: the evaluator trains your 3D segmentation network with the fixed recipe of seg3d.py on four
Medical Segmentation Decathlon tasks, one stage per task, and segments their held-out volumes (evaluator and
self-check use this same file).

Stages "hippocampus", "prostate", "heart" and "unseen" (imports network.py):
  setup   load the task's preprocessed training volumes: the public files for the first three; for "unseen" they and
          its configuration arrive in ctx.pre_dir (held_pre.npz) (untimed)
  timed   net = make_network(in_channels, num_classes, patch_size, spacing); seg3d.train with the task's iterations
  finish  sliding-window segmentation of the task's held-out volumes (given after the timed phase); parameters,
          peak GPU memory, divergence                                                         (≤ 900 s per task)
The evaluator trains the nnU-Net-style U-Net of baselines/unet through the same stages on the same seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import seg3d as S
from abench_stage import Context, Stage

KNOWN = ("hippocampus", "prostate", "heart")
STAGE_NAMES = KNOWN + ("unseen",)
MAX_PARAMS = 64_000_000
PARAMS = {"tasks": {t: {k: v for k, v in S.TASKS[t].items()} for t in KNOWN}, "max_params": MAX_PARAMS,
          "recipe": {"lr": S.LR, "momentum": S.MOMENTUM, "weight_decay": S.WD, "clip": S.CLIP, "fg_fraction": S.FG_FRACTION}}
HELD_OUT_KEYS = [f"{t}_test_{k}" for t in STAGE_NAMES for k in ("img", "img_shape")]
HELD_OUT_PRE_KEYS = ["unseen_train_img", "unseen_train_img_shape", "unseen_train_seg", "unseen_train_seg_shape",
                     "unseen_config"]
PIPELINE = [{"stage": t, "timeout": 1800 + 900, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "held_out_pre": t == "unseen", "models": False} for t in STAGE_NAMES]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def config(ctx: Context) -> dict:
    if ctx.stage != "unseen":
        return dict(S.TASKS[ctx.stage])
    with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
        cfg = json.loads(str(z["unseen_config"][0]))
    cfg["spacing"], cfg["patch"] = tuple(cfg["spacing"]), tuple(cfg["patch"])
    return cfg


def _volumes(ctx: Context) -> list:
    if ctx.stage == "unseen":
        with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
            imgs = S.unpack(z["unseen_train_img"], z["unseen_train_img_shape"])
            segs = S.unpack(z["unseen_train_seg"], z["unseen_train_seg_shape"])
    else:
        with np.load(ctx.public / f"{ctx.stage}.npz", allow_pickle=False) as z:
            imgs, segs = S.unpack(z["img"], z["img_shape"]), S.unpack(z["seg"], z["seg_shape"])
    return [(i.astype(np.float32), s.astype(np.uint8)) for i, s in zip(imgs, segs)]


def _setup(ctx: Context):
    return {"cfg": config(ctx), "vols": _volumes(ctx), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    cfg = st["cfg"]
    mod = ctx.import_submission("network.py", required=("make_network",))
    torch.manual_seed(ctx.seed)
    net = mod.make_network(in_channels=cfg["channels"], num_classes=cfg["classes"], patch_size=tuple(cfg["patch"]),
                           spacing=tuple(cfg["spacing"])).to(ctx.device)
    res = S.train(net, st["vols"], cfg, ctx.seed, ctx.device, st["ov"].get("iterations"))
    return {"net": net, "res": res, "cfg": cfg}


def _finish(ctx: Context, st):
    net, res, cfg = st["net"], st["res"], st["cfg"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        imgs = S.unpack(z[f"{ctx.stage}_test_img"], z[f"{ctx.stage}_test_img_shape"])
    n_par = sum(p.numel() for p in net.parameters())
    out = {"params": np.asarray([n_par], np.int64), "diverged": np.asarray([res["diverged"]]),
           "peak_mem_gb": np.asarray([res["peak_mem_gb"]]), "curve": res["curve"]}
    net.eval()
    f = S.predictor(net, ctx.device)
    preds = [S.sliding_window(f, im.astype(np.float32), cfg["patch"], cfg["classes"]) for im in imgs]
    out["pred"], out["pred_shape"] = S.pack(preds)
    out["pred"] = out["pred"].astype(np.uint8)
    return out


STAGES = {t: Stage(setup=_setup, timed=_timed, finish=_finish) for t in STAGE_NAMES}


def split_dev(imgs: list, segs: list, frac: float = 0.2, seed: int = 1809) -> tuple:
    idx = np.random.default_rng(seed).permutation(len(imgs))
    k = max(1, int(round(frac * len(imgs))))
    return [i for i in idx[k:]], [i for i in idx[:k]]


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: 20% of each public task's training volumes are held out; the "unseen" stage is played by the
    hippocampus volumes with both structures merged into one class (another patch size). --quick: 100 iterations."""
    pub = out / "public"
    pub.mkdir(parents=True, exist_ok=True)
    held, pre, ref = {}, {}, {}
    hip = None
    for t in KNOWN:
        with np.load(public / f"{t}.npz", allow_pickle=False) as z:
            imgs, segs = S.unpack(z["img"], z["img_shape"]), S.unpack(z["seg"], z["seg_shape"])
        tr, te = split_dev(imgs, segs)
        a, b = S.pack([imgs[i] for i in tr]), S.pack([segs[i] for i in tr])
        np.savez(pub / f"{t}.npz", img=a[0], img_shape=a[1], seg=b[0], seg_shape=b[1])
        held[f"{t}_test_img"], held[f"{t}_test_img_shape"] = S.pack([imgs[i] for i in te])
        ref[f"{t}_test_seg"], ref[f"{t}_test_seg_shape"] = S.pack([segs[i] for i in te])
        if t == "hippocampus":
            hip = (imgs, [(s > 0).astype(np.uint8) for s in segs], tr, te)
    imgs, segs, tr, te = hip
    cfg = {"channels": 1, "classes": 2, "spacing": [1.0, 1.0, 1.0], "patch": [48, 48, 48], "batch": 8,
           "iterations": 2500, "nsd_mm": 1.0, "modality": "MRI"}
    a, b = S.pack([imgs[i] for i in tr]), S.pack([segs[i] for i in tr])
    pre = {"unseen_train_img": a[0], "unseen_train_img_shape": a[1], "unseen_train_seg": b[0],
           "unseen_train_seg_shape": b[1], "unseen_config": np.asarray([json.dumps(cfg)])}
    held["unseen_test_img"], held["unseen_test_img_shape"] = S.pack([imgs[i] for i in te])
    ref["unseen_test_seg"], ref["unseen_test_seg_shape"] = S.pack([segs[i] for i in te])
    ref["unseen_config"] = pre["unseen_config"]
    if quick:
        (pub / "overrides.json").write_text(json.dumps({"iterations": 100}))
    np.savez(out / "held_out.npz", **held)
    np.savez(out / "held_pre.npz", **pre)
    return {"public": pub, "held_out": out / "held_out.npz", "held_pre": out / "held_pre.npz", "reference": ref,
            "time_scale": 1.0}

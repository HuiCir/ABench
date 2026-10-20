"""T-OPT-002 pipeline: train the fixed ViT (vit_model.py) on CIFAR-10 with your update rule, then classify the test set.

Stage "train"  (imports optimizer.py)
  setup   load train.npz to the GPU, build the model with the seed (untimed)
  timed   import optimizer.py and hparam_space.yaml, build RULE with the declared defaults, train STEPS steps
  finish  save model.pt; report the loss curve, divergence and optimizer-state size
Stage "eval"   (does not import your code; receives only model.pt)
  timed   load model.pt
  finish  predicted classes for the held-out test images (labels stay with the evaluator)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

PARAMS = {"epochs": 30, "batch": 512, "warmup": 200, "final_lr_fraction": 0.0, "lr_1d_ratio": 0.15,
          "label_smoothing": 0.1, "augment": "random crop (4-pixel reflect padding) + horizontal flip",
          "precision": "bf16 autocast, fp32 weights and optimizer arithmetic", "state_numel_cap_ratio": 3.0}
TRAIN_TIMEOUT = 1200.0
HELD_OUT_KEYS = ["x"]
PIPELINE = [
    {"stage": "train", "timeout": TRAIN_TIMEOUT + 120, "finish_cap": 60, "tail_slack": 10, "threads": 8,
     "held_out": False, "models": False},
    {"stage": "eval", "timeout": 300, "finish_cap": 60, "tail_slack": 10, "threads": 8, "held_out": True,
     "models": False, "carry": {"train": ["model.pt"]}},
]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup_train(ctx: Context):
    import torch
    from vit_model import ViT
    prm = dict(PARAMS, **_overrides(ctx))
    with np.load(ctx.public / "train.npz", allow_pickle=False) as z:
        x, y = z["x"], z["y"].astype(np.int64)
    torch.manual_seed(ctx.seed)
    model = ViT().to(ctx.device)
    return {"prm": prm, "x": torch.from_numpy(x).to(ctx.device), "y": torch.from_numpy(y).to(ctx.device), "model": model}


def _schedule(step: int, total: int, prm: dict) -> float:
    if step < prm["warmup"]:
        return (step + 1) / prm["warmup"]
    t = (step - prm["warmup"]) / max(1, total - prm["warmup"])
    lo = prm["final_lr_fraction"]
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def _augment(xb, gen):
    """Random 32×32 crop of the 4-pixel reflect-padded image, then a random horizontal flip (per image, seeded)."""
    import torch
    import torch.nn.functional as F
    B = xb.shape[0]
    xp = F.pad(xb, (4, 4, 4, 4), mode="reflect")                      # (B, 3, 40, 40)
    i = torch.randint(0, 9, (B,), device=xb.device, generator=gen)
    j = torch.randint(0, 9, (B,), device=xb.device, generator=gen)
    ar = torch.arange(32, device=xb.device)
    rows = (i[:, None] + ar[None, :])[:, None, :, None].expand(B, 3, 32, 40)
    out = xp.gather(2, rows)                                           # (B, 3, 32, 40)
    cols = (j[:, None] + ar[None, :])[:, None, None, :].expand(B, 3, 32, 32)
    out = out.gather(3, cols)                                          # (B, 3, 32, 32)
    flip = torch.rand(B, device=xb.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def _timed_train(ctx: Context, st):
    import torch
    import torch.nn.functional as F
    from optim_api import make_torch_optimizer, state_numel, Ops
    from vit_model import to_input
    prm, model, X, Y = st["prm"], st["model"], st["x"], st["y"]
    rule, lr = load_rule(ctx.submission)
    two_d = [p for p in model.parameters() if p.ndim >= 2]
    one_d = [p for p in model.parameters() if p.ndim < 2]
    opt = make_torch_optimizer(rule, [{"params": two_d, "lr": lr, "base_lr": lr},
                                      {"params": one_d, "lr": lr * prm["lr_1d_ratio"], "base_lr": lr * prm["lr_1d_ratio"]}], lr)
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(ctx.seed)
    n, B = len(Y), prm["batch"]
    per_epoch = n // B
    total = prm["epochs"] * per_epoch
    curve, diverged, step = [], 0, 0
    t0 = time.perf_counter()
    for _ in range(prm["epochs"]):
        perm = torch.randperm(n, device=ctx.device, generator=gen)
        for b in range(per_epoch):
            if not rule.owns_schedule:
                f = _schedule(step, total, prm)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
            idx = perm[b * B:(b + 1) * B]
            xb = _augment(to_input(X[idx]), gen)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
                loss = F.cross_entropy(model(xb).float(), Y[idx], label_smoothing=prm["label_smoothing"])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                v = float(loss.detach())
                curve.append((step, v))
                if not math.isfinite(v) or v > 20:
                    diverged = 1
            step += 1
            if diverged:
                break
        if diverged:
            break
    ctx.meta["train_wall_seconds_self_reported"] = round(time.perf_counter() - t0, 3)
    n_params = sum(p.numel() for p in model.parameters())
    n_state = sum(state_numel(s.get("rule_state", {}), Ops("torch")) for s in opt.state.values())
    return {"curve": curve, "diverged": diverged, "state_ratio": n_state / n_params, "model": model}


def load_rule(submission: Path):
    """RULE from optimizer.py with the defaults declared in hparam_space.yaml (no tuning by the evaluator)."""
    import importlib.util
    import sys
    import yaml
    sys.path.insert(0, str(submission))
    spec = importlib.util.spec_from_file_location("abench_submission_optimizer", submission / "optimizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = yaml.safe_load((submission / "hparam_space.yaml").read_text(encoding="utf-8"))
    tun = {k: v["default"] for k, v in (hs.get("tunable") or {}).items()}
    if len(tun) > 4:
        raise SystemExit("hparam_space.yaml declares more than 4 tunable hyperparameters")
    lr = float(tun.pop("lr"))
    rule = mod.RULE(**dict(hs.get("fixed") or {}, **tun))
    if bool(hs.get("owns_schedule", False)):
        rule.owns_schedule = True
    return rule, lr


def _finish_train(ctx: Context, st):
    import torch
    torch.save({k: v.detach().cpu() for k, v in st["model"].state_dict().items()}, ctx.work / "model.pt")
    return {"curve": np.asarray(st["curve"], dtype=np.float64).reshape(-1, 2),
            "diverged": np.asarray([st["diverged"]]), "state_ratio": np.asarray([st["state_ratio"]])}


def _timed_eval(ctx: Context, _):
    import torch
    from vit_model import ViT
    model = ViT().to(ctx.device)
    model.load_state_dict(torch.load(ctx.work / "model.pt", map_location=ctx.device, weights_only=True))
    return model.eval()


def _finish_eval(ctx: Context, model):
    import torch
    from vit_model import to_input
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        x = torch.from_numpy(z["x"]).to(ctx.device)
    preds = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for i in range(0, len(x), 1000):
            preds.append(model(to_input(x[i:i + 1000])).float().argmax(1).cpu())
    return {"pred": torch.cat(preds).numpy().astype(np.int64)}


STAGES = {"train": Stage(setup=_setup_train, timed=_timed_train, finish=_finish_train),
          "eval": Stage(timed=_timed_eval, finish=_finish_eval)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: train on train.npz minus dev_idx, classify dev_idx; --quick trains 3 epochs."""
    with np.load(public / "train.npz", allow_pickle=False) as z:
        x, y, dev = z["x"], z["y"], z["dev_idx"]
    mask = np.ones(len(y), bool)
    mask[dev] = False
    (out / "public").mkdir(parents=True, exist_ok=True)
    np.savez(out / "public" / "train.npz", x=x[mask], y=y[mask])
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"epochs": 3, "warmup": 50}))
    np.savez(out / "held_out.npz", x=x[dev])
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {"y": y[dev]}, "time_scale": 1.0}

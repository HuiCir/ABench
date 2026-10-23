"""B-OPT-002 pipeline: a program in the primitive language (primitive_dsl.py) is checked for novelty and stability on
a fixed gradient stream, then used to train the fixed language model of B-OPT-001, whose held-out loss is measured
(evaluator and self-check use this same file).

Stage "screen"  (reads program.yaml; CPU work)
  timed   parse and validate program.yaml (no Python from the submission is imported or run)
  finish  behaviour.screen on a gradient stream drawn from the held-out screen seed: similarity to known optimizers,
          descent direction, finiteness at gradient scales 1e-6 and 1e6
Stage "train"  (reads program.yaml)
  setup   map the training token shards, build the model on the GPU with the seed (untimed)
  timed   compile the program into a FunctionalRule and train STEPS steps with its learning rate (B-OPT-001 recipe)
  finish  save model.pt; report the loss curve, divergence and the size of the optimizer state
Stage "eval"   (does not read your program; receives only model.pt)
  timed   load model.pt
  finish  mean next-token loss on the held-out tokens
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

PARAMS = {"steps": 3000, "batch": 32, "seq": 1024, "warmup": 150, "final_lr_fraction": 0.1, "lr_1d_ratio": 0.15,
          "precision": "bf16 autocast, fp32 weights and optimizer arithmetic", "state_numel_cap_ratio": 3.0,
          "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"], "eval_batch": 16}
TRAIN_TIMEOUT = 1500.0
HELD_OUT_KEYS = ["tokens", "stream_seed"]
PIPELINE = [
    {"stage": "screen", "timeout": 120, "finish_cap": 300, "tail_slack": 10, "threads": 8, "held_out": True,
     "models": False},
    {"stage": "train", "timeout": TRAIN_TIMEOUT + 120, "finish_cap": 60, "tail_slack": 10, "threads": 8,
     "held_out": False, "models": False},
    {"stage": "eval", "timeout": 600, "finish_cap": 180, "tail_slack": 10, "threads": 8, "held_out": True,
     "models": False, "carry": {"train": ["model.pt"]}},
]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's dev split
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup_train(ctx: Context):
    import torch
    from lm_model import LM, Config
    from token_data import read_shard
    prm = dict(PARAMS, **_overrides(ctx))
    toks = [read_shard(ctx.public / s) for s in prm["train_shards"] if (ctx.public / s).is_file()]
    stream = np.concatenate([np.asarray(t) for t in toks])
    torch.manual_seed(ctx.seed)
    model = LM(Config()).to(ctx.device)
    return {"prm": prm, "stream": stream, "model": model}


def _schedule(step: int, prm: dict) -> float:
    if step < prm["warmup"]:
        return (step + 1) / prm["warmup"]
    t = (step - prm["warmup"]) / max(1, prm["steps"] - prm["warmup"])
    lo = prm["final_lr_fraction"]
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def load_rule(submission: Path):
    """The submission's program.yaml compiled into a FunctionalRule, and its declared learning rate."""
    from primitive_dsl import load_program
    return load_program(submission / "program.yaml")


def _timed_screen(ctx: Context, _):
    from primitive_dsl import ProgramRule
    import yaml
    prog = yaml.safe_load((ctx.submission / "program.yaml").read_text(encoding="utf-8"))
    rule = ProgramRule(prog)
    return {"rule": rule, "lines": len(rule.v["lines"]), "calls": rule.v["calls"], "state": len(rule.v["state"])}


def _finish_screen(ctx: Context, st):
    from behaviour import screen
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        seed = int(z["stream_seed"][0])
    r = screen(st["rule"], seed)
    return {"max_similarity": np.asarray([r["max_similarity"]]), "descent_cosine": np.asarray([r["descent_cosine"]]),
            "finite": np.asarray([r["finite"]]), "stable_extreme_scales": np.asarray([r["stable_extreme_scales"]]),
            "closest": np.asarray([r["closest"]], dtype=np.str_),
            "program_size": np.asarray([st["lines"], st["calls"], st["state"]], np.int64)}


def _timed_train(ctx: Context, st):
    import torch
    from optim_api import make_torch_optimizer, state_numel, Ops
    prm, model, stream = st["prm"], st["model"], st["stream"]
    rule, lr = load_rule(ctx.submission)
    two_d = [p for n, p in model.named_parameters() if p.ndim >= 2]
    one_d = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = make_torch_optimizer(rule, [{"params": two_d, "lr": lr, "base_lr": lr},
                                      {"params": one_d, "lr": lr * prm["lr_1d_ratio"], "base_lr": lr * prm["lr_1d_ratio"]}], lr)
    rng = np.random.default_rng(ctx.seed)
    B, T = prm["batch"], prm["seq"]
    curve, diverged = [], 0
    t0 = time.perf_counter()
    for step in range(prm["steps"]):
        if not rule.owns_schedule:
            f = _schedule(step, prm)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
        off = rng.integers(0, len(stream) - T - 1, B)
        chunk = np.stack([stream[o:o + T + 1] for o in off]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1]).to(ctx.device, non_blocking=True)
        y = torch.from_numpy(chunk[:, 1:]).to(ctx.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
            loss = model.loss(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == prm["steps"] - 1:
            v = float(loss.detach())
            curve.append((step, v))
            if not math.isfinite(v) or v > 20:
                diverged = 1
                break
    ctx.meta["train_wall_seconds_self_reported"] = round(time.perf_counter() - t0, 3)
    n_params = sum(p.numel() for p in model.parameters())
    n_state = sum(state_numel(s.get("rule_state", {}), Ops("torch")) for s in opt.state.values())
    return {"curve": curve, "diverged": diverged, "state_ratio": n_state / n_params, "model": model}


def _finish_train(ctx: Context, st):
    import torch
    torch.save({k: v.detach().cpu() for k, v in st["model"].state_dict().items()}, ctx.work / "model.pt")
    c = np.asarray(st["curve"], dtype=np.float64).reshape(-1, 2)
    return {"curve": c, "diverged": np.asarray([st["diverged"]]), "state_ratio": np.asarray([st["state_ratio"]])}


def _timed_eval(ctx: Context, _):
    import torch
    from lm_model import LM, Config
    model = LM(Config()).to(ctx.device)
    model.load_state_dict(torch.load(ctx.work / "model.pt", map_location=ctx.device, weights_only=True))
    model.eval()
    return model


def _finish_eval(ctx: Context, model):
    import torch
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = z["tokens"].astype(np.int64)
    T, bs = PARAMS["seq"], PARAMS["eval_batch"]
    n = (len(toks) - 1) // T
    x_all = torch.from_numpy(toks[: n * T].reshape(n, T))
    y_all = torch.from_numpy(toks[1: n * T + 1].reshape(n, T))
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for i in range(0, n, bs):
            x, y = x_all[i:i + bs].to(ctx.device), y_all[i:i + bs].to(ctx.device)
            loss = model.loss(x, y)
            total += float(loss) * y.numel()
            count += y.numel()
    return {"val_loss": np.asarray([total / max(count, 1)]), "val_tokens": np.asarray([count])}


STAGES = {"screen": Stage(timed=_timed_screen, finish=_finish_screen),
          "train": Stage(setup=_setup_train, timed=_timed_train, finish=_finish_train),
          "eval": Stage(timed=_timed_eval, finish=_finish_eval)}


PUBLIC_STREAM_SEED = 17       # the self-check's gradient stream; the evaluator's seed is secret


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out; --quick trains
    300 steps on the first shard only."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    if not quick:
        write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    else:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": 300, "warmup": 30}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]), stream_seed=np.asarray([PUBLIC_STREAM_SEED], np.int64))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

"""B-OPT-004 pipeline: train the fixed language model (lm_model.py, as in B-OPT-001) with your update rule at three batch
sizes for the same number of training tokens, then measure held-out loss at each batch size.

For each batch size B in PARAMS["batches"] (sequences of 1,024 tokens):
Stage "train_b<B>"  (imports optimizer.py)
  setup   map the training token shards, build the model on the GPU with the seed (untimed)
  timed   build RULE with the declared defaults; peak lr = lr * (B / 32) ** lr_batch_exponent (from hparam_space.yaml);
          train TOKENS / (B * 1,024) optimizer steps, each accumulating B / 32 micro-batches of 32 sequences
  finish  save model.pt; report the loss curve, divergence and the size of the optimizer state
Stage "eval_b<B>"   (does not import your code; receives only model.pt)
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

BATCHES = (32, 128, 512)
MICRO = 32
PARAMS = {"tokens": 3000 * 32 * 1024, "batches": list(BATCHES), "micro_batch": MICRO, "seq": 1024, "warmup_fraction": 0.05,
          "final_lr_fraction": 0.1, "lr_1d_ratio": 0.15, "precision": "bf16 autocast, fp32 weights and optimizer arithmetic",
          "state_numel_cap_ratio": 3.0, "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"],
          "eval_batch": 16}
TRAIN_TIMEOUT = 1500.0
HELD_OUT_KEYS = ["tokens"]
PIPELINE = []
for _b in BATCHES:
    PIPELINE += [
        {"stage": f"train_b{_b}", "timeout": TRAIN_TIMEOUT + 120, "finish_cap": 60, "tail_slack": 10, "threads": 8,
         "held_out": False, "models": False},
        {"stage": f"eval_b{_b}", "timeout": 600, "finish_cap": 180, "tail_slack": 10, "threads": 8, "held_out": True,
         "models": False, "carry": {f"train_b{_b}": ["model.pt"]}},
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


def _schedule(step: int, steps: int, warmup: int, lo: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def load_rule(submission: Path):
    """RULE from optimizer.py with the defaults declared in hparam_space.yaml, and the declared lr batch exponent."""
    import yaml
    import importlib.util
    import sys
    sys.path.insert(0, str(submission))
    spec = importlib.util.spec_from_file_location("abench_submission_optimizer", submission / "optimizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = yaml.safe_load((submission / "hparam_space.yaml").read_text(encoding="utf-8"))
    tun = {k: v["default"] for k, v in (hs.get("tunable") or {}).items()}
    if len(tun) > 4:
        raise SystemExit("hparam_space.yaml declares more than 4 tunable hyperparameters")
    lr = float(tun.pop("lr"))
    expo = float(hs.get("lr_batch_exponent", 0.0))
    if not 0.0 <= expo <= 1.0:
        raise SystemExit("lr_batch_exponent must lie in [0, 1]")
    rule = mod.RULE(**dict(hs.get("fixed") or {}, **tun))
    if bool(hs.get("owns_schedule", False)):
        rule.owns_schedule = True
    return rule, lr, expo


def _make_timed_train(B: int):
    def _timed_train(ctx: Context, st):
        import torch
        from optim_api import make_torch_optimizer, state_numel, Ops
        prm, model, stream = st["prm"], st["model"], st["stream"]
        rule, lr, expo = load_rule(ctx.submission)
        lr = lr * (B / prm["batches"][0]) ** expo
        two_d = [p for n, p in model.named_parameters() if p.ndim >= 2]
        one_d = [p for n, p in model.named_parameters() if p.ndim < 2]
        opt = make_torch_optimizer(rule, [{"params": two_d, "lr": lr, "base_lr": lr},
                                          {"params": one_d, "lr": lr * prm["lr_1d_ratio"], "base_lr": lr * prm["lr_1d_ratio"]}], lr)
        rng = np.random.default_rng(ctx.seed)
        T, micro = prm["seq"], prm["micro_batch"]
        accum = max(1, B // micro)
        steps = max(1, prm["tokens"] // (B * T))
        warmup = max(1, int(prm["warmup_fraction"] * steps))
        curve, diverged = [], 0
        t0 = time.perf_counter()
        for step in range(steps):
            if not rule.owns_schedule:
                f = _schedule(step, steps, warmup, prm["final_lr_fraction"])
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
            tot = 0.0
            for _ in range(accum):
                off = rng.integers(0, len(stream) - T - 1, micro)
                chunk = np.stack([stream[o:o + T + 1] for o in off]).astype(np.int64)
                x = torch.from_numpy(chunk[:, :-1]).to(ctx.device, non_blocking=True)
                y = torch.from_numpy(chunk[:, 1:]).to(ctx.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
                    loss = model.loss(x, y) / accum
                loss.backward()
                tot += float(loss.detach())
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % max(1, steps // 30) == 0 or step == steps - 1:
                curve.append((step, tot))
                if not math.isfinite(tot) or tot > 20:
                    diverged = 1
                    break
        ctx.meta["train_wall_seconds_self_reported"] = round(time.perf_counter() - t0, 3)
        n_params = sum(p.numel() for p in model.parameters())
        n_state = sum(state_numel(s.get("rule_state", {}), Ops("torch")) for s in opt.state.values())
        return {"curve": curve, "diverged": diverged, "state_ratio": n_state / n_params, "model": model}
    return _timed_train


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


STAGES = {}
for _b in BATCHES:
    STAGES[f"train_b{_b}"] = Stage(setup=_setup_train, timed=_make_timed_train(_b), finish=_finish_train)
    STAGES[f"eval_b{_b}"] = Stage(timed=_timed_eval, finish=_finish_eval)


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out; --quick trains
    on the first shard only with a token budget of 1/10 (300 micro-batch equivalents)."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    if not quick:
        write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    else:
        (out / "public" / "overrides.json").write_text(json.dumps({"tokens": PARAMS["tokens"] // 10}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

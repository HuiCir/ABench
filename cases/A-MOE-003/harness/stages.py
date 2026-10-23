"""A-MOE-003 pipeline: the evaluator trains the fixed μ2 language model (moe_lm.py) with your feed-forward layer in
every block, then measures held-out loss, load balance and compute (evaluator and self-check use this same file).

One stage, "run" (imports moe.py), repeated for 3 seeds:
  setup   map the training shards (untimed)
  timed   build moe_lm.LM(Config(), FFN) with the seed; train 3,000 steps of 32 x 1,024 tokens with the fixed recipe of
          lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0); your aux["loss"] terms are added
  finish  (1) held-out loss and per-expert token counts over the held-out tokens
          (2) forward FLOPs per token on a held-out batch (FlopCounterMode) against the top-2-of-8 budget
          (3) causality and batch-independence checks: changing later tokens, or other sequences in the batch, must not
              change the outputs at earlier positions of a sequence
The evaluator runs the top-2 baseline (baselines/top2) through the same stage on the same seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train

STEPS, LR = 3000, 1e-3
PARAMS = {"steps": STEPS, "lr": LR, "recipe": lm_train.RECIPE, "flops_budget_ratio": 1.05,
          "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": "run", "timeout": 1800 + 300, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    return {"stream": np.concatenate(toks), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    from moe_lm import LM, Config
    ffn = ctx.import_submission("moe.py", required=("FFN",)).FFN
    torch.manual_seed(ctx.seed)
    model = LM(Config(), ffn).to(ctx.device)
    steps = int(st["ov"].get("steps", STEPS))
    prm = {k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}
    res = lm_train.train(model, st["stream"], steps, prm, ctx.seed, ctx.device, LR)
    return {"model": model, "res": res, "prm": prm}


def dense_reference_flops(device: str, seq: int) -> float:
    """FLOPs per token of the dense μ2 model with two SwiGLU MLPs of width 1,024 per layer (top-2 of eight)."""
    import torch.nn as nn
    from lm_model import MLP
    from moe_lm import LM, Config

    class Top2Dense(nn.Module):
        def __init__(self, d, layer, n_layers):
            super().__init__()
            self.a, self.b = MLP(Config()), MLP(Config())

        def forward(self, x):
            return self.a(x) + self.b(x), {}
    return lm_train.forward_flops_per_token(LM(Config(), Top2Dense).to(device), seq, device)


def _finish(ctx: Context, st):
    import torch
    from moe_lm import total_ffn_budget, Config
    model, res = st["model"], st["res"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = z["tokens"]
    counts = []

    def loss_fn(m, x, y):
        v = m.loss(x, y)
        counts.append([np.asarray(a["counts"].detach().float().cpu()) if "counts" in a else None for a in m.last_aux])
        return v
    loss = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, toks, st["prm"], ctx.device, loss_fn)
    out = {"val_loss": np.asarray([loss]), "diverged": np.asarray([float(res["diverged"])]),
           "tokens_per_second": np.asarray([res["tokens_per_second"]]), "curve": res["curve"][::10]}
    if counts and all(c is not None for c in counts[0]):
        per_layer = np.stack([np.sum([b[i] for b in counts], axis=0) for i in range(len(counts[0]))])
        out["expert_counts"] = per_layer.astype(np.float64)                  # (layers, experts)
    T = Config().max_seq
    x = torch.from_numpy(np.asarray(toks[: 8 * T]).astype(np.int64).reshape(8, T)).to(ctx.device)
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            ref_logits = model(x).float()
        out["flops_ratio"] = np.asarray([fc.get_total_flops() / x.numel() / dense_reference_flops(ctx.device, T)])
        t = T // 2
        x2 = x.clone()
        x2[:, t + 1:] = torch.randint(0, 50257, x2[:, t + 1:].shape, device=x.device)
        causal = (model(x2).float()[:, : t + 1] - ref_logits[:, : t + 1]).abs().max().item()
        x3 = x.clone()
        x3[1:] = torch.randint(0, 50257, x3[1:].shape, device=x.device)
        batch = (model(x3).float()[0] - ref_logits[0]).abs().max().item()
    scale = ref_logits.abs().max().item() + 1e-6
    out["causality_violation"] = np.asarray([float(causal / scale > 1e-3)])
    out["batch_dependence"] = np.asarray([float(batch / scale > 1e-3)])
    budget = total_ffn_budget(Config().d_model)
    out["ffn_params_ratio"] = np.asarray([max(model.ffn_parameters()) / budget])
    return out


STAGES = {"run": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out; --quick trains
    300 steps."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": STEPS // 10, "warmup": 15}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

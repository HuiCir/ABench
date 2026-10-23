"""U-BYTE-007 pipeline: the evaluator trains your tokenizer-free byte-level model with a fixed recipe and measures
bits per byte on held-out text, with and without character-level noise (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training bytes (untimed)
  timed   build() with the seed; train 3,000 steps of 32 windows of 4,096 bytes with the fixed recipe of lm_train.py
          (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0), the loss being model.loss(x)
  finish  bits per byte on the held-out bytes (windows of 4,096) and on the same bytes with 5% character-level noise
          (byte_data.add_noise); forward FLOPs per byte on a held-out window and parameters, next to the baseline
          (static 4-byte patches); causality check
The evaluator runs the static-patch baseline (baselines/static4) through the same stage on the same seeds.
"""
from __future__ import annotations

import importlib.util
import json
import math

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import byte_data
import lm_train

STEPS, LR, SEQ, NOISE_RATE, NOISE_SEED = 3000, 1e-3, 4096, 0.05, 2024
EVAL_BYTES = 8_388_608
PARAMS = {"steps": STEPS, "lr": LR, "seq": SEQ, "recipe": lm_train.RECIPE, "eval_bytes": EVAL_BYTES,
          "noise_rate": NOISE_RATE, "train_files": ["bytes_train_000001.bin", "bytes_train_000002.bin"]}
HELD_OUT_KEYS = ["bytes"]
PIPELINE = [{"stage": "run", "timeout": 1800 + 300, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
BASELINE = Path(__file__).resolve().parent / "baselines" / "static4" / "model.py"


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    parts = [np.asarray(byte_data.read_bytes(ctx.public / f)) for f in PARAMS["train_files"] if (ctx.public / f).is_file()]
    return {"stream": np.concatenate(parts), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    build = ctx.import_submission("model.py", required=("build",)).build
    torch.manual_seed(ctx.seed)
    model = build().to(ctx.device)
    steps = int(st["ov"].get("steps", STEPS))
    prm = dict({k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}, seq=SEQ)
    res = lm_train.train(model, st["stream"], steps, prm, ctx.seed, ctx.device, LR, loss_fn=lambda m, x, y: m.loss(x))
    return {"model": model, "res": res, "prm": prm, "ov": st["ov"]}


def bits_per_byte(model, b: np.ndarray, device: str, batch: int = 8) -> float:
    nats = lm_train.heldout_loss(model, np.concatenate([b, b[-1:]]), {"seq": SEQ, "eval_batch": batch}, device,
                                 loss_fn=lambda m, x, y: m.loss(x))
    return nats / math.log(2)


def _baseline_model(device: str):
    spec = importlib.util.spec_from_file_location("abench_byte_baseline", BASELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build().to(device)


def _finish(ctx: Context, st):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        b = np.asarray(z["bytes"])[: int(st["ov"].get("eval_bytes", EVAL_BYTES))]
    model, res = st["model"], st["res"]
    bad = lm_train.DIVERGED_LOSS / math.log(2)
    clean = bad if res["diverged"] else bits_per_byte(model, b, ctx.device)
    noisy_b = byte_data.add_noise(b, NOISE_RATE, NOISE_SEED)
    noisy = bad if res["diverged"] else bits_per_byte(model, noisy_b[: len(noisy_b) // SEQ * SEQ], ctx.device)
    x = torch.from_numpy(b[:SEQ].astype(np.int64))[None].to(ctx.device)
    base = _baseline_model(ctx.device)
    model.eval()
    base.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            ya = model(x).float()
        f = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc:
            base(x)
        fb = fc.get_total_flops()
        t = SEQ // 2
        x2 = x.clone()
        x2[:, t:] = torch.randint(1, 256, (1, SEQ - t), device=x.device)
        yb = model(x2).float()
    leak = float((ya[:, :t + 1] - yb[:, :t + 1]).abs().max() / (ya[:, :t + 1].abs().max() + 1e-6))
    n_par = sum(p.numel() for p in model.parameters())
    n_base = sum(p.numel() for p in base.parameters())
    return {"bpb": np.asarray([clean]), "bpb_noisy": np.asarray([noisy]), "diverged": np.asarray([float(res["diverged"])]),
            "flops_ratio": np.asarray([f / max(fb, 1)]), "flops_per_byte": np.asarray([f / SEQ]),
            "params_ratio": np.asarray([n_par / n_base]), "causal_leak": np.asarray([leak]),
            "curve": res["curve"][::10], "bytes_per_second": np.asarray([res["tokens_per_second"]])}


STAGES = {"run": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 8,388,608 bytes of the second training file are held out; --quick trains
    300 steps and evaluates 1,048,576 bytes."""
    b1 = np.asarray(byte_data.read_bytes(public / PARAMS["train_files"][0]))
    b2 = np.asarray(byte_data.read_bytes(public / PARAMS["train_files"][1]))
    hold = EVAL_BYTES
    (out / "public").mkdir(parents=True, exist_ok=True)
    b1.tofile(out / "public" / PARAMS["train_files"][0])
    b2[:-hold].tofile(out / "public" / PARAMS["train_files"][1])
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": STEPS // 10, "warmup": 15,
                                                                   "eval_bytes": 1_048_576}))
    np.savez(out / "held_out.npz", bytes=b2[-hold:])
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}



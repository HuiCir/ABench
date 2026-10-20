"""PRO-SPD-004 pipeline: time-to-target pretraining on one 4 × H100 node (evaluator and self-check use this file).

Stage "train" (runs your train.py under torchrun on all GPUs)
  timed   torchrun --standalone --nproc_per_node=<GPUs> train.py --data <public> --out <dir> --seed <seed>; the clock
          runs from the launch until every training process has exited, so compilation and data loading count
  finish  check out/model.pt and count its parameters
Stage "eval" (imports your model.py; receives only the out/ directory)
  timed   build the model and load out/model.pt on one GPU
  finish  mean next-token loss on the first 10,485,760 tokens of the FineWeb validation shard
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

TARGET_LOSS = 2.92
VAL_TOKENS = 10_485_760
PARAM_CAP = 450_000_000
TRAIN_CAP = 3 * 3600.0
PARAMS = {"target_val_loss": TARGET_LOSS, "val_tokens": VAL_TOKENS, "param_cap": PARAM_CAP,
          "train_cap_seconds": TRAIN_CAP, "gpus": 4, "vocab": 50257}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [
    {"stage": "train", "timeout": TRAIN_CAP + 600, "finish_cap": 120, "tail_slack": 30, "threads": 32,
     "held_out": False, "models": False},
    {"stage": "eval", "timeout": 1800, "finish_cap": 1200, "tail_slack": 30, "threads": 16, "held_out": True,
     "models": False, "carry": {"train": ["out"]}},
]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _timed_train(ctx: Context, _):
    import torch
    n = int(_overrides(ctx).get("gpus", torch.cuda.device_count() if torch.cuda.is_available() else 1))
    out = ctx.work / "out"
    out.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={max(1, n)}",
           str(ctx.submission / "train.py"), "--data", str(ctx.public), "--out", str(out), "--seed", str(ctx.seed)]
    if _overrides(ctx).get("quick"):
        cmd.append("--quick")
    env = dict(os.environ, PYTHONPATH=str(ctx.submission) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(ctx.work / "train_stdout.txt", "wb") as so, open(ctx.work / "train_stderr.txt", "wb") as se:
        r = subprocess.run(cmd, cwd=ctx.work, env=env, stdout=so, stderr=se)
    if r.returncode != 0:
        tail = (ctx.work / "train_stderr.txt").read_bytes()[-3000:].decode("utf-8", "replace")
        raise SystemExit(f"train.py exited with code {r.returncode}\n{tail}")
    return out


def _finish_train(ctx: Context, out):
    import torch
    sd = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    return {"n_params": np.asarray([n], np.int64)}


def _timed_eval(ctx: Context, _):
    import torch
    mod = ctx.import_submission("model.py", required=("build_model",))
    model = mod.build_model()
    model.load_state_dict(torch.load(ctx.work / "out" / "model.pt", map_location="cpu", weights_only=True))
    seq = int(getattr(mod, "EVAL_SEQ_LEN", 1024))
    return model.to(ctx.device).eval(), seq


def _finish_eval(ctx: Context, st):
    import torch
    import torch.nn.functional as F
    model, seq = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = torch.from_numpy(z["tokens"].astype(np.int64))
    n_pred = len(toks) - 1
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for s in range(0, n_pred, seq):
            e = min(s + seq, n_pred)
            x = toks[s:e].unsqueeze(0).to(ctx.device)
            y = toks[s + 1:e + 1].unsqueeze(0).to(ctx.device)
            logits = model(x)
            logits = logits.logits if hasattr(logits, "logits") else logits
            total += float(F.cross_entropy(logits[0].float(), y[0], reduction="sum"))
            count += e - s
    return {"val_loss": np.asarray([total / max(count, 1)]), "val_tokens": np.asarray([count])}


STAGES = {"train": Stage(timed=_timed_train, finish=_finish_train),
          "eval": Stage(timed=_timed_eval, finish=_finish_eval)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: train on the public shards; the last 10,485,761 tokens of the last training shard
    play the held-out role (they are also in your training data, so the dev loss is optimistic). --quick passes
    --quick to train.py and evaluates on 1,048,577 tokens."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from token_data import read_shard
    shards = sorted(public.glob("fineweb_train_*.bin"))
    last = read_shard(shards[-1])
    n = 1_048_577 if quick else VAL_TOKENS + 1
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", tokens=np.asarray(last[-n:]))
    pub = out / "public"
    pub.mkdir(exist_ok=True)
    for s in shards:
        (pub / s.name).symlink_to(s)
    if quick:
        (pub / "overrides.json").write_text(json.dumps({"quick": True}))
    return {"public": pub, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

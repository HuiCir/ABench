"""A-NORM-005 pipeline: the evaluator trains the fixed language model (norm_lm.py) with your normalisation and
residual scheme at two scales and two learning rates, then measures held-out loss (evaluator and self-check use this
same file).

For each run r = (scale, lr) in RUNS, one stage "run_<scale>_<lr>" (imports norm.py):
  setup   map the training shards (untimed)
  timed   build norm_lm.LM(SCALES[scale], SCHEME) with the seed and train it with the fixed recipe of lm_train.py
          (AdamW, warm-up and cosine decay, bf16, clipping 1.0) for STEPS[scale] steps of 32 x 1,024 tokens
  finish  held-out loss, number of loss spikes, divergence; forward FLOPs per token and scheme parameters of your
          model and of the baseline scheme at the same scale (the compute gate)
The evaluator runs the pre-RMSNorm baseline (baselines/prenorm) through the same stages on the same seeds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train

STEPS = {"mu1": 1500, "mu2": 3000}
LRS = {"lr1e-3": 1e-3, "lr4e-3": 4e-3}
RUNS = [(s, l) for s in STEPS for l in LRS]
PARAMS = {"steps": STEPS, "learning_rates": LRS, "recipe": lm_train.RECIPE, "flops_ratio_max": 1.02,
          "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": f"run_{s}_{l}", "timeout": (1500 if s == "mu2" else 600) + 300, "finish_cap": 240,
             "tail_slack": 10, "threads": 8, "held_out": True, "models": False} for s, l in RUNS]
BASELINE = Path(__file__).resolve().parent / "baselines" / "prenorm" / "norm.py"


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _load_scheme(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod.SCHEME


def _setup(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    return {"stream": np.concatenate(toks), "ov": _overrides(ctx)}


def _make_timed(scale: str, lr_key: str):
    def _timed(ctx: Context, st):
        import torch
        from norm_lm import LM, SCALES
        scheme = ctx.import_submission("norm.py", required=("SCHEME",)).SCHEME
        torch.manual_seed(ctx.seed)
        model = LM(SCALES[scale], scheme).to(ctx.device)
        steps = int(st["ov"].get("steps", {}).get(scale, STEPS[scale]))
        prm = {k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}
        res = lm_train.train(model, st["stream"], steps, prm, ctx.seed, ctx.device, LRS[lr_key])
        return {"model": model, "res": res, "scheme": scheme, "prm": prm}
    return _timed


def _make_finish(scale: str):
    def _finish(ctx: Context, st):
        import torch
        from norm_lm import LM, SCALES
        with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
            toks = z["tokens"]
        model, res = st["model"], st["res"]
        loss = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, toks, st["prm"], ctx.device)
        seq = SCALES[scale].max_seq
        flops = lm_train.forward_flops_per_token(model, seq, ctx.device)
        base = LM(SCALES[scale], _load_scheme(BASELINE, "abench_baseline_norm")).to(ctx.device)
        base_flops = lm_train.forward_flops_per_token(base, seq, ctx.device)
        n_par = sum(p.numel() for p in model.parameters())
        base_par = sum(p.numel() for p in base.parameters())
        del base
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return {"val_loss": np.asarray([loss]), "diverged": np.asarray([float(res["diverged"])]),
                "spikes": np.asarray([float(res["spikes"])]), "curve": res["curve"][::10],
                "flops_ratio": np.asarray([flops / max(base_flops, 1.0)]),
                "params_ratio": np.asarray([n_par / max(base_par, 1)]),
                "scheme_params": np.asarray([float(model.scheme_parameters())]),
                "tokens_per_second": np.asarray([res["tokens_per_second"]])}
    return _finish


STAGES = {f"run_{s}_{l}": Stage(setup=_setup, timed=_make_timed(s, l), finish=_make_finish(s)) for s, l in RUNS}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out; --quick trains
    every run for a tenth of its steps."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": {k: v // 10 for k, v in STEPS.items()},
                                                                   "warmup": 15}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

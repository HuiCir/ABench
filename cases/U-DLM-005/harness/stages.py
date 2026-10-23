"""U-DLM-005 pipeline: a submitted sampler drives the frozen MDLM through the counting denoiser at three budgets,
unconditionally and continuing held-out prefixes; the samples are then scored by an autoregressive model the
submission never sees (evaluator and self-check use this same file).

Stage "sample" (imports sampler.py; the MDLM is under models/mdlm-owt):
  timed   import sampler.py, load the MDLM (untimed for the sampler, but inside this process)
  finish  for NFE in NFES: N_UNCOND unconditional samples of 1,024 tokens and one 256-token continuation of each
          held-out 256-token prefix, in batches of BATCH, each batch with its own Denoiser(budget = NFE) and seed;
          wall-clock seconds per budget; samples saved to samples.npz
Stage "score" (does not import your code; GPT-2 Large under judge_models/, or GPT-2 under models/ in the self-check)
  finish  generative NLL per sample (continuations conditioned on their prefix) and unigram entropy per sample;
          the same for the held-out reference texts (1,024-token windows whose first 256 tokens are the prefixes)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import dlm
from abench_stage import Context, Stage

NFES = (16, 32, 64)
N_UNCOND, BATCH, LENGTH, PREFIX, CONT = 256, 32, 1024, 256, 256
MODEL, SCORER, DEV_SCORER = "mdlm-owt", "gpt2-large", "gpt2"
PARAMS = {"nfes": list(NFES), "n_uncond": N_UNCOND, "batch": BATCH, "length": LENGTH, "prefix": PREFIX,
          "continuation": CONT, "scorer": SCORER, "entropy_floor_ratio": 0.9}
HELD_OUT_KEYS = ["prefixes", "references"]
PIPELINE = [{"stage": "sample", "timeout": 300 + 3000, "finish_cap": 3000, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True},
            {"stage": "score", "timeout": 60, "finish_cap": 1800, "tail_slack": 10, "threads": 8, "held_out": True,
             "models": False, "judge_models": True, "carry": {"sample": ["samples.npz"]}}]


def _overrides(ctx):
    import json
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _load_mdlm(path: Path, device: str):
    from transformers import AutoModelForMaskedLM
    return AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True, local_files_only=True).to(device).eval()


def _timed(ctx: Context, _):
    mod = ctx.import_submission("sampler.py", required=("sample",))
    return mod, _load_mdlm(ctx.models / MODEL, ctx.device)


def _finish(ctx: Context, st):
    import torch
    mod, model = st
    ov = _overrides(ctx)
    n_unc = int(ov.get("n_uncond", N_UNCOND))
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        prefixes = z["prefixes"].astype(np.int64)
    out, seed = {}, int(ctx.seed)
    for nfe in ov.get("nfes", NFES):
        t0 = time.perf_counter()
        unc, cond, calls = [], [], []
        for i in range(0, n_unc, BATCH):
            b = min(BATCH, n_unc - i)
            den = dlm.Denoiser(model, nfe)
            x = mod.sample(den, None, LENGTH, nfe, dlm.Gen(seed + 7919 * nfe + i, ctx.device, b))
            unc.append(_check(x, b, LENGTH, None))
            calls.append(den.calls)
        for i in range(0, len(prefixes), BATCH):
            pre = torch.as_tensor(prefixes[i:i + BATCH], device=ctx.device)
            den = dlm.Denoiser(model, nfe)
            x = mod.sample(den, pre, PREFIX + CONT, nfe, dlm.Gen(seed + 104729 * nfe + i, ctx.device, len(pre)))
            cond.append(_check(x, len(pre), PREFIX + CONT, prefixes[i:i + BATCH]))
            calls.append(den.calls)
        torch.cuda.synchronize() if ctx.device.startswith("cuda") else None
        out[f"uncond_{nfe}"] = np.concatenate(unc)
        out[f"cond_{nfe}"] = np.concatenate(cond)
        out[f"seconds_{nfe}"] = np.asarray([time.perf_counter() - t0])
        out[f"max_calls_{nfe}"] = np.asarray([max(calls)])
    np.savez(ctx.work / "samples.npz", **{k: v for k, v in out.items() if k.startswith(("uncond_", "cond_"))})
    return {k: v for k, v in out.items() if k.startswith(("seconds_", "max_calls_"))}


def _check(x, b: int, length: int, prefix) -> np.ndarray:
    a = x.detach().cpu().numpy().astype(np.int64) if hasattr(x, "detach") else np.asarray(x, np.int64)
    if a.shape != (b, length):
        raise ValueError(f"sample() returned shape {a.shape}, expected {(b, length)}")
    if (a == dlm.MASK).any() or (a < 0).any() or (a >= dlm.VOCAB).any():
        raise ValueError("samples must be fully unmasked GPT-2 token ids")
    if prefix is not None and not np.array_equal(a[:, :prefix.shape[1]], prefix):
        raise ValueError("the prefix must be kept unchanged")
    return a


def _score_timed(ctx: Context, _):
    return None


def _score_finish(ctx: Context, _):
    import torch
    from transformers import AutoModelForCausalLM
    name = SCORER if ctx.models is not None and (ctx.models / SCORER).is_dir() else DEV_SCORER
    path = (ctx.models / SCORER) if name == SCORER else _dev_scorer(ctx)
    scorer = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True).to(ctx.device).eval()
    out = {"scorer": np.asarray([name], dtype=np.str_)}
    with np.load(ctx.work / "samples.npz", allow_pickle=False) as z:
        keys = list(z.files)
        for k in keys:
            toks = z[k]
            start = PREFIX if k.startswith("cond_") else 1
            out[f"nll_{k}"] = dlm.gen_nll(scorer, toks, start, ctx.device)
            out[f"entropy_{k}"] = np.asarray([dlm.unigram_entropy(t[start:] if start > 1 else t) for t in toks])
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        ref = z["references"].astype(np.int64)
    out["nll_reference"] = dlm.gen_nll(scorer, ref[:, :PREFIX + CONT], PREFIX, ctx.device)
    out["entropy_reference_cond"] = np.asarray([dlm.unigram_entropy(t[PREFIX:PREFIX + CONT]) for t in ref])
    out["entropy_reference_uncond"] = np.asarray([dlm.unigram_entropy(t) for t in ref])
    del scorer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def _dev_scorer(ctx: Context) -> Path:
    """The self-check has no judge models: GPT-2 small from the sandbox's models directory scores instead."""
    import os
    return Path(os.environ.get("ABENCH_MODELS_DIR", "")) / DEV_SCORER


STAGES = {"sample": Stage(timed=_timed, finish=_finish), "score": Stage(timed=_score_timed, finish=_score_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: 64 public 1,024-token windows; their first 256 tokens are the prefixes and the windows the
    references (quick: 32 unconditional samples, the first 32 windows, NFE 16 only)."""
    import json
    with np.load(public / "dev_text.npz", allow_pickle=False) as z:
        windows = z["windows"].astype(np.int64)
    if quick:
        windows = windows[:32]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", prefixes=windows[:, :PREFIX], references=windows[:, :LENGTH])
    pub = out / "public"
    pub.mkdir(exist_ok=True)
    for f in public.iterdir():
        if f.is_file() and not (pub / f.name).exists():
            (pub / f.name).symlink_to(f)
    if quick:
        (pub / "overrides.json").write_text(json.dumps({"n_uncond": 32, "nfes": [16]}))
    return {"public": pub, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

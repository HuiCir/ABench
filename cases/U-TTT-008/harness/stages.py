"""U-TTT-008 pipeline: the evaluator takes your modified SmolLM2-135M (at least half of its attention layers replaced by
layers with a fixed-size state), continues its training on 50M FineWeb-Edu tokens with a fixed recipe, and measures
long-context language modelling and retrieval (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training tokens (untimed)
  timed   build(models_dir + "/SmolLM2-135M", device) -> smol_lm.SmolLM with your mixers; train STEPS steps of 12 x 4,096
          tokens with the fixed recipe of lm_train.py (AdamW peak lr 3e-4, warm-up and cosine decay, bf16, clipping)
  finish  (1) loss at every position of the first 8,192 tokens of each held-out PG19 book
          (2) passkey retrieval: a 5-digit key hidden at a random depth of 2k, 4k and 8k tokens of book text; correct if
              the greedy prediction of every key token after the question is right (teacher forcing)
          (3) held-out FineWeb-Edu loss on windows of 2,048 tokens
          (4) per-layer cache growth between 1,024 and 2,048 tokens of context (bf16 autocast): the number of layers
              whose cache does not grow; decoding consistency; decoding throughput after an 8,192-token prefill
The evaluator runs the sliding-window baseline (baselines/swa) through the same stage on the same seeds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train
import smol_lm

STEPS, LR, SEQ = 1017, 3e-4, 4096
PRM = {"batch": 12, "accum": 3, "seq": SEQ, "warmup": 50, "weight_decay": 0.1, "grad_clip": 1.0}
PARAMS = {"steps": STEPS, "lr": LR, "recipe": dict(lm_train.RECIPE, **PRM), "min_constant_layers": 15,
          "book_tokens": 8192, "passkey_lengths": [2048, 4096, 8192], "train_shards": ["fwedu_train_000001.bin"]}
HELD_OUT_KEYS = ["tokens", "books", "pk_2048", "pk_4096", "pk_8192", "pk_answer_len"]
PIPELINE = [{"stage": "run", "timeout": 3600 + 300, "finish_cap": 1200, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]
BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    return {"stream": np.asarray(read_shard(ctx.public / PARAMS["train_shards"][0])), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    build = ctx.import_submission("model.py", required=("build",)).build
    torch.manual_seed(ctx.seed)
    model = build(str(ctx.models / "SmolLM2-135M"), ctx.device)
    prm = dict(PRM, **{k: v for k, v in st["ov"].items() if k in lm_train.RECIPE})
    res = lm_train.train(model, st["stream"], int(st["ov"].get("steps", STEPS)), prm, ctx.seed, ctx.device, LR)
    return {"model": model, "res": res, "prm": prm, "ov": st["ov"]}


def position_losses(model, books: np.ndarray, device: str) -> np.ndarray:
    """books (n, L) int -> mean loss at each position 1..L-1 (n averaged)."""
    import torch
    import torch.nn.functional as F
    tot = np.zeros(books.shape[1] - 1)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for b in books:
            x = torch.from_numpy(b.astype(np.int64))[None].to(device)
            lg = model(x[:, :-1]).float()
            tot += F.cross_entropy(lg[0], x[0, 1:], reduction="none").cpu().numpy()
    return tot / len(books)


def passkey_accuracy(model, prompts: np.ndarray, n_ans: int, device: str) -> float:
    """prompts (n, L): context, question and the answer's n_ans tokens at the end; exact teacher-forced argmax."""
    import torch
    ok = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for p in prompts:
            x = torch.from_numpy(p.astype(np.int64))[None].to(device)
            pred = model(x[:, :-1]).argmax(-1)[0, -n_ans:]
            ok += int((pred == x[0, -n_ans:]).all())
    return ok / len(prompts)


def layer_growth(model, device: str, t1: int = 1024, t2: int = 2048) -> np.ndarray:
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(1, model.c.vocab_size, (1, t2), generator=g).to(device)
    with torch.no_grad():
        c1, c2 = model.prefill(x[:, :t1])[1]["layers"], model.prefill(x)[1]["layers"]
    return np.array([(smol_lm.cache_bytes(b) - smol_lm.cache_bytes(a)) / (t2 - t1) for a, b in zip(c1, c2)])


def decode_tokens_per_second(ctx: Context, model, prefix: int = 8192, steps: int = 64) -> float:
    import torch
    x = torch.randint(1, model.c.vocab_size, (1, prefix + steps), device=ctx.device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        _, cache = model.prefill(x[:, :prefix])
        ctx.sync()
        t0 = time.perf_counter()
        for t in range(prefix, prefix + steps):
            _, cache = model.step(x[:, t:t + 1], cache)
        ctx.sync()
    return steps / (time.perf_counter() - t0)


def _finish(ctx: Context, st):
    import torch
    model, res, prm = st["model"], st["res"], st["prm"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        held = {k: z[k] for k in z.files}
    n_books = int(st["ov"].get("n_books", len(held["books"])))
    out = {"diverged": np.asarray([float(res["diverged"])]), "curve": res["curve"][::10]}
    out["position_loss"] = position_losses(model, held["books"][:n_books], ctx.device)
    n_ans = int(held["pk_answer_len"][0])
    for L in PARAMS["passkey_lengths"]:
        out[f"passkey_{L}"] = np.asarray([passkey_accuracy(model, held[f"pk_{L}"], n_ans, ctx.device)])
    toks = held["tokens"][: int(st["ov"].get("eval_tokens", 2_097_152)) + 1]
    out["val_loss_fwedu"] = np.asarray([lm_train.DIVERGED_LOSS if res["diverged"] else
                                        lm_train.heldout_loss(model, toks, dict(prm, seq=2048, eval_batch=8), ctx.device)])
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        out["layer_growth"] = layer_growth(model, ctx.device)
    x = torch.from_numpy(toks[:2 * 1088].astype(np.int64).reshape(2, 1088)).to(ctx.device)
    out["decode_mismatch"] = np.asarray([smol_lm.decode_consistency(model.float(), x, 64)])
    out["decode_tokens_per_second"] = np.asarray([decode_tokens_per_second(ctx, model)])
    return out


STAGES = {"run": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,153 training tokens are held out for the short-context loss; the
    long-context material (PG19 books, passkey prompts) is the evaluator's own and is not in the sandbox, so the
    self-check builds stand-ins from the held-out training tokens (documents are short, so they test the pipeline, not
    long-range ability); --quick trains 60 steps."""
    from token_data import read_shard, write_shard
    s = np.asarray(read_shard(public / PARAMS["train_shards"][0]))
    hold = 2_097_153
    tail = s[-hold:]
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], s[:-hold])
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": 60, "warmup": 6, "eval_tokens": 262_144,
                                                                   "n_books": 4}))
    rng = np.random.default_rng(0)
    books = np.stack([tail[i * 8193:(i + 1) * 8193] for i in range(16)])
    held = {"tokens": tail, "books": books, "pk_answer_len": np.asarray([3])}
    for L in PARAMS["passkey_lengths"]:
        held[f"pk_{L}"] = np.stack([tail[o:o + L] for o in rng.integers(0, len(tail) - L, 8)])
    np.savez(out / "held_out.npz", **held)
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

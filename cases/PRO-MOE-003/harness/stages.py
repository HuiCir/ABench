"""PRO-MOE-003 pipeline: audit of the pretraining process, held-out loss and compute per token of the delivered
mixture-of-experts model, and a sampled resume of the training run (evaluator and self-check use this file).

Stage "audit" (evaluator code only; pro_stages.audit): chain, traceability (the run starts from random
  initialisation: its start entry records no starting weights), code identity.
Stage "eval" (imports model.py of the submission, which is the code of the registered run):
  timed   build_model(); load checkpoint/model.pt (a state dict)
  finish  stored elements of model.pt; forward FLOPs per token on 8 held-out windows of 1,024 tokens
          (torch.utils.flop_counter, the routing the model actually performs); mean next-token loss on the
          10,485,760 held-out FineWeb tokens in windows of EVAL_SEQ_LEN                               (≤ 3,600 s)
Stage "resume" (pro_stages.resume): one registered checkpoint, picked with the secret number resume_u, resumed for at
  most 1 hour; the logged training loss must match within 1%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pro_stages as P
from abench_stage import Context, Stage

VAL_TOKENS, MAX_ELEMENTS, MAX_FLOPS_PER_TOKEN = 10_485_760, 2_500_000_000, 0.8e9
RESUME_METRIC, RESUME_REL = "loss", 0.01
PARAMS = {"val_tokens": VAL_TOKENS, "max_stored_elements": MAX_ELEMENTS, "max_forward_flops_per_token": MAX_FLOPS_PER_TOKEN,
          "flops_windows": 8, "flops_seq": 1024, "vocab": 50257, "resume_metric": RESUME_METRIC}
HELD_OUT_KEYS = ["tokens", "resume_u"]
PIPELINE = [
    {"stage": "audit", "timeout": 3600, "finish_cap": 3000, "tail_slack": 30, "threads": 16, "held_out": False, "models": False},
    {"stage": "eval", "timeout": 4800, "finish_cap": 3600, "tail_slack": 30, "threads": 16, "held_out": True, "models": False},
    {"stage": "resume", "timeout": 3600 + 1800, "finish_cap": 3600 + 1200, "tail_slack": 30, "threads": 64,
     "held_out": True, "models": False},
]


def _finish_audit(ctx: Context, _):
    return P.audit(ctx.submission, None)


def _timed_eval(ctx: Context, _):
    import torch
    mod = ctx.import_submission("model.py", required=("build_model",))
    model = mod.build_model()
    sd = torch.load(ctx.submission / "checkpoint" / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    return model.to(ctx.device).eval(), int(getattr(mod, "EVAL_SEQ_LEN", 1024)), n


def flops_per_token(model, toks, device: str, windows: int = 8, seq: int = 1024) -> float:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    total = 0
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        for w in range(windows):
            model(toks[w * seq:(w + 1) * seq].unsqueeze(0).to(device))
        total = fc.get_total_flops()
    return total / (windows * seq)


def _finish_eval(ctx: Context, st):
    import torch
    import torch.nn.functional as F
    model, seq, n = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = torch.from_numpy(z["tokens"].astype(np.int64))
    fpt = flops_per_token(model, toks, ctx.device)
    n_pred, total, count = len(toks) - 1, 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for s in range(0, n_pred, seq):
            e = min(s + seq, n_pred)
            logits = model(toks[s:e].unsqueeze(0).to(ctx.device))
            logits = logits.logits if hasattr(logits, "logits") else logits
            total += float(F.cross_entropy(logits[0].float(), toks[s + 1:e + 1].to(ctx.device), reduction="sum"))
            count += e - s
    return {"val_loss": np.asarray([total / max(count, 1)]), "val_tokens": np.asarray([count]),
            "stored_elements": np.asarray([n], np.int64), "flops_per_token": np.asarray([fpt])}


def _finish_resume(ctx: Context, _):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        u = float(z["resume_u"][0])
    return P.resume(ctx.submission, ctx.work, u, RESUME_METRIC, RESUME_REL)


_id = lambda ctx, s: s                                                       # noqa: E731
STAGES = {"audit": Stage(timed=_id, finish=_finish_audit), "eval": Stage(timed=_timed_eval, finish=_finish_eval),
          "resume": Stage(timed=_id, finish=_finish_resume)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the last 10,485,761 tokens of the last training shard play the held-out role (also training data,
    so the dev loss is optimistic); --quick: 1,048,577 tokens."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from token_data import read_shard
    last = read_shard(sorted(public.glob("fineweb_train_*.bin"))[-1])
    n = 1_048_577 if quick else VAL_TOKENS + 1
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", tokens=np.asarray(last[-n:]), resume_u=np.asarray([0.5]))
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

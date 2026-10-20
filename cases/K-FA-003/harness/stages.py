"""K-FA-003 pipeline: attention forward kernel, correctness and speed against PyTorch SDPA (evaluator and self-check).

One stage, "bench":
  setup   nothing (the CUDA context is created before the clock starts)
  timed   import kernel.py (module-level compilation or autotuning counts here, capped by the stage timeout)
  finish  for every held-out shape: random bf16 q, k, v from the shape's seed; your attention() vs an fp32 reference
          for accuracy; CUDA-event timing of your attention() and of SDPA (flash backend where available)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

WARMUP, REPS = 5, 20
FORBIDDEN = re.compile(r"scaled_dot_product|flash_attn|_flash_attention|_efficient_attention|xformers|"
                       r"_scaled_dot|sdpa_kernel|cudnn_attention", re.I)
PARAMS = {"dtype": "bfloat16", "layout": "(B, H, T, D)", "warmup": WARMUP, "reps": REPS,
          "error_rule": "max |out - ref| <= max(0.02, 4 x SDPA's own max error) and mean <= max(0.002, 4 x SDPA's mean error)"}
HELD_OUT_KEYS = ["shapes"]                     # rows: B, H, T, D, causal, seed
PIPELINE = [{"stage": "bench", "timeout": 600 + 900, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]

_REFS: dict = {}                               # evaluator references, captured before your module is imported


def _scan(sub: Path) -> int:
    hits = 0
    for p in sub.rglob("*.py"):
        hits += len(FORBIDDEN.findall(p.read_text(encoding="utf-8", errors="replace")))
    return hits


def _timed(ctx: Context, _):
    import torch
    import torch.nn.functional as F
    _REFS.update(sdpa=F.scaled_dot_product_attention, event=torch.cuda.Event, softmax=torch.softmax)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        _REFS.update(backend=SDPBackend.FLASH_ATTENTION, sdpa_kernel=sdpa_kernel)
    except Exception:
        pass
    mod = ctx.import_submission("kernel.py", required=("attention",))
    return mod


def _ref_fp32(q, k, v, causal):
    """Exact attention in fp32, computed one (batch, head) at a time to bound memory."""
    import torch
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    scale = q.shape[-1] ** -0.5
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            s = (q[b, h].float() @ k[b, h].float().T) * scale
            if causal:
                T = s.shape[0]
                s.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1), float("-inf"))
            out[b, h] = _REFS["softmax"](s, -1) @ v[b, h].float()
    return out


def _time(fn) -> float:
    import torch
    _EVENT = _REFS["event"]
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = _EVENT(enable_timing=True), _EVENT(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def _sdpa(q, k, v, causal):
    if "sdpa_kernel" in _REFS:
        try:
            with _REFS["sdpa_kernel"]([_REFS["backend"]]):
                return _REFS["sdpa"](q, k, v, is_causal=causal)
        except RuntimeError:
            pass
    return _REFS["sdpa"](q, k, v, is_causal=causal)


def _finish(ctx: Context, mod):
    import torch
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        shapes = z["shapes"].astype(np.int64)
    rows = []
    for B, H, T, D, causal, seed in shapes:
        g = torch.Generator(device=ctx.device)
        g.manual_seed(int(seed))
        q, k, v = (torch.randn(int(B), int(H), int(T), int(D), device=ctx.device, dtype=torch.bfloat16, generator=g)
                   for _ in range(3))
        c = bool(causal)
        ref = _ref_fp32(q, k, v, c)
        out = mod.attention(q, k, v, causal=c)
        ok_shape = tuple(out.shape) == tuple(q.shape)
        err = (out.float() - ref).abs() if ok_shape else None
        base = (_sdpa(q, k, v, c).float() - ref).abs()
        t_sub = _time(lambda: mod.attention(q, k, v, causal=c)) if ok_shape else float("nan")
        t_ref = _time(lambda: _sdpa(q, k, v, c))
        rows.append([float(err.max()) if ok_shape else float("inf"), float(err.mean()) if ok_shape else float("inf"),
                     float(base.max()), float(base.mean()), t_sub, t_ref, float(ok_shape)])
        del q, k, v, ref, out
        torch.cuda.empty_cache()
    return {"rows": np.asarray(rows, np.float64), "forbidden_api": np.asarray([_scan(ctx.submission)])}


STAGES = {"bench": Stage(timed=_timed, finish=_finish)}

DEV_SHAPES = np.array([  # B, H, T, D, causal, seed: public shapes for the self-check
    [4, 16, 1024, 64, 0, 1], [4, 16, 1024, 64, 1, 2], [2, 16, 2048, 128, 1, 3], [1, 16, 4096, 64, 1, 4],
    [8, 8, 512, 128, 0, 5], [2, 12, 1536, 64, 1, 6]], np.int64)


def make_dev(public, out: Path, quick: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", shapes=DEV_SHAPES[:2] if quick else DEV_SHAPES)
    return {"public": None, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

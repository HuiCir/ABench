"""C-FA-004 pipeline: a trainable attention (forward and backward) for bf16 inputs: correctness of the output and the
three input gradients, speed against PyTorch SDPA, and peak memory (evaluator and self-check).

One stage, "bench":
  timed   import kernel.py (module-level compilation or autotuning counts here, capped by the stage timeout)
  finish  for every held-out shape: random bf16 q, k, v and output gradient from the shape's seed; out = attention(q, k,
          v, causal) and out.backward(dout); output and dq, dk, dv against an fp32 reference (autograd, one (batch,
          head) at a time); CUDA-event timing of forward + backward of yours and of SDPA (flash backend where
          available); peak extra memory of one forward + backward of each
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

WARMUP, REPS = 3, 10
FORBIDDEN = re.compile(r"scaled_dot_product|flash_attn|_flash_attention|_efficient_attention|xformers|"
                       r"_scaled_dot|sdpa_kernel|cudnn_attention", re.I)
PARAMS = {"dtype": "bfloat16", "layout": "(B, H, T, D)", "warmup": WARMUP, "reps": REPS,
          "error_rule": "per tensor (out, dq, dk, dv): max |x - ref| <= max(abs_floor, 4 x SDPA's own max error) and mean "
                        "<= max(abs_floor / 10, 4 x SDPA's mean error); abs_floor 0.02 for out, 0.05 for gradients",
          "memory_rule": "peak memory above the inputs during forward + backward <= 2 x SDPA's"}
HELD_OUT_KEYS = ["shapes"]                     # rows: B, H, T, D, causal, seed
PIPELINE = [{"stage": "bench", "timeout": 600 + 1200, "finish_cap": 1200, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
FLOORS = (0.02, 0.05, 0.05, 0.05)              # out, dq, dk, dv

_REFS: dict = {}                               # evaluator references, captured before your module is imported


def _scan(sub: Path) -> int:
    return sum(len(FORBIDDEN.findall(p.read_text(encoding="utf-8", errors="replace"))) for p in sub.rglob("*.py"))


def _timed(ctx: Context, _):
    import torch
    import torch.nn.functional as F
    _REFS.update(sdpa=F.scaled_dot_product_attention, event=torch.cuda.Event, softmax=torch.softmax)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        _REFS.update(backend=SDPBackend.FLASH_ATTENTION, sdpa_kernel=sdpa_kernel)
    except Exception:
        pass
    return ctx.import_submission("kernel.py", required=("attention",))


def _ref_fp32(q, k, v, do, causal):
    """Exact attention and its gradients in fp32, one (batch, head) at a time."""
    import torch
    outs = [torch.empty(q.shape, dtype=torch.float32, device=q.device) for _ in range(4)]
    scale = q.shape[-1] ** -0.5
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            qq, kk, vv = (x[b, h].float().detach().requires_grad_(True) for x in (q, k, v))
            s = (qq @ kk.T) * scale
            if causal:
                T = s.shape[0]
                s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1), float("-inf"))
            o = _REFS["softmax"](s, -1) @ vv
            o.backward(do[b, h].float())
            for t, val in zip(outs, (o.detach(), qq.grad, kk.grad, vv.grad)):
                t[b, h] = val
    return outs


def _sdpa(q, k, v, causal):
    if "sdpa_kernel" in _REFS:
        try:
            with _REFS["sdpa_kernel"]([_REFS["backend"]]):
                return _REFS["sdpa"](q, k, v, is_causal=causal)
        except RuntimeError:
            pass
    return _REFS["sdpa"](q, k, v, is_causal=causal)


def _fwd_bwd(fn, q, k, v, do, causal):
    for t in (q, k, v):
        t.grad = None
    out = fn(q, k, v, causal)
    out.backward(do)
    return out, q.grad, k.grad, v.grad


def _time(run) -> float:
    import torch
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = _REFS["event"](enable_timing=True), _REFS["event"](enable_timing=True)
        a.record()
        run()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def _peak(run) -> float:
    import torch
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() - base)


def _finish(ctx: Context, mod):
    import torch
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        shapes = z["shapes"].astype(np.int64)
    sub = lambda q, k, v, c: mod.attention(q, k, v, causal=c)       # noqa: E731
    rows = []
    for B, H, T, D, causal, seed in shapes:
        g = torch.Generator(device=ctx.device)
        g.manual_seed(int(seed))
        q, k, v, do = (torch.randn(int(B), int(H), int(T), int(D), device=ctx.device, dtype=torch.bfloat16, generator=g)
                       for _ in range(4))
        for t in (q, k, v):
            t.requires_grad_(True)
        c = bool(causal)
        refs = _ref_fp32(q, k, v, do, c)
        mine = _fwd_bwd(sub, q, k, v, do, c)
        ok = all(x is not None and tuple(x.shape) == tuple(q.shape) for x in mine)
        base = _fwd_bwd(_sdpa, q, k, v, do, c)
        row = []
        for x, b, r in zip(mine, base, refs):
            e = (x.float() - r).abs() if ok else None
            eb = (b.float() - r).abs()
            row += [float(e.max()) if ok else float("inf"), float(e.mean()) if ok else float("inf"), float(eb.max()), float(eb.mean())]
        del refs, mine, base
        torch.cuda.empty_cache()
        t_sub = _time(lambda: _fwd_bwd(sub, q, k, v, do, c)) if ok else float("nan")
        t_ref = _time(lambda: _fwd_bwd(_sdpa, q, k, v, do, c))
        m_sub = _peak(lambda: _fwd_bwd(sub, q, k, v, do, c)) if ok else float("inf")
        m_ref = _peak(lambda: _fwd_bwd(_sdpa, q, k, v, do, c))
        rows.append(row + [t_sub, t_ref, m_sub, m_ref, float(ok)])
        del q, k, v, do
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

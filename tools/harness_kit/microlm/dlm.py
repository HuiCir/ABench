"""Masked (absorbing-state) discrete diffusion for the micro language-model cases (evaluator code; identical copy in each
case that uses it).

Forward process: each position i of a clean sequence x is independently replaced by MASK at time t in [0, 1] with
probability 1 - alpha_i(t); alpha_i(0) = 1, alpha_i(1) = 0, alpha_i decreasing in t. The default schedule is linear,
alpha(t) = 1 - t, the same for every position.

Evaluator's bound (Sahoo et al. 2024, MDLM; per-position schedules as in Shi et al. 2024): with the denoiser's
distribution p(. | z_t, t) at each masked position,
    NELBO(x) = ∫_0^1 Σ_i  -alpha_i'(t) / (1 - alpha_i(t)) · 1[z_t^i = MASK] · ( -log p(x_i | z_t, t) )  dt
is an upper bound on -log p(x) for the generative model that unmasks positions in reverse. nelbo_per_token()
estimates it by Monte Carlo with K stratified times per sequence (unbiased); the perplexity bound is exp(NELBO / T).
Only the log-probabilities at masked positions enter, so the bound is valid for any denoiser output.

DenoiserLM: the fixed bidirectional transformer of the diffusion cases (B-OPT-001's blocks without the causal mask,
the time t added to the token embeddings through a sinusoidal embedding and a linear map, tied embeddings).
"""
from __future__ import annotations

import math

import numpy as np

MASK = 50257                     # first id after the GPT-2 vocabulary; the vocabulary is padded to 50,304
T_MIN = 1e-5


def linear_alpha(t, seq: int):
    """alpha(t) = 1 - t for every position: (B,) -> (B, seq)."""
    return (1.0 - t)[:, None].expand(-1, seq)


def alpha_and_rate(alpha_fn, t, seq: int, h: float = 1e-4):
    """alpha (B, seq) and w = -alpha'(t) / (1 - alpha(t)) by central differences in float64."""
    import torch
    t64 = t.double()
    a = alpha_fn(t64, seq).double()
    lo, hi = (t64 - h).clamp(min=0.0), (t64 + h).clamp(max=1.0)
    da = (alpha_fn(hi, seq).double() - alpha_fn(lo, seq).double()) / (hi - lo)[:, None]
    w = (-da) / (1.0 - a).clamp(min=1e-12)
    return a.clamp(0.0, 1.0), torch.nan_to_num(w, nan=0.0, posinf=0.0).clamp(min=0.0)


def check_schedule(alpha_fn, seq: int, device: str = "cpu", min_window: float | None = None) -> dict:
    """alpha must lie in [0, 1], be non-increasing in t, and reach 1 at t = 0 and 0 at t = 1 (tolerance 1e-3).
    With min_window, every position must also take at least that much time to go from alpha = 0.9 to alpha = 0.1, so
    that many positions are unmasked in parallel at any time (this excludes autoregressive orders)."""
    import torch
    t = torch.linspace(0.0, 1.0, 2001, device=device, dtype=torch.float64)
    a = alpha_fn(t, seq).double()
    if a.shape != (len(t), seq) or not torch.isfinite(a).all():
        return {"ok": False, "reason": f"alpha must return finite values of shape (B, {seq})"}
    ok = bool((a >= -1e-6).all() and (a <= 1 + 1e-6).all() and (a[1:] <= a[:-1] + 1e-6).all()
              and (a[0] - 1).abs().max() < 1e-3 and a[-1].abs().max() < 1e-3)
    if not ok:
        return {"ok": False, "reason": "alpha must decrease from 1 at t = 0 to 0 at t = 1 within [0, 1]"}
    if min_window is not None:
        t90 = t[(a > 0.9).sum(0).clamp(max=len(t) - 1)]          # first grid time with alpha <= 0.9, per position
        t10 = t[(a > 0.1).sum(0).clamp(max=len(t) - 1)]
        width = float((t10 - t90).min())
        if width < min_window - 1e-3:
            return {"ok": False, "reason": f"some position goes from alpha 0.9 to 0.1 in {width:.3f} < {min_window}"}
    return {"ok": True, "reason": ""}


def corrupt(x, t, alpha, generator=None):
    """z_t: positions masked with probability 1 - alpha (B, T); returns (z, mask)."""
    import torch
    u = torch.rand(x.shape, device=x.device, generator=generator)
    m = u > alpha.to(u.dtype)
    return torch.where(m, torch.full_like(x, MASK), x), m


def nelbo_per_token(model, tokens: np.ndarray, seq: int, device: str, alpha_fn=None, k_times: int = 8,
                    batch: int = 16, seed: int = 0) -> float:
    """Monte-Carlo NELBO (nats per token) of `model(z, t) -> logits` on windows of `seq` tokens."""
    import torch
    import torch.nn.functional as F
    alpha_fn = alpha_fn or linear_alpha
    toks = np.asarray(tokens).astype(np.int64)
    n = len(toks) // seq
    X = torch.from_numpy(toks[: n * seq].reshape(n, seq))
    g = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for i in range(0, n, batch):
            x = X[i:i + batch].to(device)
            B = len(x)
            u = torch.rand(B, device=device, generator=g)
            for j in range(k_times):
                t = ((j + u) / k_times).clamp(T_MIN, 1.0)
                a, w = alpha_and_rate(alpha_fn, t, seq)
                z, m = corrupt(x, t, a, g)
                logp = F.log_softmax(model(z, t).float(), dim=-1)
                nll = -logp.gather(-1, x[..., None])[..., 0].double()
                total += float((w * nll * m).sum()) / k_times
            count += x.numel()
    model.train()
    return total / max(count, 1)


def mdlm_loss(model, x, alpha_fn=None, t=None):
    """The MDLM training objective: the same weighted cross-entropy as the bound, one time per sequence (low-
    discrepancy across the batch), averaged per token. Usable as a starting point by submissions."""
    import torch
    import torch.nn.functional as F
    alpha_fn = alpha_fn or linear_alpha
    B, T = x.shape
    if t is None:
        t = ((torch.arange(B, device=x.device) + torch.rand(1, device=x.device)) / B).clamp(T_MIN, 1.0)
    a, w = alpha_and_rate(alpha_fn, t, T)
    z, m = corrupt(x, t, a)
    logits = model(z, t).float()
    nll = F.cross_entropy(logits.view(-1, logits.size(-1)), x.reshape(-1), reduction="none").view(B, T)
    return (w.to(nll.dtype) * nll * m).sum() / (B * T)


def build_denoiser(cfg):
    """The fixed bidirectional denoiser at the sizes of an lm_model.Config."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import lm_model as base

    class BiAttention(base.Attention):
        def forward(self, x, cos, sin):
            B, T, _ = x.shape
            q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
            k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
            v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
            q, k = base.apply_rope(q, cos, sin), base.apply_rope(k, cos, sin)
            rep = self.h // self.kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            return self.o(y.transpose(1, 2).reshape(B, T, -1))

    class DenoiserLM(base.LM):
        def __init__(self, c):
            super().__init__(c)
            for b in self.blocks:
                bi = BiAttention(c)
                bi.load_state_dict(b.attn.state_dict())
                b.attn = bi
            self.time = nn.Linear(256, c.d_model, bias=True)
            nn.init.normal_(self.time.weight, std=c.init_std)
            nn.init.zeros_(self.time.bias)

        def forward(self, z, t):
            if self._rope is None or self._rope[0].device != z.device:
                self._rope = base.rope_cache(self.c.max_seq, self.c.d_model // self.c.n_heads, self.c.rope_base, z.device)
            cos, sin = self._rope
            half = 128
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=z.device) / half)
            ang = t.float()[:, None] * 1000.0 * freqs[None]
            temb = self.time(torch.cat([ang.sin(), ang.cos()], -1))
            x = self.embed(z) + temb[:, None, :].to(self.embed.weight.dtype)
            for b in self.blocks:
                x = b(x, cos, sin)
            return F.linear(self.norm(x), self.embed.weight)

    return DenoiserLM(cfg)

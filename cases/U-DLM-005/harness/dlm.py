"""U-DLM-005: the frozen masked diffusion LM as a counting denoiser, the baseline sampler, and the quality measures
(evaluator code, agent-visible).

Model: MDLM trained on OpenWebText (Sahoo et al. 2024; kuleshov-group/mdlm-owt, Apache-2.0): GPT-2 vocabulary plus a
mask token MASK = 50257, sequences of up to 1,024 tokens, no time conditioning (the network sees only the partly
masked sequence).

Denoiser(model)(x) -> log-probabilities (B, L, 50257) of the clean token at every position, given x (B, L) with MASK at
masked positions. As in MDLM's parametrisation, the mask token gets no probability and positions that are not masked
are returned as one-hot (log-probability 0 for their token). Every call counts one function evaluation (NFE),
whatever the batch size or which positions are of interest; the evaluator raises if a sampler exceeds its budget.

ancestral(): the baseline sampler. With NFE = K, time goes from 1 to 0 in K equal steps; at step t -> s every still
masked position is unmasked with probability (t - s) / t (the linear schedule alpha_t = 1 - t), its token drawn from
the denoiser's distribution; the last step unmasks everything.

Quality: generative perplexity under GPT-2 Large (the evaluator's scorer, not in the sandbox; GPT-2 small plays its
role in the self-check) and the unigram entropy of each sample (low entropy flags repetitive text, which lowers
perplexity without being better).
"""
from __future__ import annotations

import math

import numpy as np

MASK, VOCAB = 50257, 50257


class BudgetExceeded(RuntimeError):
    pass


class Denoiser:
    def __init__(self, model, budget: int, dtype=None):
        self.model, self.budget, self.calls = model, budget, 0
        self.dtype = dtype

    def __call__(self, x):
        import torch
        self.calls += 1
        if self.calls > self.budget:
            raise BudgetExceeded(f"more than {self.budget} denoiser calls")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = self.model(input_ids=x, timesteps=torch.zeros(x.shape[0], device=x.device))
        if not torch.is_tensor(logits):
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits.logits
        lp = logits[..., :VOCAB].float().log_softmax(-1)
        keep = x != MASK
        if keep.any():
            onehot = torch.full_like(lp, -math.inf)
            onehot.scatter_(-1, x.clamp(max=VOCAB - 1).unsqueeze(-1), 0.0)
            lp = torch.where(keep.unsqueeze(-1), onehot, lp)
        return lp


def ancestral(denoiser, prefix, length: int, nfe: int, generator):
    """Baseline sampler (see the module docstring). prefix: (B, P) LongTensor or None; returns (B, length)."""
    import torch
    B = prefix.shape[0] if prefix is not None else generator.batch
    dev = generator.device
    x = torch.full((B, length), MASK, dtype=torch.long, device=dev)
    if prefix is not None:
        x[:, :prefix.shape[1]] = prefix.to(dev)
    for k in range(nfe):
        t, s = 1.0 - k / nfe, 1.0 - (k + 1) / nfe
        masked = x == MASK
        if not masked.any():
            break
        lp = denoiser(x)
        probs = lp.exp()
        draw = torch.multinomial(probs.view(-1, VOCAB), 1, generator=generator.torch).view(B, length)
        p_unmask = 1.0 if k == nfe - 1 else (t - s) / t
        u = torch.rand(B, length, device=dev, generator=generator.torch)
        x = torch.where(masked & (u < p_unmask), draw, x)
    return x


class Gen:
    """What a sampler gets besides the denoiser: a seeded torch generator, the device and the batch size."""

    def __init__(self, seed: int, device: str, batch: int):
        import torch
        self.torch = torch.Generator(device=device).manual_seed(seed)
        self.device, self.batch = device, batch


def unigram_entropy(tokens: np.ndarray) -> float:
    """Entropy (nats) of the token frequency distribution within one sample."""
    _, c = np.unique(tokens, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def gen_nll(scorer, tokens: np.ndarray, start: int, device: str, batch: int = 8) -> np.ndarray:
    """Mean negative log-likelihood per token of tokens[:, start:] under an autoregressive scorer (GPT-2 family),
    conditioning on everything before; one value per sample. exp of it is the generative perplexity."""
    import torch
    out = []
    for i in range(0, len(tokens), batch):
        x = torch.as_tensor(tokens[i:i + batch], dtype=torch.long, device=device).clamp(max=VOCAB - 1)
        with torch.no_grad():
            logits = scorer(x).logits[:, :-1].float()
        lp = torch.gather(logits.log_softmax(-1), 2, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        lo = max(start - 1, 0)
        out.extend((-lp[:, lo:].mean(1)).cpu().tolist())
    return np.asarray(out)

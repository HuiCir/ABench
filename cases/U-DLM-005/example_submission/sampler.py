"""U-DLM-005 reference sampler: confidence-ordered unmasking with a cosine schedule (MaskGIT-style).

At every step the denoiser's distribution is sampled at all masked positions (temperature 1, so diversity is kept);
the sampled tokens with the highest probability are committed, as many as the cosine schedule prescribes for that
step (the number of masked positions after step k of K is round(n * cos(pi/2 * (k+1)/K))), and the rest stay masked.
"""
import math

MASK = 50257


def sample(denoiser, prefix, length, nfe, gen):
    import torch
    B = prefix.shape[0] if prefix is not None else gen.batch
    x = torch.full((B, length), MASK, dtype=torch.long, device=gen.device)
    if prefix is not None:
        x[:, :prefix.shape[1]] = prefix.to(gen.device)
    n0 = int((x == MASK).sum(1).max())
    for k in range(nfe):
        masked = x == MASK
        if not masked.any():
            break
        lp = denoiser(x)
        draw = torch.multinomial(lp.exp().view(-1, lp.shape[-1]), 1, generator=gen.torch).view(B, length)
        conf = torch.gather(lp, 2, draw.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(masked, conf, torch.full_like(conf, -math.inf))
        remain = 0 if k == nfe - 1 else int(round(n0 * math.cos(math.pi / 2 * (k + 1) / nfe)))
        n_commit = (masked.sum(1) - remain).clamp(min=0)
        order = conf.argsort(1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(length, device=x.device).expand(B, -1))
        commit = masked & (rank < n_commit.unsqueeze(1))
        x = torch.where(commit, draw, x)
    return x

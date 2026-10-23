"""A-LCTX-006 long-context measurements (evaluator code, agent-visible).

Long-document loss: windows of 16,384 GPT-2 tokens from PG-19 books (Rae et al. 2020, Apache-2.0). For a context
length L, the model reads the last L tokens of each window and the loss is the mean next-token loss on the window's
final 1,024 tokens, so the measure asks how well the model uses (or at least survives) L tokens of context.

Copy test: a sequence of length L of FineWeb text in which a 64-token segment from another document is inserted at a
random position in the first quarter and repeated at the very end; the loss on the repeat (tokens 2-64 of it) is low
only if the model retrieves the first occurrence across ~3/4 L tokens.
"""
from __future__ import annotations

import numpy as np

SEG, LONG_LENGTHS, COPY_LENGTHS, TAIL = 64, (4096, 8192, 16384), (8192, 16384), 1024


def make_copy(stream: np.ndarray, L: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n sequences of length L and the start of the repeated segment in each (always L - SEG)."""
    x = np.empty((n, L), np.int64)
    for i in range(n):
        o = int(rng.integers(0, len(stream) - L - 1))
        x[i] = stream[o:o + L]
        s = int(rng.integers(0, len(stream) - SEG - 1))
        seg = stream[s:s + SEG]
        p = int(rng.integers(16, L // 4))
        x[i, p:p + SEG] = seg
        x[i, L - SEG:] = seg
    return x, np.full(n, L - SEG, np.int64)


def tail_loss(model, windows: np.ndarray, L: int, device: str, tail: int = TAIL, batch: int = 1) -> np.ndarray:
    """Per-window mean loss on the last `tail` tokens, reading the last L tokens of each window."""
    import torch
    import torch.nn.functional as F
    out = []
    model.eval()
    for i in range(0, len(windows), batch):
        w = torch.as_tensor(windows[i:i + batch, -L - 1:], dtype=torch.long, device=device)
        x, y = w[:, :-1], w[:, 1:]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(x)[:, -tail:].float()
        l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, -tail:].reshape(-1), reduction="none")
        out.extend(l.view(len(x), -1).mean(1).cpu().tolist())
    return np.asarray(out)


def copy_loss(model, x: np.ndarray, start: np.ndarray, device: str) -> np.ndarray:
    """Per-sequence mean loss on tokens 2..SEG of the repeated segment."""
    import torch
    import torch.nn.functional as F
    out = []
    model.eval()
    for i in range(len(x)):
        t = torch.as_tensor(x[i:i + 1], dtype=torch.long, device=device)
        s = int(start[i])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(t[:, :-1])[:, s:].float()
        out.append(float(F.cross_entropy(logits[0], t[0, s + 1:])))
    return np.asarray(out)

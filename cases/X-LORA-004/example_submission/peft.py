"""X-LORA-004 reference submission, model code: LoRA (Hu et al. 2022) on the query and value projections of every layer of the
frozen roberta-base, rank 8, α = 16, plus a trainable classification head (dense + output layer on the <s> token), as
in the anchor's GLUE setup. Task-specific parameters: 294,912 (LoRA) + 592,130 (head) = 887,042. train.py trains them
and saves checkpoint/states.pt; predict() rebuilds the frozen encoder with LoRA and copies one task's state into it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RANK, ALPHA, MAX_LEN = 8, 16, 128


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


class Head(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.dense, self.out = nn.Linear(d, d), nn.Linear(d, k)

    def forward(self, h):
        return self.out(torch.tanh(self.dense(h[:, 0])))


def build(models_dir: str, device: str, k: int):
    from transformers import AutoModel, AutoTokenizer
    path = f"{models_dir}/roberta-base"
    tok = AutoTokenizer.from_pretrained(path)
    enc = AutoModel.from_pretrained(path, add_pooling_layer=False)
    for p in enc.parameters():
        p.requires_grad_(False)
    for layer in enc.encoder.layer:
        layer.attention.self.query = LoRALinear(layer.attention.self.query, RANK, ALPHA)
        layer.attention.self.value = LoRALinear(layer.attention.self.value, RANK, ALPHA)
    head = Head(enc.config.hidden_size, k)
    return tok, enc.to(device), head.to(device)


def trainable(enc, head) -> dict:
    st = {f"enc.{n}": p for n, p in enc.named_parameters() if n.endswith((".A", ".B"))}
    st.update({f"head.{n}": p for n, p in head.named_parameters()})
    return st


def batches(tok, s1, s2, device):
    kw = dict(truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    b = tok(s1, s2, **kw) if s2 is not None else tok(s1, **kw)
    return {k: v.to(device) for k, v in b.items()}


def predict(models_dir, task, state, s1, s2, device):
    k = state["head.out.weight"].shape[0]
    tok, enc, head = build(models_dir, device, k)
    named = dict(trainable(enc, head))
    with torch.no_grad():
        for n, v in state.items():
            named[n].copy_(v.to(device))
    enc.eval(); head.eval()
    preds = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for i in range(0, len(s1), 128):
            b = batches(tok, s1[i:i + 128], s2[i:i + 128] if s2 is not None else None, device)
            preds.append(head(enc(**b).last_hidden_state).float().argmax(-1).cpu())
    return torch.cat(preds).numpy()

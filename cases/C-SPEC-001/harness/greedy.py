"""C-SPEC-001 reference decoding (evaluator code): plain greedy decoding of the target model with its key-value cache,
recording the margin between the two largest logits at every step, and the tie-tolerant comparison of a submission's
output with it.

A submission's output is exact if it equals the reference tokens, or if it first differs at a step where the reference
margin is below TIE_MARGIN (two candidates practically tied in fp32, where a different but equally valid order of
floating-point operations may pick the other one); the comparison stops there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TIE_MARGIN = 1e-3


def eos_ids(model_dir: str) -> list[int]:
    g = json.loads((Path(model_dir) / "generation_config.json").read_text())
    e = g.get("eos_token_id", [])
    return [int(x) for x in (e if isinstance(e, list) else [e])]


def load_target(model_dir: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True,
                                             attn_implementation="sdpa").to(device).eval()
    return m


def greedy(model, prompt: list[int], max_new: int, eos: list[int], device: str):
    """(tokens, margins) of plain greedy decoding; stops after an EOS token or max_new tokens."""
    import torch
    from transformers import DynamicCache
    cache = DynamicCache()
    ids = torch.tensor([prompt], device=device)
    out, margins = [], []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0, -1].float()
            top = torch.topk(logits, 2)
            tok = int(top.indices[0])
            margins.append(float(top.values[0] - top.values[1]))
            out.append(tok)
            if tok in eos:
                break
            ids = torch.tensor([[tok]], device=device)
    return out, margins


def compare(sub: list[int], ref: list[int], margins: list[float]) -> tuple[bool, int]:
    """(exact, number of matching leading tokens)."""
    n = 0
    for a, b in zip(sub, ref):
        if a != b:
            return bool(margins[n] < TIE_MARGIN), n
        n += 1
    return len(sub) == len(ref), n

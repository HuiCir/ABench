"""Reference submission: speculative decoding with the provided draft model (Leviathan et al. 2023; Chen et al. 2023),
greedy verification, and a draft length adapted to the recent acceptance rate.

Each round the draft (Qwen2.5-0.5B-Instruct, bf16) proposes k tokens greedily; the target (Qwen2.5-1.5B-Instruct,
fp32, as in the evaluator's reference) scores the k proposals and one more position in a single forward pass; the
longest agreeing prefix is kept plus the target's own next token, so the output equals the target's greedy output.
Both key-value caches are cropped back to the accepted length. k starts at 4 and moves between 1 and 8: +1 after a
fully accepted round, -1 after a round that accepted less than half (a simple form of the adaptive draft lengths of
later work).
"""
from __future__ import annotations

import json
from pathlib import Path

TARGET, DRAFT = "Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct"
K0, K_MIN, K_MAX = 4, 1, 8


def load(checkpoint_dir: str, models_dir: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())

    def m(name, dtype):
        return AutoModelForCausalLM.from_pretrained(str(Path(models_dir) / name), torch_dtype=dtype, local_files_only=True,
                                                    attn_implementation="sdpa").to(device).eval()
    g = json.loads((Path(models_dir) / TARGET / "generation_config.json").read_text())
    eos = g["eos_token_id"] if isinstance(g["eos_token_id"], list) else [g["eos_token_id"]]
    return {"target": m(TARGET, torch.float32), "draft": m(DRAFT, torch.bfloat16), "device": device,
            "eos": set(int(e) for e in eos), "k0": int(cfg.get("k0", K0))}


def generate(model, prompt: list, max_new: int) -> list:
    import torch
    from transformers import DynamicCache
    tgt, drf, dev, eos = model["target"], model["draft"], model["device"], model["eos"]
    tc, dc = DynamicCache(), DynamicCache()
    ids = torch.tensor([prompt], device=dev)
    nxt = int(tgt(input_ids=ids, past_key_values=tc, use_cache=True).logits[0, -1].argmax())
    if len(prompt) > 1:                             # draft cache up to the last prompt token
        drf(input_ids=ids[:, :-1], past_key_values=dc, use_cache=True)
    out = [nxt]
    k = model["k0"]
    last = ids[0, -1:].tolist()                     # tokens the draft has not yet seen (its cache ends before them)
    while len(out) < max_new and out[-1] not in eos:
        k = max(K_MIN, min(K_MAX, k, max_new - len(out)))
        feed = last + [out[-1]]
        props = []
        x = torch.tensor([feed], device=dev)
        for _ in range(k):
            t = int(drf(input_ids=x, past_key_values=dc, use_cache=True).logits[0, -1].argmax())
            props.append(t)
            x = torch.tensor([[t]], device=dev)
        ver = torch.tensor([[out[-1]] + props], device=dev)
        logits = tgt(input_ids=ver, past_key_values=tc, use_cache=True).logits[0].float()
        best = logits.argmax(-1).tolist()          # target's choice after out[-1], props[0], ..., props[k-1]
        n = 0
        while n < k and props[n] == best[n]:
            n += 1
        new = props[:n] + [best[n]]
        for t in new:
            out.append(t)
            if t in eos or len(out) >= max_new:
                break
        acc = len(prompt) + len(out) - 1           # tokens whose keys and values both caches should hold
        tc.crop(acc)
        dc.crop(min(dc.get_seq_length(), acc))
        last = out[dc.get_seq_length() - len(prompt):-1] if dc.get_seq_length() >= len(prompt) else []
        k = k + 1 if n == k else (k - 1 if n < k / 2 else k)
    return out[:max_new]

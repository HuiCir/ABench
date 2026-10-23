"""Shared evaluator code of the post-training cases on SmolLM2 (X-SFT-005, X-DPO-006, X-DISTILL-010; identical copy in
each, agent-visible): chat formatting, greedy generation, sequence log-probabilities, held-out loss and parameter counts.

The submission's load() returns a causal LM with the interface of a transformers model: forward(input_ids,
attention_mask=None) -> object with .logits (B, T, V), and generate() as in transformers. The evaluator, not the
submission, formats prompts and decodes, so every submission is compared under the same template and decoding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"     # ChatML, as used by SmolLM2-Instruct
STOP = ("<|im_end|>", "<|endoftext|>")


def tokenizer(base_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def stop_ids(tok) -> list:
    return [i for i in (tok.convert_tokens_to_ids(s) for s in STOP) if i is not None and i >= 0]


def format_prompt(prompt: str) -> str:
    return CHAT.format(prompt=prompt)


def generate(model, tok, prompts: list, max_new: int, device: str, batch: int = 32) -> list:
    """Greedy decoding of CHAT-formatted prompts, stopping at <|im_end|> or <|endoftext|>; returns the decoded
    responses without special tokens. Prompts are batched by length."""
    import torch
    order = np.argsort([len(p) for p in prompts])
    out = [""] * len(prompts)
    stops = stop_ids(tok)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([format_prompt(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True).strip()
    return out


def response_logprob(model, tok, prompts: list, responses: list, device: str, batch: int = 16) -> np.ndarray:
    """Sum of log-probabilities of each response's tokens (and the closing <|im_end|>) given the formatted prompt."""
    import torch
    out = np.zeros(len(prompts))
    end = tok.convert_tokens_to_ids("<|im_end|>")
    for s in range(0, len(prompts), batch):
        seqs, starts = [], []
        for p, r in zip(prompts[s:s + batch], responses[s:s + batch]):
            a = tok(format_prompt(p), add_special_tokens=False)["input_ids"]
            b = tok(r, add_special_tokens=False)["input_ids"] + [end]
            seqs.append(a + b)
            starts.append(len(a))
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, x in enumerate(seqs):                     # right padding for log-probabilities
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        tgt = ids[:, 1:]
        lp = logp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for i, (x, st) in enumerate(zip(seqs, starts)):
            out[s + i] = float(lp[i, st - 1:len(x) - 1].sum())
    return out


def heldout_nll(model, tokens: np.ndarray, device: str, seq: int = 1024, batch: int = 8) -> float:
    """Mean next-token cross-entropy (nats) over non-overlapping windows of `seq` + 1 tokens."""
    import torch
    n = (len(tokens) - 1) // seq
    tot, cnt = 0.0, 0
    for s in range(0, n, batch):
        rows = [tokens[i * seq:i * seq + seq + 1].astype(np.int64) for i in range(s, min(n, s + batch))]
        x = torch.from_numpy(np.stack(rows)).to(device)
        with torch.no_grad():
            logits = model(input_ids=x[:, :-1]).logits.float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
        tot += float(loss)
        cnt += x[:, 1:].numel()
    return tot / max(cnt, 1)


def checkpoint_numbers(ck: Path) -> int:
    """Numbers stored in the checkpoint's tensor files (.safetensors, .pt, .bin): the trained-parameter cost."""
    import torch
    n = 0
    for p in sorted(Path(ck).rglob("*")):
        if p.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(p), "pt") as f:
                for k in f.keys():
                    n += int(np.prod(f.get_slice(k).get_shape()))
        elif p.suffix in (".pt", ".bin"):
            obj = torch.load(p, map_location="cpu", weights_only=True)
            stack = [obj]
            while stack:
                o = stack.pop()
                if torch.is_tensor(o):
                    n += o.numel()
                elif isinstance(o, dict):
                    stack += list(o.values())
                elif isinstance(o, (list, tuple)):
                    stack += list(o)
    return n


def load_base(base_dir: str, device: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True,
                                                torch_dtype=getattr(torch, dtype)).to(device).eval()

"""Reference training for X-DPO-006: direct preference optimisation (beta 0.1) of rank-16 LoRA updates on
SmolLM2-360M-Instruct, the starting model as reference policy, one epoch over 20,000 public pairs (answers cut at 512
tokens), ChatML formatting as the evaluator uses, AdamW (lr 5e-5), batch 8 pairs.

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--quick] [--limit N]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src import lora  # noqa: E402

CHAT = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def unpack(u, o):
    b = np.asarray(u, np.uint8).tobytes()
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]


def seq_logp(model, tok, prompts, answers, dev):
    """Sum of answer-token log-probabilities, as lmpost.response_logprob, with gradients."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    seqs, starts = [], []
    for p, a in zip(prompts, answers):
        x = tok(CHAT.format(p=p), add_special_tokens=False)["input_ids"][-512:]
        y = tok(a, add_special_tokens=False)["input_ids"][:512] + [end]
        seqs.append(x + y)
        starts.append(len(x))
    L = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
    lp = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(lp)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        mask[i, st - 1:len(s) - 1] = 1
    return (lp * mask).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20_000)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    base = f"{os.environ.get('ABENCH_MODELS_DIR', 'models')}/SmolLM2-360M-Instruct"
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    pol = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16).to(dev)
    ref = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16).to(dev).eval()
    lora.attach(pol, 16, 32.0)
    with np.load(Path(a.data) / "prefs_public.npz", allow_pickle=False) as z:
        P = unpack(z["train_prompt_utf8"], z["train_prompt_offsets"])
        C = unpack(z["train_chosen_utf8"], z["train_chosen_offsets"])
        R = unpack(z["train_rejected_utf8"], z["train_rejected_offsets"])
    n = 32 if a.quick else min(a.limit, len(P))
    order = np.random.default_rng(0).permutation(len(P))[:n]
    params = [p for p in pol.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5)
    bs = 8
    pol.train()
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        p, c, r = [P[i] for i in idx], [C[i] for i in idx], [R[i] for i in idx]
        with torch.no_grad():
            rc, rr = seq_logp(ref, tok, p, c, dev), seq_logp(ref, tok, p, r, dev)
        pc, pr = seq_logp(pol, tok, p, c, dev), seq_logp(pol, tok, p, r, dev)
        loss = -torch.nn.functional.logsigmoid(a.beta * ((pc - rc) - (pr - rr))).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if (s // bs) % 200 == 0:
            print(f"pairs {s}/{n} loss {loss.item():.3f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_file(lora.state(pol), str(Path(a.out) / "lora.safetensors"))
    (Path(a.out) / "lora.json").write_text(json.dumps({"r": 16, "alpha": 32.0}))


if __name__ == "__main__":
    main()

"""Reference training for X-SFT-005: one epoch of supervised fine-tuning with LoRA (rank 16, alpha 32) on the public
conversations, loss on assistant turns only, ChatML formatting as the evaluator uses, sequences up to 1,024 tokens,
AdamW (lr 2e-4, cosine), batch 16.

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--quick] [--limit N]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src import lora  # noqa: E402


def unpack(u, o):
    b = np.asarray(u, np.uint8).tobytes()
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]


def encode(tok, conv, max_len=1024):
    ids, mask = [], []
    for m in conv:
        role = "assistant" if m["role"] == "assistant" else "user"
        if m["role"] == "system":
            continue
        head = tok(f"<|im_start|>{role}\n", add_special_tokens=False)["input_ids"]
        body = tok(m["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids += head + body
        mask += [0] * len(head) + ([1] * len(body) if role == "assistant" else [0] * len(body))
    return ids[:max_len], mask[:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50_000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    models = os.environ.get("ABENCH_MODELS_DIR", "models")
    tok = AutoTokenizer.from_pretrained(f"{models}/SmolLM2-360M", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(f"{models}/SmolLM2-360M", local_files_only=True, torch_dtype=torch.bfloat16).to(dev)
    lora.attach(model, 16, 32.0)
    with np.load(Path(a.data) / "sft_public.npz", allow_pickle=False) as z:
        convs = [json.loads(x) for x in unpack(z["conv_utf8"], z["conv_offsets"])]
    convs = convs[:64 if a.quick else a.limit]
    data = [encode(tok, c) for c in convs]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    bs = 16
    order = np.random.default_rng(0).permutation(len(data))
    steps = math.ceil(len(order) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    for st in range(steps):
        batch = [data[i] for i in order[st * bs:(st + 1) * bs]]
        L = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, m) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
            lab[i, :len(x)] = torch.where(torch.tensor(m) == 1, torch.tensor(x), torch.tensor(-100))
        out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if st % 200 == 0:
            print(f"step {st}/{steps} loss {out.loss.item():.3f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_file(lora.state(model), str(Path(a.out) / "lora.safetensors"))
    (Path(a.out) / "lora.json").write_text(json.dumps({"r": 16, "alpha": 32.0}))


if __name__ == "__main__":
    main()

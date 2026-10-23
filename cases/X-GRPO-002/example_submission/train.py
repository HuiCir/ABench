"""X-GRPO-002 reference submission: GRPO (Shao et al. 2024) with LoRA on Qwen2.5-0.5B-Instruct.

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--steps 200] [--quick]

Each step: 16 prompts (half Countdown instances generated with a training seed, half GSM8K training problems), 8
sampled responses per prompt (temperature 1.0, at most 384 new tokens), reward 1 for a verified answer, 0.1 for a
well-formed but wrong one, 0 otherwise; advantages are the rewards standardised within each prompt's group; one
on-policy update of the token-level policy-gradient loss (the GRPO objective with ratio 1 and without a KL term),
averaged over the group's tokens; AdamW, learning rate 2e-5 on the LoRA parameters (rank 16 on every projection).
The model's prompt format is the evaluator's (ChatML, as in interface/lmpost.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import lora  # noqa: E402
from src import verify_math as V  # noqa: E402

CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
MODEL = "Qwen2.5-0.5B-Instruct"


def reward(kind: str, spec: dict, response: str) -> float:
    a = V.extract(response)
    if a is None:
        return 0.0
    ok = V.check_countdown(a, spec["nums"], spec["target"]) if kind == "countdown" else V.check_gsm8k(a, spec["gold"])
    return 1.0 if ok else 0.1


def main():
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="models dir (default: $ABENCH_MODELS_DIR)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    import os
    models = a.models or os.environ.get("ABENCH_MODELS_DIR", "")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(f"{models}/{MODEL}")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(f"{models}/{MODEL}", torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    lora.attach(model, 16, 32.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.0)
    gsm = json.loads((Path(a.data) / "gsm8k_train.json").read_text())
    rng = np.random.default_rng(0)
    steps, n_prompt, group, max_new = (3, 4, 4, 64) if a.quick else (a.steps, 16, 8, 384)
    end = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    for step in range(steps):
        cd = V.make_countdown(rng, n_prompt // 2)
        items = [("countdown", c, V.countdown_prompt(c)) for c in cd]
        items += [("gsm8k", g, V.gsm8k_prompt(g["question"])) for g in (gsm[i] for i in rng.integers(0, len(gsm), n_prompt - len(cd)))]
        prompts = [CHAT.format(prompt=p) for _, _, p in items for _ in range(group)]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        model.eval()
        with torch.no_grad():
            seq = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=max_new,
                                 eos_token_id=end, pad_token_id=tok.pad_token_id)
        resp_ids = seq[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(resp_ids, skip_special_tokens=True)
        r = np.asarray([reward(k, s, t) for (k, s, _), t in zip([it for it in items for _ in range(group)], texts)])
        rg = r.reshape(-1, group)
        adv = ((rg - rg.mean(1, keepdims=True)) / (rg.std(1, keepdims=True) + 1e-4)).reshape(-1)
        if not np.any(adv):
            continue
        model.train()
        mask = (resp_ids != tok.pad_token_id).float()
        full = seq
        attn = torch.cat([enc["attention_mask"], mask.long()], 1)
        logits = model(full, attention_mask=attn).logits[:, enc["input_ids"].shape[1] - 1:-1].float()
        lp = torch.gather(logits.log_softmax(-1), 2, resp_ids.unsqueeze(-1)).squeeze(-1)
        A = torch.tensor(adv, device=dev, dtype=torch.float32).unsqueeze(1)
        loss = -(A * lp * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step} reward {r.mean():.3f} acc {(r == 1).mean():.3f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(lora.state(model), str(out / "lora.safetensors"))
    (out / "lora.json").write_text(json.dumps({"r": 16, "alpha": 32.0, "base": MODEL}))


if __name__ == "__main__":
    main()

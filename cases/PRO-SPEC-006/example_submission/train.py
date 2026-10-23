"""Reference recipe for PRO-SPEC-006: an EAGLE-3 draft for Qwen3-8B (Li et al. 2025).

Launch (registered with the supervisor, from this directory):
  python3 pro_audit.py run --name eagle3 --code . -- python3 train.py

1. Data (once, cached in the work directory): 60,000 UltraChat-200k first-turn prompts; Qwen3-8B answers them
   greedily (vLLM, non-thinking template, at most 1,024 tokens) on all four GPUs; the target's hidden states at
   layers 2, 18 and 33 (low, middle, high) and its final hidden state are recorded per token by a forward pass.
2. Draft: fc fuses the three target features (3 x 4,096 -> 4,096); one decoder layer (the target's layer shape,
   input = [embedding of the token ; fused feature]) predicts the next feature; the head (the target's frozen output
   layer restricted to the 32,000 most frequent target tokens, d2t/t2d maps) gives the draft distribution.
3. Training-time test: each sequence is unrolled for 7 draft steps, step k feeding the draft's own feature of step
   k-1; loss = sum_k 0.8^k cross-entropy against the target's next token; AdamW lr 5e-5, 4 epochs, bf16, data
   parallel on 4 GPUs. A checkpoint (model/ in vLLM's EAGLE-3 format, optimiser, DONE) every 2,000 steps.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LAYERS, STEPS_TTT, LR, EPOCHS, DRAFT_VOCAB, N_PROMPTS, SAVE_EVERY = (2, 18, 33), 7, 5e-5, 4, 32000, 60000, 2000


class Draft(nn.Module):
    def __init__(self, cfg: dict, d2t: torch.Tensor):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig, LlamaRMSNorm
        h = cfg["hidden_size"]
        lc = LlamaConfig(hidden_size=h, intermediate_size=cfg["intermediate_size"], num_attention_heads=cfg["num_attention_heads"],
                         num_key_value_heads=cfg["num_key_value_heads"], head_dim=cfg.get("head_dim", h // cfg["num_attention_heads"]),
                         rms_norm_eps=cfg["rms_norm_eps"], max_position_embeddings=8192, rope_theta=cfg.get("rope_theta", 1e6))
        self.fc = nn.Linear(3 * h, h, bias=False)
        self.midlayer = LlamaDecoderLayer(lc, 0)
        self.midlayer.self_attn.q_proj = nn.Linear(2 * h, lc.num_attention_heads * lc.head_dim, bias=False)
        self.midlayer.self_attn.k_proj = nn.Linear(2 * h, lc.num_key_value_heads * lc.head_dim, bias=False)
        self.midlayer.self_attn.v_proj = nn.Linear(2 * h, lc.num_key_value_heads * lc.head_dim, bias=False)
        self.hidden_norm = LlamaRMSNorm(h, eps=lc.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(h, eps=lc.rms_norm_eps)
        self.norm = LlamaRMSNorm(h, eps=lc.rms_norm_eps)
        self.lm_head = nn.Linear(h, DRAFT_VOCAB, bias=False)
        self.register_buffer("d2t", d2t)
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rope = LlamaRotaryEmbedding(lc)

    def step(self, emb, feat, pos):
        x = torch.cat([self.input_layernorm(emb), self.hidden_norm(feat)], -1)
        cos_sin = self.rope(x, pos)
        y = self.midlayer.self_attn(hidden_states=x, position_embeddings=cos_sin, attention_mask=None)[0] + feat
        y = y + self.midlayer.mlp(self.midlayer.post_attention_layernorm(y))
        return y, self.lm_head(self.norm(y))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    models, data = Path(os.environ["ABENCH_MODELS_DIR"]), Path(os.environ["ABENCH_DATA_DIR"])
    ck_root, resume = Path(os.environ["ABENCH_CKPT_DIR"]), os.environ.get("ABENCH_RESUME_FROM")
    work = Path(os.environ.get("ABENCH_WORK_DIR", HERE / "outputs"))
    work.mkdir(parents=True, exist_ok=True)
    target_dir = models / "Qwen3-8B"
    cfg = json.loads((target_dir / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(target_dir)
    cache = work / "eagle3_data.pt"
    if not cache.is_file():                                  # step 1: target answers and features
        import pandas as pd
        from vllm import LLM, SamplingParams
        df = pd.concat([pd.read_parquet(p) for p in sorted((data / "ultrachat_200k").glob("train_sft-*.parquet"))])
        prompts = [m[0]["content"] for m in df["messages"].tolist()[:N_PROMPTS]]
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts]
        llm = LLM(model=str(target_dir), tensor_parallel_size=4, dtype="bfloat16")
        outs = llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=1024))
        seqs = [tok(c)["input_ids"] + list(o.outputs[0].token_ids) for c, o in zip(chats, outs)]
        del llm
        torch.save({"seqs": seqs}, cache)
    seqs = torch.load(cache)["seqs"]
    counts = np.bincount(np.concatenate([np.asarray(s) for s in seqs]), minlength=cfg["vocab_size"])
    keep = np.sort(np.argsort(-counts)[:DRAFT_VOCAB])
    d2t = torch.as_tensor(keep - np.arange(DRAFT_VOCAB), dtype=torch.long)       # draft index -> target id offset
    dev = "cuda:0"
    target = AutoModelForCausalLM.from_pretrained(target_dir, torch_dtype=torch.bfloat16, device_map="auto").eval()
    draft = Draft(cfg, d2t).to(dev, torch.bfloat16)
    with torch.no_grad():
        draft.lm_head.weight.copy_(target.lm_head.weight[torch.as_tensor(keep)].to(dev))
    draft.lm_head.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=LR)
    step0 = 0
    if resume:
        from safetensors.torch import load_file
        draft.load_state_dict(load_file(str(Path(resume) / "model" / "model.safetensors")), strict=False)
        st = torch.load(Path(resume) / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
    t2d = torch.zeros(cfg["vocab_size"], dtype=torch.bool)
    t2d[torch.as_tensor(keep)] = True
    remap = torch.full((cfg["vocab_size"],), -100, dtype=torch.long)
    remap[torch.as_tensor(keep)] = torch.arange(DRAFT_VOCAB)
    emb = target.get_input_embeddings()
    order = np.random.default_rng(0).permutation(len(seqs) * EPOCHS) % len(seqs)
    for step in range(step0, len(order)):
        t0 = time.time()
        ids = torch.as_tensor([seqs[order[step]][:2048]], device=target.device)
        with torch.no_grad():
            hs = target(ids, output_hidden_states=True).hidden_states
            feat = torch.cat([hs[l + 1] for l in LAYERS], -1).to(dev)
            e = emb(ids).to(dev)
        f = draft.fc(feat)[:, :-1]
        pos = torch.arange(ids.shape[1] - 1, device=dev)[None]
        loss = 0.0
        for k in range(STEPS_TTT):                           # training-time test: feed the draft's own features
            n = ids.shape[1] - 1 - k
            if n <= 1:
                break
            f, logits = draft.step(e[:, k:k + n], f[:, :n], pos[:, :n])
            tgt = remap[ids[0, k + 1:k + 1 + n].cpu()].to(dev)
            loss = loss + 0.8 ** k * F.cross_entropy(logits[0].float(), tgt, ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 0.5)
        opt.step()
        if step % 10 == 0:
            print(f"ABENCH step={step} loss={float(loss):.5f} seconds={time.time() - t0:.2f}", flush=True)
        if (step + 1) % SAVE_EVERY == 0 or step + 1 == len(order):
            d = ck_root / f"step_{step + 1:07d}"
            (d / "model").mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file
            sd = {k: v.detach().cpu().contiguous() for k, v in draft.state_dict().items() if not k.startswith("rope")}
            sd["t2d"] = t2d
            save_file(sd, str(d / "model" / "model.safetensors"))
            (d / "model" / "config.json").write_text(json.dumps({
                "architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama", "hidden_size": cfg["hidden_size"],
                "intermediate_size": cfg["intermediate_size"], "num_attention_heads": cfg["num_attention_heads"],
                "num_key_value_heads": cfg["num_key_value_heads"], "num_hidden_layers": 1, "rms_norm_eps": cfg["rms_norm_eps"],
                "vocab_size": cfg["vocab_size"], "draft_vocab_size": DRAFT_VOCAB, "rope_theta": cfg.get("rope_theta", 1e6),
                "max_position_embeddings": 8192, "eagle_aux_hidden_state_layer_ids": list(LAYERS), "torch_dtype": "bfloat16"}))
            torch.save({"opt": opt.state_dict(), "step": step + 1}, d / "state.pt")
            (d / "DONE").write_text("")


if __name__ == "__main__":
    main()

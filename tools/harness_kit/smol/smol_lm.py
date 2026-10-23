"""SmolLM2-135M in plain PyTorch with replaceable token mixers and an explicit decoding cache (evaluator code;
identical copy in each case that uses it).

Architecture (HuggingFaceTB/SmolLM2-135M, Llama layout): 30 layers of width 576, attention with 9 query heads and 3
key-value heads of dimension 64, rotary position embedding (theta 100,000, the rotate-half convention of Hugging Face
Llama), SwiGLU MLP of width 1,536, RMSNorm (eps 1e-5), tied embeddings, vocabulary 49,152.

  model = load_base(base_dir, device)        # the pretrained model, weights from model.safetensors
  model.layers[i].mixer = MyMixer(...)       # any module with the Mixer interface below
  logits = model(idx)                        # full causal forward (training, evaluation)
  logits, cache = model.prefill(idx)         # forward that also returns the decoding cache
  logits, cache = model.step(idx_1, cache)   # one new token per sequence
  cache_bytes(cache)                         # bytes of every tensor held in the cache

Mixer interface: forward(x, pos0, cache) -> (y, new_cache) with x (B, T, d) normalised input, pos0 the position of
x[:, 0] (0 for a full forward or a prefill), cache None or what the same mixer returned before. The cache must hold
everything the mixer needs to continue the sequence; it may be any nesting of tuples, lists, dicts and tensors.
"""
from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 49152
    d_model: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    head_dim: int = 64
    d_ff: int = 1536
    rope_theta: float = 100000.0
    eps: float = 1e-5
    max_seq: int = 8192


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(seq: int, dim: int, theta: float, device, pos0: int = 0):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(pos0, pos0 + seq, device=device).float()
    f = torch.outer(t, inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], -1)


def apply_rope(x, cos, sin):                       # x (B, H, T, D); cos, sin (T, D)
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


class GQAttention(nn.Module):
    """The original attention with a key-value cache of shape (B, kv_heads, T, head_dim) per layer."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def qkv(self, x, pos0: int):
        c = self.c
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        cos, sin = rope(T, c.head_dim, c.rope_theta, x.device, pos0)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(self, q, k, v, causal: bool):
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        B, H, T, D = y.shape
        return self.o_proj(y.transpose(1, 2).reshape(B, T, H * D))

    def forward(self, x, pos0: int = 0, cache=None):
        q, k, v = self.qkv(x, pos0)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        causal = cache is None
        return self.attend(q, k, v, causal and q.shape[2] > 1), (k, v)


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down_proj = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(c.d_model, c.eps)
        self.mixer = GQAttention(c)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, pos0: int = 0, cache=None):
        y, cache = self.mixer(self.input_layernorm(x), pos0, cache)
        x = x + y
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class SmolLM(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)

    def _run(self, idx, pos0: int, caches):
        x = self.embed_tokens(idx)
        new = []
        for i, layer in enumerate(self.layers):
            x, ci = layer(x, pos0, None if caches is None else caches[i])
            new.append(ci)
        return F.linear(self.norm(x), self.embed_tokens.weight), new

    def forward(self, idx):
        return self._run(idx, 0, None)[0]

    def loss(self, idx, targets):
        logits = self(idx)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    def prefill(self, idx):
        logits, caches = self._run(idx, 0, None)
        return logits, {"pos": idx.shape[1], "layers": caches}

    def step(self, idx, cache):
        logits, caches = self._run(idx, cache["pos"], cache["layers"])
        return logits, {"pos": cache["pos"] + idx.shape[1], "layers": caches}


HF_NAMES = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}


def hf_name(name: str) -> str:
    if name in HF_NAMES:
        return HF_NAMES[name]
    name = name.replace(".mixer.", ".self_attn.")
    return "model." + name


def load_base(base_dir: str, device: str = "cpu", dtype=torch.float32) -> SmolLM:
    from safetensors.torch import load_file
    sd = load_file(str(Path(base_dir) / "model.safetensors"))
    model = SmolLM()
    own = model.state_dict()
    missing = [k for k in own if hf_name(k) not in sd]
    if missing:
        raise KeyError(f"weights missing in model.safetensors: {missing[:5]}")
    model.load_state_dict({k: sd[hf_name(k)].to(own[k].dtype) for k in own})
    return model.to(device=device, dtype=dtype)


def cache_bytes(obj) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(cache_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(cache_bytes(v) for v in obj)
    return 0


def kv_bytes_per_token(model: SmolLM, device: str, t1: int = 1024, t2: int = 2048, batch: int = 1) -> float:
    """Growth of the cache in bytes per token of context between t1 and t2 tokens (random tokens, batch 1)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(1, model.c.vocab_size, (batch, t2), generator=g).to(device)
    with torch.no_grad():
        b1 = cache_bytes(model.prefill(x[:, :t1])[1])
        b2 = cache_bytes(model.prefill(x)[1])
    return (b2 - b1) / ((t2 - t1) * batch)


def decode_consistency(model: SmolLM, idx, n_new: int = 32) -> float:
    """Largest relative difference between logits from prefill + step-by-step decoding and from the full forward."""
    with torch.no_grad():
        full = model(idx).float()
        T = idx.shape[1] - n_new
        logits, cache = model.prefill(idx[:, :T])
        steps = [logits[:, -1:].float()]
        for t in range(T, idx.shape[1] - 1):
            lg, cache = model.step(idx[:, t:t + 1], cache)
            steps.append(lg.float())
        inc = torch.cat(steps, 1)
    ref = full[:, T - 1: idx.shape[1] - 1]
    return float((inc - ref).abs().max() / (ref.abs().max() + 1e-6))



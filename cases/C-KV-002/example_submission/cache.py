"""Reference submission: per-token asymmetric quantisation of keys and values (a simplified KIVI, Liu et al. 2024).

Every cached key and value vector (one head, one token: 128 numbers) is quantised to b bits with its own minimum and
scale in fp16 (0.25 bits of overhead per element) and stored bit-packed in uint8. Budgets map to bit widths:
4.5 -> keys 4 / values 4 bits (4.25 stored bits), 2.5 -> 2 / 2 (2.25), 2.0 -> keys 2 / values 1 (1.75).
Unlike KIVI, keys are quantised per token rather than per channel and there is no full-precision window of recent
tokens; both would help accuracy at 2 bits but cost bytes at the shorter contexts.
Written against the transformers DynamicCache interface (update / get_seq_length); the operator checks it against the
frozen transformers version at activation.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

BITS = {4.5: (4, 4), 2.5: (2, 2), 2.0: (2, 1)}


def quantize(x: torch.Tensor, bits: int):
    """x (B, H, T, D) -> packed uint8 (B, H, T, D*bits/8), minimum and scale (B, H, T, 1) in fp16."""
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    q = ((x - lo) / scale).round().clamp(0, 2 ** bits - 1).to(torch.uint8)
    per = 8 // bits
    q = q.view(*q.shape[:-1], -1, per)
    packed = torch.zeros(q.shape[:-1], dtype=torch.uint8, device=x.device)
    for i in range(per):
        packed |= q[..., i] << (bits * i)
    return packed, lo.half(), scale.half()


def dequantize(packed, lo, scale, bits: int, dtype):
    per = 8 // bits
    mask = 2 ** bits - 1
    q = torch.stack([(packed >> (bits * i)) & mask for i in range(per)], -1).flatten(-2)
    return (q.to(torch.float32) * scale.float() + lo.float()).to(dtype)


class QuantCache(DynamicCache):
    def __init__(self, kbits: int, vbits: int):
        super().__init__()
        self.kbits, self.vbits = kbits, vbits
        self.store: list = []                     # per layer: [k_parts, v_parts], each a list of (packed, lo, scale)
        self._len = 0

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        while len(self.store) <= layer_idx:
            self.store.append([[], []])
        ks, vs = self.store[layer_idx]
        ks.append(quantize(key_states, self.kbits))
        vs.append(quantize(value_states, self.vbits))
        if layer_idx == 0:
            self._len += key_states.shape[-2]
        k = torch.cat([dequantize(*p, self.kbits, key_states.dtype) for p in ks], -2)
        v = torch.cat([dequantize(*p, self.vbits, value_states.dtype) for p in vs], -2)
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._len


def make_cache(budget_bits: float, config):
    kb, vb = BITS[budget_bits]
    return QuantCache(kb, vb)

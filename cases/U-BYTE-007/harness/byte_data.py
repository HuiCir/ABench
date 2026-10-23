"""U-BYTE-007 byte data and character-level noise (evaluator and agent use this same file).

The text is FineWeb (the B-OPT-001 shards) decoded from GPT-2 tokens back to UTF-8 bytes; document boundaries
(<|endoftext|>) are the byte 0, which does not occur in the text. Files are raw uint8 with no header.

add_noise(b, rate, seed): character-level noise on ASCII letters — each letter is, with probability `rate`, replaced
by a random letter, has its case swapped, or is deleted (one third each).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BOUNDARY = 0


def read_bytes(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint8, mode="r")


def add_noise(b: np.ndarray, rate: float, seed: int) -> np.ndarray:
    b = np.asarray(b, np.uint8)
    rng = np.random.default_rng(seed)
    letter = ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122))
    hit = letter & (rng.random(len(b)) < rate)
    op = rng.integers(0, 3, len(b))
    out = b.copy()
    sub = hit & (op == 0)
    rnd = rng.integers(0, 52, len(b))
    out[sub] = np.where(rnd[sub] < 26, 65 + rnd[sub], 97 + rnd[sub] - 26).astype(np.uint8)
    case = hit & (op == 1)
    out[case] = out[case] ^ 0x20
    keep = ~(hit & (op == 2))
    return out[keep]

"""T-MM-004 helpers (evaluator code, agent-visible): packed JPEG images and the retrieval and compositionality scores.

Images are JPEG files (shorter side resized to 256 and centre-cropped to 256×256, quality 90) stored as one uint8
array plus offsets: image i is jpeg[offsets[i]:offsets[i+1]]. Captions are packed UTF-8 (textpack.py).
"""
from __future__ import annotations

import numpy as np


def jpegs(buf: np.ndarray, offsets: np.ndarray, idx=None) -> list:
    rows = range(len(offsets) - 1) if idx is None else idx
    b = np.asarray(buf, np.uint8)
    return [b[offsets[i]:offsets[i + 1]].tobytes() for i in rows]


def pack_bytes(xs) -> tuple:
    off = np.zeros(len(xs) + 1, np.int64)
    np.cumsum([len(x) for x in xs], out=off[1:])
    return np.frombuffer(b"".join(xs), np.uint8).copy(), off


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def retrieval(img: np.ndarray, txt: np.ndarray, txt_img: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Recall@K for image→text (an image is hit if any of its captions is in the top K) and text→image, by cosine
    similarity; ties are broken against the right answer (the target must be strictly better than the K-th)."""
    s = normalise(img) @ normalise(txt).T                       # (n_img, n_txt)
    out = {}
    # text → image: rank of the caption's own image
    own = s[txt_img, np.arange(len(txt_img))]
    better = (s >= own[None, :]).sum(0) - 1                     # images at least as similar, excluding itself
    for k in ks:
        out[f"t2i_r{k}"] = float(np.mean(better < k))
    # image → text: best rank among the image's captions
    best = np.full(len(img), np.inf)
    for j, i in enumerate(txt_img):
        r = (s[i] >= s[i, j]).sum() - 1
        best[i] = min(best[i], r)
    for k in ks:
        out[f"i2t_r{k}"] = float(np.mean(best < k))
    out["recall_mean"] = float(np.mean([out[f"{d}_r{k}"] for d in ("i2t", "t2i") for k in ks]))
    return out


def compositional(img: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """1 where the true caption is strictly more similar to the image than the hard negative."""
    a, p, n = normalise(img), normalise(pos), normalise(neg)
    return ((a * p).sum(1) > (a * n).sum(1)).astype(np.float64)

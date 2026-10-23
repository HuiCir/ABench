"""Packed text arrays: many documents as one UTF-8 byte buffer plus offsets, stored in .npz without pickle.

A fixed-width NumPy string array would take 4 bytes per character of the *longest* document for every document
(Yelp reviews reach 5,000 characters), so texts are stored as
    <key>_utf8     uint8 (total bytes,)   the UTF-8 encodings of all documents, concatenated
    <key>_offsets  int64 (N + 1,)         document i is utf8[offsets[i]:offsets[i + 1]]

    with np.load(path) as z:
        texts = unpack(z["text_utf8"], z["text_offsets"])          # list[str]
"""
from __future__ import annotations

import numpy as np


def pack(texts) -> tuple[np.ndarray, np.ndarray]:
    enc = [str(t).encode("utf-8") for t in texts]
    offsets = np.zeros(len(enc) + 1, np.int64)
    np.cumsum([len(b) for b in enc], out=offsets[1:])
    return np.frombuffer(b"".join(enc), np.uint8).copy(), offsets


def unpack(utf8: np.ndarray, offsets: np.ndarray, idx=None) -> list[str]:
    buf = np.asarray(utf8, np.uint8).tobytes()
    off = np.asarray(offsets, np.int64)
    rows = range(len(off) - 1) if idx is None else np.asarray(idx, np.int64)
    return [buf[off[i]:off[i + 1]].decode("utf-8") for i in rows]

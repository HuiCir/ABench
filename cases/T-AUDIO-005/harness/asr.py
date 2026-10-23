"""T-AUDIO-005 helpers (evaluator code, agent-visible): packed audio, text normalisation and word/character error rates.

Audio is stored as one int16 array of 16 kHz samples plus offsets (utterance i is audio[offsets[i]:offsets[i+1]]);
transcripts are packed UTF-8 (textpack.py). LibriSpeech transcripts are upper-case letters, apostrophes and spaces;
`normalise` maps any hypothesis to that alphabet before the error rates are computed.
"""
from __future__ import annotations

import re

import numpy as np

SR = 16000
ALPHABET = " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def utterances(audio: np.ndarray, offsets: np.ndarray, idx=None) -> list:
    rows = range(len(offsets) - 1) if idx is None else idx
    return [audio[offsets[i]:offsets[i + 1]] for i in rows]


def pack_audio(xs) -> tuple:
    off = np.zeros(len(xs) + 1, np.int64)
    np.cumsum([len(x) for x in xs], out=off[1:])
    return np.concatenate([np.asarray(x, np.int16) for x in xs]) if xs else np.zeros(0, np.int16), off


def normalise(s: str) -> str:
    s = re.sub(r"[^A-Z' ]", " ", str(s).upper())
    return re.sub(r" +", " ", s).strip()


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token sequences (O(len(a)·len(b)) with one row)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + np.asarray([x != y for y in b], np.int64)
        # insertion/deletion handled sequentially
        best = np.minimum(sub, prev[1:] + 1)
        for j in range(1, len(b) + 1):
            cur[j] = min(best[j - 1], cur[j - 1] + 1)
        prev = cur
    return int(prev[-1])


def error_counts(refs: list, hyps: list) -> tuple:
    """(word errors, reference words, character errors, reference characters) over a corpus."""
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        r, h = normalise(r), normalise(h)
        rw, hw = r.split(), h.split()
        we += edit_distance(rw, hw)
        wn += len(rw)
        ce += edit_distance(list(r), list(h))
        cn += len(r)
    return we, wn, ce, cn


def babble(clean: np.ndarray, others: list, snr_db: float, rng) -> np.ndarray:
    """Mix `clean` with the sum of three other utterances (looped or cut to its length) at the given SNR."""
    x = clean.astype(np.float64)
    noise = np.zeros_like(x)
    for o in others[:3]:
        o = o.astype(np.float64)
        reps = int(np.ceil(len(x) / max(len(o), 1)))
        start = int(rng.integers(0, max(1, len(o))))
        noise += np.tile(np.roll(o, -start), reps)[:len(x)]
    px, pn = np.mean(x ** 2) + 1e-9, np.mean(noise ** 2) + 1e-9
    y = x + noise * np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)

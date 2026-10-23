"""T-AUDIO-005: LibriSpeech (Panayotov et al. 2015, CC BY 4.0) from the parquet export openslr/librispeech_asr
(revision 71cacbfb) and the frozen encoder facebook/wav2vec2-base (revision 0b5b8e86, Apache-2.0; pre-trained only).

Public: a 10-hour, 80-speaker subset of train-clean-100 (whole speakers, in shard order) and dev-clean, as 16 kHz
int16 audio with transcripts. Hidden: test-clean, test-other, and test-clean mixed with three-speaker babble from
train-other-500 at 5-15 dB SNR (mixtures, levels and babble utterances drawn with ABENCH_T_AUDIO_SEED). All splits
are speaker-disjoint by LibriSpeech's construction.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check, text_pack

REV = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
W2V_REV = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
SECRET_SEED_ENV = "ABENCH_T_AUDIO_SEED"
TRAIN_SHARDS = [f"clean/train.100/{i:04d}.parquet" for i in range(14)]
SPLITS = {"dev_clean": "clean/validation/0000.parquet", "test_clean": "clean/test/0000.parquet",
          "test_other": "other/test/0000.parquet", "babble": "other/train.500/0000.parquet"}
TRAIN_SECONDS = 10 * 3600
W2V_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
BASE = f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/{REV}/"
SOURCES = ([{"name": f"librispeech_asr/{f}", "url": BASE + f, "license": "CC BY 4.0"} for f in TRAIN_SHARDS + list(SPLITS.values())]
           + [{"name": f"wav2vec2-base/{f}", "url": f"https://huggingface.co/facebook/wav2vec2-base/resolve/{W2V_REV}/{f}",
               "license": "Apache-2.0"} for f in W2V_FILES])
GROUPS = {"test_clean": 0, "test_other": 1, "test_clean_babble": 2}


def _harness():
    h = ROOT / "cases" / "T-AUDIO-005" / "harness"
    if str(h) not in sys.path:
        sys.path.insert(0, str(h))
    import asr
    return asr


def _read(p: Path):
    """[(audio int16, text, speaker)] from one parquet shard (operator environment: pyarrow, soundfile)."""
    import pyarrow.parquet as pq
    import soundfile as sf
    t = pq.read_table(p, columns=["audio", "text", "speaker_id"]).to_pylist()
    out = []
    for r in t:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
        assert sr == 16000
        out.append((a, r["text"], int(r["speaker_id"])))
    return out


def _arrays(rows) -> dict:
    asr = _harness()
    audio, off = asr.pack_audio([r[0] for r in rows])
    u, o = text_pack([r[1] for r in rows])
    return {"audio": audio, "audio_offsets": off, "text_utf8": u, "text_offsets": o,
            "speaker": np.asarray([r[2] for r in rows], np.int64)}


def _freeze(train, dev, tc, to, babble_src, seed) -> Frozen:
    asr = _harness()
    rng = np.random.default_rng(seed)
    noisy = []
    for a, t, s in tc:
        others = [babble_src[int(i)][0] for i in rng.choice(len(babble_src), 3, replace=False)]
        noisy.append((asr.babble(a, others, float(rng.uniform(5, 15)), rng), t, s))
    rows = tc + to + noisy
    hid = _arrays(rows)
    hid["group"] = np.asarray([0] * len(tc) + [1] * len(to) + [2] * len(noisy), np.int64)
    return Frozen(public={"train_10h.npz": _arrays(train), "dev_clean.npz": _arrays(dev)}, hidden=hid)


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (babble mixtures)")
    d = raw / "librispeech_asr"
    sums = {f: check(d / f) for f in TRAIN_SHARDS + list(SPLITS.values())}
    train, secs, spk = [], 0.0, set()
    for f in TRAIN_SHARDS:                              # whole speakers in shard order until 10 hours
        for a, t, s in _read(d / f):
            if s not in spk and secs >= TRAIN_SECONDS:
                break
            spk.add(s)
            train.append((a, t, s))
            secs += len(a) / 16000
        if secs >= TRAIN_SECONDS:
            break
    fz = _freeze(train, _read(d / SPLITS["dev_clean"]), _read(d / SPLITS["test_clean"]), _read(d / SPLITS["test_other"]),
                 _read(d / SPLITS["babble"]), seed)
    fz.sources_sha256 = sums
    w = raw / "wav2vec2-base"
    for f in W2V_FILES:
        sums[f"wav2vec2-base/{f}"] = check(w / f)
    fz.models = {"wav2vec2-base": w}
    fz.notes = {"train_hours": secs / 3600, "train_speakers": len(spk)}
    return fz


def synthetic(n: int) -> Frozen:
    """Plumbing only: tones whose pitch encodes a letter, 'transcripts' of two to four letters."""
    rng = np.random.default_rng(n)
    def utt():
        word = "".join(rng.choice(list("ABCDE"), int(rng.integers(2, 5))))
        sig = np.concatenate([np.sin(2 * np.pi * (300 + 100 * (ord(c) - 65)) * np.arange(1600) / 16000) for c in word])
        return (np.round(sig * 8000).astype(np.int16), word, int(rng.integers(0, 20)))
    mk = lambda k: [utt() for _ in range(k)]
    fz = _freeze(mk(max(20, n // 10)), mk(10), mk(6), mk(6), mk(10), 1)
    fz.sources_sha256 = {"synthetic": f"n{n}"}
    return fz

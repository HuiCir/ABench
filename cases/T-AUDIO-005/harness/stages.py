"""T-AUDIO-005 pipeline: speech recognition with a frozen wav2vec 2.0 encoder and a light trained head, scored on
clean, harder and noisy held-out speakers (evaluator and self-check).

One stage, "transcribe" (the frozen encoder is under models/wav2vec2-base):
  timed   check checkpoint/states.pt (named tensors only), import model.py, load(checkpoint_dir, models_dir, device)
  finish  transcribe(state, audios) on the held-out utterances in batches sorted by length; the time of these calls
          divided by the audio duration is the real-time factor; peak GPU memory is recorded       (≤ 1,800 s)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import asr
from abench_stage import Context, Stage
from textpack import unpack

PARAMS = {"sample_rate": asr.SR, "alphabet": asr.ALPHABET, "batch_seconds": 160, "groups": {"test_clean": 0, "test_other": 1, "test_clean_babble": 2}}
HELD_OUT_KEYS = ["audio", "audio_offsets"]
PIPELINE = [{"stage": "transcribe", "timeout": 300 + 1800, "finish_cap": 1800, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]
MAX_STATE = 20_000_000


def _timed(ctx: Context, _):
    import torch
    ck = ctx.submission / "checkpoint"
    extra = sorted(p.name for p in ck.rglob("*") if p.is_file() and p.name != "states.pt")
    if extra:
        raise SystemExit(f"checkpoint/ may contain only states.pt; found {extra[:5]}")
    states = torch.load(ck / "states.pt", map_location="cpu", weights_only=True)
    if not isinstance(states, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in states.items()):
        raise SystemExit("checkpoint/states.pt must hold a dict of named tensors")
    n = sum(v.numel() for v in states.values())
    if n > MAX_STATE:
        raise SystemExit(f"states.pt holds {n} elements, more than {MAX_STATE}")
    mod = ctx.import_submission("model.py", required=("load", "transcribe"))
    return mod, mod.load(states, str(ctx.models), ctx.device), n


def _finish(ctx: Context, st):
    import torch
    mod, state, n_params = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        audios = asr.utterances(z["audio"], z["audio_offsets"])
    order = np.argsort([len(a) for a in audios], kind="stable")
    hyps = [""] * len(audios)
    if ctx.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0, i = time.perf_counter(), 0
    while i < len(order):                               # batches of at most batch_seconds of audio
        j, tot = i, 0
        while j < len(order) and (j == i or tot + len(audios[order[j]]) <= PARAMS["batch_seconds"] * asr.SR):
            tot += len(audios[order[j]])
            j += 1
        out = mod.transcribe(state, [audios[k] for k in order[i:j]])
        for k, h in zip(order[i:j], out):
            hyps[k] = str(h)
        i = j
    ctx.sync()
    secs = time.perf_counter() - t0
    dur = sum(len(a) for a in audios) / asr.SR
    peak = torch.cuda.max_memory_allocated() / 2**20 if ctx.device.startswith("cuda") else 0.0
    u = "\n".join(h.replace("\n", " ") for h in hyps).encode("utf-8")
    return {"hyp_utf8": np.frombuffer(u, np.uint8).copy(), "rtf": np.asarray([secs / max(dur, 1e-9)]),
            "state_params": np.asarray([n_params], np.int64), "peak_mb": np.asarray([peak])}


STAGES = {"transcribe": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: dev-clean (quick: its first 200 utterances), plus the same utterances with babble from other
    dev-clean speakers as the noisy group."""
    with np.load(public / "dev_clean.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    n = len(d["audio_offsets"]) - 1
    idx = np.arange(min(n, 200) if quick else n)
    auds = asr.utterances(d["audio"], d["audio_offsets"], idx)
    texts = unpack(d["text_utf8"], d["text_offsets"], idx)
    rng = np.random.default_rng(3)
    noisy = [asr.babble(a, [auds[int(k)] for k in rng.choice(len(auds), 3, replace=False)], float(rng.uniform(5, 15)), rng)
             for a in auds]
    audio, off = asr.pack_audio(auds + noisy)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", audio=audio, audio_offsets=off)
    from textpack import pack
    u, o = pack(texts + texts)
    return {"public": public, "held_out": out / "held_out.npz",
            "reference": {"text_utf8": u, "text_offsets": o, "group": np.asarray([0] * len(auds) + [2] * len(auds))},
            "time_scale": 1.0}

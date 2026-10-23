"""T-MM-004 pipeline: image-text retrieval and compositional checks with frozen image and text encoders plus trained
adapters (evaluator and self-check).

One stage, "encode" (the frozen encoders are under models/dinov2-small and models/all-MiniLM-L12-v2):
  timed   check checkpoint/states.pt (named tensors only, at most 10 million numbers), import model.py,
          load(states, models_dir, device)
  finish  encode_images on the held-out images and encode_texts on their captions and on the compositional pairs,
          in batches of 256, timed; embeddings are returned as float32 (dimension at most 1,024)        (≤ 1,200 s)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import mm
from abench_stage import Context, Stage
from textpack import unpack

PARAMS = {"batch": 256, "max_dim": 1024, "max_state": 10_000_000}
HELD_OUT_KEYS = ["jpeg", "jpeg_offsets", "caption_utf8", "caption_offsets", "sc_jpeg", "sc_jpeg_offsets",
                 "sc_pos_utf8", "sc_pos_offsets", "sc_neg_utf8", "sc_neg_offsets"]
PIPELINE = [{"stage": "encode", "timeout": 300 + 1200, "finish_cap": 1200, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


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
    if n > PARAMS["max_state"]:
        raise SystemExit(f"states.pt holds {n} numbers, more than {PARAMS['max_state']}")
    mod = ctx.import_submission("model.py", required=("load", "encode_images", "encode_texts"))
    return mod, mod.load(states, str(ctx.models), ctx.device), n


def _encode(fn, state, items, batch):
    out, t0 = [], time.perf_counter()
    for s in range(0, len(items), batch):
        e = np.asarray(fn(state, items[s:s + batch]), np.float32)
        if e.ndim != 2 or e.shape[0] != len(items[s:s + batch]) or e.shape[1] > PARAMS["max_dim"]:
            raise SystemExit(f"embeddings must have shape (batch, d ≤ {PARAMS['max_dim']}), got {e.shape}")
        out.append(e)
    return np.concatenate(out), (time.perf_counter() - t0) * 1e3 / max(len(items), 1)


def _finish(ctx: Context, st):
    mod, state, n = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    b = PARAMS["batch"]
    img, ms_i = _encode(mod.encode_images, state, mm.jpegs(d["jpeg"], d["jpeg_offsets"]), b)
    txt, ms_t = _encode(mod.encode_texts, state, unpack(d["caption_utf8"], d["caption_offsets"]), b)
    sc_img, _ = _encode(mod.encode_images, state, mm.jpegs(d["sc_jpeg"], d["sc_jpeg_offsets"]), b)
    sc_pos, _ = _encode(mod.encode_texts, state, unpack(d["sc_pos_utf8"], d["sc_pos_offsets"]), b)
    sc_neg, _ = _encode(mod.encode_texts, state, unpack(d["sc_neg_utf8"], d["sc_neg_offsets"]), b)
    return {"img": img, "txt": txt, "sc_img": sc_img, "sc_pos": sc_pos, "sc_neg": sc_neg,
            "ms_per_image": np.asarray([ms_i]), "ms_per_text": np.asarray([ms_t]), "state_params": np.asarray([n], np.int64)}


STAGES = {"encode": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: Karpathy val (quick: its first 500 images) for retrieval; compositional pairs are made from the same
    images by swapping a caption's first two words of a pair of captions (a weak stand-in for SugarCrepe)."""
    with np.load(public / "coco_val.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    n_img = len(d["jpeg_offsets"]) - 1
    keep = np.arange(min(n_img, 500) if quick else n_img)
    caps = unpack(d["caption_utf8"], d["caption_offsets"])
    owner = d["caption_image"]
    sel = np.flatnonzero(np.isin(owner, keep))
    j, jo = mm.pack_bytes(mm.jpegs(d["jpeg"], d["jpeg_offsets"], keep))
    from textpack import pack
    cu, co = pack([caps[i] for i in sel])
    first = [sel[owner[sel] == i][0] for i in keep[:200]]
    pos = [caps[i] for i in first]
    neg = [" ".join(c.split()[1::-1] + c.split()[2:]) if len(c.split()) > 2 else c + " not" for c in pos]
    pu, po = pack(pos)
    nu, no = pack(neg)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", jpeg=j, jpeg_offsets=jo, caption_utf8=cu, caption_offsets=co,
             sc_jpeg=j, sc_jpeg_offsets=jo, sc_pos_utf8=pu, sc_pos_offsets=po, sc_neg_utf8=nu, sc_neg_offsets=no)
    ref = {"caption_image": owner[sel], "sc_image": np.arange(len(first)), "sc_subset": np.zeros(len(first), np.int64)}
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}

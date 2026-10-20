"""BI-DNA-002 pipeline: genomic sequence classification on three Genomic Benchmarks datasets (evaluator and self-check).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                  (load time, killed at 120 s)
  finish  for each dataset, call predict(model, name, seqs) -> (M, 2)              (≤ 120 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

DATASETS = ["human_nontata_promoters", "human_enhancers_cohn", "human_ocr_ensembl"]
PARAMS = {"datasets": DATASETS}
HELD_OUT_KEYS = [f"seq_{d}" for d in DATASETS]
PIPELINE = [{"stage": "predict", "timeout": 120 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for d in DATASETS:
            out[f"proba_{d}"] = np.asarray(mod.predict(model, d, list(z[f"seq_{d}"])), dtype=np.float32)
    return out


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: per dataset, the sequences in dev_idx. For an unbiased dev estimate, train your
    checkpoint without them (the reference train.py does this unless --include-dev is given)."""
    out.mkdir(parents=True, exist_ok=True)
    held, ref = {}, {}
    for d in DATASETS:
        with np.load(public / f"{d}.npz", allow_pickle=False) as z:
            seq, y, dev = z["seq"], z["y"], z["dev_idx"]
        if quick:
            dev = dev[:600]
        held[f"seq_{d}"], ref[f"y_{d}"] = seq[dev], y[dev]
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}

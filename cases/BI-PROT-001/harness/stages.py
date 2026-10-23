"""BI-PROT-001 pipeline: fitness prediction for single-substitution mutants of 30 held-out deep mutational scanning
assays from ProteinGym, with a frozen ESM-2 and 64 labelled mutants per assay (evaluator and self-check use this same
file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, models_dir, device) -> model           (load time, killed at 300 s)
  finish  for each held-out assay: predict(model, assay) -> one score per query mutant, where assay holds the
          wild-type sequence, the query mutants, 64 labelled mutants (few-shot) and the coarse selection type
                                                                                          (all 30 ≤ 1,800 s)
The labels of the query mutants never leave the evaluator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_SHOT = 64
PARAMS = {"few_shot": N_SHOT, "n_assays": 30, "max_seq_len": 600}
HELD_OUT_KEYS_SUFFIX = ("_wt", "_mut", "_fs_mut", "_fs_y", "_type")
PIPELINE = [{"stage": "predict", "timeout": 300 + 1800, "finish_cap": 1800, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]
HELD_OUT_KEYS = [f"a{j:02d}{s}" for j in range(30) for s in HELD_OUT_KEYS_SUFFIX]


def assay_ids(keys) -> list[str]:
    return sorted({k.split("_")[0] for k in keys if k.endswith("_wt")})


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for a in assay_ids(z.files):
            assay = {"wild_type": str(z[f"{a}_wt"]), "mutants": [str(m) for m in z[f"{a}_mut"]],
                     "few_shot_mutants": [str(m) for m in z[f"{a}_fs_mut"]], "few_shot_scores": z[f"{a}_fs_y"].copy(),
                     "selection_type": str(z[f"{a}_type"])}
            s = np.asarray(mod.predict(model, assay), np.float64).reshape(-1)
            if len(s) != len(assay["mutants"]):
                raise ValueError(f"{a}: predict returned {len(s)} scores for {len(assay['mutants'])} mutants")
            out[f"{a}_score"] = s
    return out


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def split_assay(mut: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """(few-shot mutants, few-shot labels, query mutants, query labels)."""
    p = rng.permutation(len(mut))
    fs, q = p[:N_SHOT], p[N_SHOT:]
    return mut[fs], y[fs], mut[q], y[q]


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: 20 of the public assays (5 with --quick) play the held-out role, with 64 labelled
    mutants each; fit anything you train on the other public assays only, or the dev estimate is optimistic."""
    files = sorted(public.glob("dev_*.npz"))
    rng = np.random.default_rng(2021)
    pick = sorted(rng.choice(len(files), size=min(5 if quick else 20, len(files)), replace=False))
    held, ref = {}, {}
    for j, i in enumerate(pick):
        with np.load(files[i], allow_pickle=False) as z:
            fm, fy, qm, qy = split_assay(z["mutants"], z["scores"], rng)
            k = f"a{j:02d}"
            held[f"{k}_wt"], held[f"{k}_type"] = z["wild_type"], z["selection_type"]
            held[f"{k}_mut"], held[f"{k}_fs_mut"], held[f"{k}_fs_y"] = qm, fm, fy
            ref[f"{k}_y"], ref[f"{k}_type"] = qy, z["selection_type"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0,
            "dev_assays": [files[i].stem for i in pick]}

"""PH-MLIP-003 pipeline: energies and forces of five molecules from rMD17, and 10 ps of molecular dynamics driven by
your potential (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  setup   load the public training configurations (for the bond list of the stability check; untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                        (load time, killed at 300 s)
  finish  for each molecule: predict(model, molecule, z, pos) on the held-out configurations (energies and forces,
          timed per configuration); then md.run: Langevin dynamics at 500 K, 0.5 fs steps, for MD_STEPS steps from 4
          held-out configurations, with the forces of predict(); the stable time of each trajectory  (all ≤ 2,400 s)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import md

MOLECULES = ("aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene")
MD_STEPS, N_MD = 20000, 4                    # 10 ps
PARAMS = {"molecules": list(MOLECULES), "md_steps": MD_STEPS, "md_dt_fs": md.DT, "md_temperature": md.TEMPERATURE,
          "md_friction": md.GAMMA, "bond_tolerance": md.BOND_TOLERANCE, "trajectories": N_MD}
HELD_OUT_KEYS = [f"{m}_{k}" for m in MOLECULES for k in ("pos_test", "md_x0")]
PIPELINE = [{"stage": "predict", "timeout": 300 + 2400, "finish_cap": 2400, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    data = {}
    for m in MOLECULES:
        with np.load(ctx.public / f"{m}.npz", allow_pickle=False) as z:
            pos = z["pos"]
            b = md.bonds(z["z"], pos[0])
            data[m] = {"z": z["z"], "bonds": b, "ref": md.bond_lengths(pos, b).mean(0)}
    return {"mol": data, "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device), st


def _finish(ctx: Context, state):
    mod, model, st = state
    steps = int(st["ov"].get("md_steps", MD_STEPS))
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for m in MOLECULES:
            info = st["mol"][m]
            pos = z[f"{m}_pos_test"].astype(np.float64)
            es, fs = [], []
            t0 = time.perf_counter()
            for i in range(0, len(pos), 100):
                e, f = mod.predict(model, m, info["z"].copy(), pos[i:i + 100].copy())
                es.append(np.asarray(e, np.float64).reshape(-1))
                fs.append(np.asarray(f, np.float64).reshape(-1, len(info["z"]), 3))
            out[f"{m}_ms_per_config"] = np.asarray([1000 * (time.perf_counter() - t0) / len(pos)])
            out[f"{m}_energy"], out[f"{m}_forces"] = np.concatenate(es), np.concatenate(fs)

            def force_fn(x, m=m, zz=info["z"]):
                return mod.predict(model, m, zz.copy(), x.copy())[1]
            out[f"{m}_stable_fs"] = md.run(force_fn, info["z"], z[f"{m}_md_x0"], steps, info["ref"], info["bonds"],
                                           ctx.seed)
    out["md_steps"] = np.asarray([float(steps)])
    return out


STAGES = {"predict": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: 200 of each molecule's 1,000 training configurations play the held-out role (4 of
    them start the dynamics); train your dev checkpoint on the other 800 (the reference train.py --dev does this).
    --quick runs 1 ps of dynamics."""
    rng = np.random.default_rng(1707)
    pub, held, ref = out / "public", {}, {}
    pub.mkdir(parents=True, exist_ok=True)
    for m in MOLECULES:
        with np.load(public / f"{m}.npz", allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        perm = rng.permutation(len(d["energy"]))
        k = min(200, len(perm) // 5)                     # 200 of the 1,000 public configurations
        te, tr = perm[:k], perm[k:]
        np.savez(pub / f"{m}.npz", z=d["z"], pos=d["pos"][tr], energy=d["energy"][tr], forces=d["forces"][tr])
        held[f"{m}_pos_test"], held[f"{m}_md_x0"] = d["pos"][te], d["pos"][te[:N_MD]]
        ref[f"{m}_energy"], ref[f"{m}_forces"] = d["energy"][te], d["forces"][te]
    if quick:
        (pub / "overrides.json").write_text(json.dumps({"md_steps": 2000}))
    np.savez(out / "held_out.npz", **held)
    return {"public": pub, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}

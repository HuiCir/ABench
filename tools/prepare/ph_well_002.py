"""PH-WELL-002: three 2D datasets of The Well (Ohana et al. 2024; Hugging Face polymathic-ai, CC-BY 4.0), resampled
to 64 x 64 by block averaging (wellsurr.SYSTEMS), float16.

  turbulent_radiative_layer_2D @2ee77565   9 values of tcool; unseen: tcool 0.18 and 1.00
  active_matter @dc3e9135                 zeta in 1..17, alpha in -1..-5; unseen: every file with zeta 9.0
  gray_scott_reaction_diffusion @913ef567 6 regimes (F, k), every 10th step; unseen: the maze regime

Public: for the other parameter sets, the train and valid splits of The Well (split 0 / 1). Hidden: "id" = the test
split of those parameter sets; "unseen" = every split of the unseen parameter sets; two windows per trajectory (4
input frames, 10 target frames) at start frames drawn with the secret seed ABENCH_PH_WELL_SEED.
Raw layout: RAW/<repo>/data/{train,valid,test}/*.hdf5. Needs h5py in the operator environment.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check

SECRET_SEED_ENV = "ABENCH_PH_WELL_SEED"
REPOS = {"turbulent_radiative": ("turbulent_radiative_layer_2D", "2ee7756575ff6f90981d0308cbcb6a2ab5995bdc", (2, 6), 1),
         "active_matter": ("active_matter", "dc3e9135a75b4e5a0d979086219fc97fe668f7a3", (4, 4), 1),
         "gray_scott": ("gray_scott_reaction_diffusion", "913ef5679d5c4b4ccd6fcd52c055509dfbf190f9", (2, 2), 10)}
SOURCES = [{"name": f"{repo}/data/", "url": f"https://huggingface.co/datasets/polymathic-ai/{repo}/tree/{rev}/data",
            "license": "CC-BY-4.0", "note": f"RAW/{repo}/data/{{train,valid,test}}/*.hdf5"}
           for repo, rev, _, _ in REPOS.values()]
SPLITS = {"train": 0, "valid": 1, "test": 2}
N_WINDOWS = 2


def wellsurr():
    p = ROOT / "cases" / "PH-WELL-002" / "harness" / "wellsurr.py"
    spec = importlib.util.spec_from_file_location("wellsurr", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["wellsurr"] = m
    spec.loader.exec_module(m)
    return m


def file_params(system: str, name: str) -> list:
    num = lambda k: float(re.search(rf"{k}_(-?[0-9.]+?)(?:_|\.hdf5)", name).group(1))   # noqa: E731
    return {"turbulent_radiative": lambda: [num("tcool")], "active_matter": lambda: [num("zeta"), num("alpha")],
            "gray_scott": lambda: [num("F"), num("k")]}[system]()


def unseen(system: str, params: list) -> bool:
    p = np.asarray(params, np.float64)
    if system == "turbulent_radiative":
        return bool(np.isclose(p[0], 0.18, atol=1e-4) or np.isclose(p[0], 1.0, atol=1e-4))
    if system == "active_matter":
        return bool(np.isclose(p[0], 9.0))
    return bool(np.allclose(p, [0.029, 0.057], atol=1e-5))


def _channels(h, system: str, b: int, tstride: int) -> np.ndarray:
    """(T, C, H, W) float32 of trajectory b, channels in wellsurr.SYSTEMS order."""
    if system == "turbulent_radiative":
        parts = [h["t0_fields/density"][b, ::tstride][:, None], h["t0_fields/pressure"][b, ::tstride][:, None],
                 np.moveaxis(h["t1_fields/velocity"][b, ::tstride], -1, 1)]
    elif system == "active_matter":
        D = h["t2_fields/D"][b, ::tstride]
        E = h["t2_fields/E"][b, ::tstride]
        parts = [h["t0_fields/concentration"][b, ::tstride][:, None], np.moveaxis(h["t1_fields/velocity"][b, ::tstride], -1, 1),
                 np.moveaxis(D.reshape(*D.shape[:3], 4), -1, 1), np.moveaxis(E.reshape(*E.shape[:3], 4), -1, 1)]
    else:
        parts = [h["t0_fields/A"][b, ::tstride][:, None], h["t0_fields/B"][b, ::tstride][:, None]]
    return np.concatenate([np.asarray(p, np.float32) for p in parts], 1)


def read_system(raw: Path, system: str) -> tuple:
    import h5py
    W = wellsurr()
    repo, _, (fy, fx), ts = REPOS[system]
    trajs, params, split, sums = [], [], [], {}
    for sp, code in SPLITS.items():
        for f in sorted((raw / repo / "data" / sp).glob("*.hdf5")):
            sums[f"{repo}/data/{sp}/{f.name}"] = check(f)
            prm = file_params(system, f.name)
            with h5py.File(f, "r") as h:
                for b in range(int(h.attrs["n_trajectories"])):
                    trajs.append(W.block_mean(_channels(h, system, b, ts), fy, fx).astype(np.float16))
                    params.append(prm)
                    split.append(code)
    return np.stack(trajs), np.asarray(params, np.float32), np.asarray(split, np.int8), sums


def _freeze(data: dict, seed: int, sums: dict) -> Frozen:
    W = wellsurr()
    sys.path.insert(0, str(ROOT / "cases" / "PH-WELL-002" / "harness"))
    sys.path.insert(0, str(ROOT / "tools" / "harness_kit"))
    spec = importlib.util.spec_from_file_location("phwell_stages", ROOT / "cases" / "PH-WELL-002" / "harness" / "stages.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    rng = np.random.default_rng(seed)
    public, hidden = {}, {}
    for system, (traj, prm, split) in data.items():
        un = np.asarray([unseen(system, list(p)) for p in prm])
        pub = ~un & (split < 2)
        public[f"{system}.npz"] = {"trajectories": traj[pub], "params": prm[pub], "split": split[pub]}
        T = traj.shape[1]
        for t, m in (("id", ~un & (split == 2)), ("unseen", un)):
            idx = np.flatnonzero(m)
            starts = [sorted(rng.choice(T - W.HISTORY - W.ROLLOUT + 1, N_WINDOWS, replace=False).tolist()) for _ in idx]
            x, p, y = S.held_out_from(traj[idx], prm[idx], starts)
            hidden[f"{system}_{t}_x"], hidden[f"{system}_{t}_params"], hidden[f"{system}_{t}_y"] = x, p, y
    return Frozen(public=public, hidden=hidden, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (window starts)")
    data, sums = {}, {}
    for s in REPOS:
        traj, prm, split, su = read_system(raw, s)
        data[s] = (traj, prm, split)
        sums.update(su)
    return _freeze(data, seed, sums)


def synthetic(n: int) -> Frozen:
    """Travelling waves whose speed is the first parameter (plumbing only)."""
    W = wellsurr()
    rng = np.random.default_rng(0)
    grids = {"turbulent_radiative": [[v] for v in (0.03, 0.18, 0.56, 1.0, 3.16)],
             "active_matter": [[z, a] for z in (1.0, 9.0, 17.0) for a in (-1.0, -3.0)],
             "gray_scott": [[0.014, 0.054], [0.029, 0.057], [0.098, 0.057], [0.058, 0.065]]}
    data = {}
    xx = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for s, grid in grids.items():
        C, T = len(W.SYSTEMS[s]["channels"]), 30
        trajs, prm, split = [], [], []
        for p in grid:
            for sp in (0, 0, 1, 2):
                ph = rng.uniform(0, 2 * np.pi)
                tr = np.stack([np.stack([np.sin(xx[None, :] + ph + c + 0.2 * t * (1 + abs(p[0]))) * np.ones((64, 1))
                                         for c in range(C)]) for t in range(T)])
                trajs.append(tr.astype(np.float16))
                prm.append(p)
                split.append(sp)
        data[s] = (np.stack(trajs), np.asarray(prm, np.float32), np.asarray(split, np.int8))
    return _freeze(data, 12345, {"synthetic": f"n{n}"})

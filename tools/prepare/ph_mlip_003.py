"""PH-MLIP-003: the revised MD17 dataset (Christensen & von Lilienfeld 2020; figshare 10.6084/m9.figshare.12672038,
version 4, CC0): DFT energies (kcal/mol) and forces (kcal/mol/Å) of 100,000 configurations per molecule. Five molecules;
the official split 01: its 1,000 training configurations are public, its 1,000 test configurations are held out (the
first 4 also start the molecular-dynamics runs). Files are checked against figshare's md5."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

BASE = "https://ndownloader.figshare.com/files"
FILES = {"rmd17_aspirin.npz": ("62265757", "17fd6fb69066888613f7e16b358a7553"),
         "rmd17_ethanol.npz": ("62265733", "eb837fb8deb27d4e0d52f71a03aff776"),
         "rmd17_malonaldehyde.npz": ("62265736", "cdc1d70c0c34062ddde6e5071eb6fe21"),
         "rmd17_naphthalene.npz": ("62265751", "0efba19c9907e3852318b1e6008b3b9e"),
         "rmd17_toluene.npz": ("62265742", "0f2913d51f8149c90ab28d697a076f64"),
         "index_train_01.csv": ("62265793", "22264dcf2e42d6133362411abbbe4ce8"),
         "index_test_01.csv": ("62265781", "124c8a0849e84e244d6ce8424e1c05cf")}
SOURCES = [{"name": n, "url": f"{BASE}/{fid}", "md5": md5, "license": "CC0 (figshare article 12672038 v4)"}
           for n, (fid, md5) in FILES.items()]
MOLECULES = ("aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene")
N_ATOMS = {"aspirin": 21, "ethanol": 9, "malonaldehyde": 9, "naphthalene": 18, "toluene": 15}
N_MD = 4


def _indices(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.int64).reshape(-1)


def _freeze(mols: dict, sums: dict) -> Frozen:
    public, hidden = {}, {}
    for m, d in mols.items():
        tr, te = d["train"], d["test"]
        public[f"{m}.npz"] = {"z": d["z"], "pos": d["pos"][tr], "energy": d["energy"][tr], "forces": d["forces"][tr]}
        hidden[f"{m}_pos_test"], hidden[f"{m}_md_x0"] = d["pos"][te], d["pos"][te[:N_MD]]
        hidden[f"{m}_energy"], hidden[f"{m}_forces"] = d["energy"][te], d["forces"][te]
    return Frozen(public=public, hidden=hidden, sources_sha256=sums)


def build(raw: Path) -> Frozen:
    d = raw / "rmd17"
    sums = {n: check(d / n, md5=md5) for n, (_, md5) in FILES.items()}
    tr, te = _indices(d / "index_train_01.csv"), _indices(d / "index_test_01.csv")
    assert len(tr) == 1000 and len(te) == 1000 and not np.intersect1d(tr, te).size
    mols = {}
    for m in MOLECULES:
        with np.load(d / f"rmd17_{m}.npz") as z:
            pos, e, f, zz = z["coords"], z["energies"], z["forces"], z["nuclear_charges"]
        assert pos.shape == (100000, N_ATOMS[m], 3) and f.shape == pos.shape and e.shape == (100000,), m
        mols[m] = {"z": zz.astype(np.int64), "pos": pos.astype(np.float64), "energy": e.astype(np.float64),
                   "forces": f.astype(np.float64), "train": tr, "test": te}
    return _freeze(mols, sums)


def synthetic(n: int) -> Frozen:
    """A 5-atom CH4-like molecule with harmonic bonds for every molecule name (for pipeline tests)."""
    rng = np.random.default_rng(0)
    z = np.array([6, 1, 1, 1, 1])
    eq = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], float)
    i, j = np.triu_indices(5, 1)
    r0 = np.linalg.norm(eq[i] - eq[j], axis=1)
    k = np.where(i == 0, 300.0, 30.0)
    M = max(n, 300)
    pos = eq[None] + rng.normal(0, 0.03, (M, 5, 3))
    dv = pos[:, i] - pos[:, j]
    r = np.linalg.norm(dv, axis=-1)
    energy = (0.5 * k * (r - r0) ** 2).sum(1)
    g = (k * (r - r0) / r)[..., None] * dv
    forces = np.zeros_like(pos)
    np.add.at(forces, (slice(None), i), -g)
    np.add.at(forces, (slice(None), j), g)
    tr, te = np.arange(0, M * 2 // 3), np.arange(M * 2 // 3, M)
    mols = {m: {"z": z, "pos": pos, "energy": energy, "forces": forces, "train": tr, "test": te} for m in MOLECULES}
    return _freeze(mols, {"synthetic": f"seed0-n{n}"})

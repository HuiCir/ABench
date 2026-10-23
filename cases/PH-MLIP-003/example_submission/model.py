"""Reference submission (inference): one random-feature force field per molecule (ff.py), loaded from checkpoint/."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ff import Model


def load(checkpoint_dir: str, device: str = "cpu"):
    models = {}
    for p in sorted(Path(checkpoint_dir).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            models[p.stem] = Model(z["W"], z["b"], z["w"], float(z["c"]), z["mu"], z["sd"])
    return models


def predict(model, molecule: str, z: np.ndarray, pos: np.ndarray):
    """pos (B, N, 3) in Å -> energies (B,) in kcal/mol and forces (B, N, 3) in kcal/mol/Å."""
    return model[molecule].predict(np.asarray(pos, np.float64))

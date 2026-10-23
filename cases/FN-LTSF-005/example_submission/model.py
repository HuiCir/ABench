"""FN-LTSF-005 reference submission: load the per-dataset, per-horizon linear forecasters and apply them."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import linear as LIN  # noqa: E402


def load(checkpoint_dir: str, device: str):
    state = {}
    for p in Path(checkpoint_dir).glob("*.npz"):
        with np.load(p) as z:
            state[p.stem] = {int(k[2:]): z[k] for k in z.files}
    return state


def predict(state, dataset: str, history: np.ndarray, horizon: int) -> np.ndarray:
    """history (B, 720, C) standardised -> forecast (B, horizon, C)."""
    return LIN.predict(state[dataset][horizon], history, horizon)

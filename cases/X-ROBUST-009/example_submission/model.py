"""Reference model for X-ROBUST-009: TF-IDF + logistic regression with temperature scaling; the confidence is the
calibrated maximum class probability and the model abstains below a threshold fitted on validation data."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import features as F  # noqa: E402


def load(checkpoint_dir, models_dir, device):
    ck = Path(checkpoint_dir)
    cal = json.loads((ck / "calibration.json").read_text(encoding="utf-8"))
    return {"vecs": F.load(ck), "W": np.load(ck / "coef.npy").astype(np.float64),
            "b": np.load(ck / "intercept.npy").astype(np.float64), **cal}


def predict(state, texts):
    z = (F.transform(state["vecs"], texts) @ state["W"].T + state["b"]) / state["temperature"]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    conf = p.max(1)
    return p.argmax(1), conf, conf < state["threshold"]

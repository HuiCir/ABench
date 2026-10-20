#!/usr/bin/env python3
"""Deterministic reference agent for DS-GBM-001 (numpy only), used to test the OJ flow end to end.

It reads only what the agent view exposes ($ABENCH_DATA_DIR/train.npz, templates/), and
writes model.py, checkpoint/, report/index.html and references.yaml to $ABENCH_SUBMISSION_DIR.
Model: standardization + L2-regularized logistic regression (batch gradient descent). Not a competitive method; only for exercising the flow.

Extra self-check: tries to list the hidden directory and read judge-side files, and writes the results to stdout so the operator can confirm they are not visible.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

view = Path(os.environ["ABENCH_VIEW_DIR"])
data = Path(os.environ["ABENCH_DATA_DIR"])
out = Path(os.environ["ABENCH_SUBMISSION_DIR"])

# ---- isolation self-check (observe only, nothing depends on the result)
seen = sorted(str(p.relative_to(view)) for p in view.rglob("*") if p.is_file())
probe = {"view_files": seen,
         "env_names": sorted(k for k in os.environ if not k.startswith(("LC_", "_"))),
         "hidden_in_view": any("hidden" in s or s.endswith(".enc") for s in seen)}
print("ISOLATION_PROBE " + json.dumps(probe, ensure_ascii=False))

# ---- training
with np.load(data / "train.npz", allow_pickle=False) as z:
    X, y = z["x"].astype(np.float64), z["y"].astype(np.int64)
mu, sd = X.mean(0), X.std(0) + 1e-8
Z = (X - mu) / sd
w, b, lam, lr = np.zeros(Z.shape[1]), 0.0, 1e-2, 0.5
for _ in range(500):
    p = 1 / (1 + np.exp(-(Z @ w + b)))
    gw = Z.T @ (p - y) / len(y) + lam * w
    gb = float((p - y).mean())
    w -= lr * gw; b -= lr * gb

# ---- write the submission
out.mkdir(parents=True, exist_ok=True)
(out / "checkpoint").mkdir(exist_ok=True)
np.savez(out / "checkpoint/params.npz", w=w, b=np.array([b]), mu=mu, sd=sd)
(out / "model.py").write_text('''import numpy as np
from pathlib import Path


def load(checkpoint_dir):
    with np.load(Path(checkpoint_dir) / "params.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def predict_proba(model, X):
    Z = (np.asarray(X, dtype=np.float64) - model["mu"]) / model["sd"]
    p1 = 1.0 / (1.0 + np.exp(-(Z @ model["w"] + model["b"][0])))
    return np.stack([1.0 - p1, p1], axis=1)
''', encoding="utf-8")

tpl = view / "templates/report_template.html"
html = tpl.read_text(encoding="utf-8")
html = html.replace("<title>⟨Method name⟩: ⟨one sentence on the problem it solves⟩</title>",
                    "<title>L2-regularized logistic regression: reference submission for pipeline checks</title>")
(out / "report").mkdir(exist_ok=True)
(out / "report/index.html").write_text(html, encoding="utf-8")
shutil.copy(view / "templates/references_template.yaml", out / "references.yaml")
print("reference agent wrote", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))

"""Judge child process: load the submitted model code and checkpoint and predict on the input features.

Started by tools/oj_eval.py in a separate temporary directory with a minimal environment. It can see only:
  the copy of the submission (model code + checkpoint) and the unlabeled input features X.
Labels always stay in the parent's memory and are never written anywhere the child can see.

Interface (tabular_classification):
  model.py defines load(checkpoint_dir: str) -> object
                predict_proba(model, X: numpy.ndarray) -> numpy.ndarray of shape (n, n_classes)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--entry", default="model.py")
    ap.add_argument("--checkpoint-dir", default="checkpoint")
    ap.add_argument("--x", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sub = Path(a.submission).resolve()
    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("submission_model", sub / a.entry)
    if spec is None or spec.loader is None:
        print(json.dumps({"error": "entry_not_importable"})); return 3
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("load", "predict_proba"):
        if not callable(getattr(mod, fn, None)):
            print(json.dumps({"error": f"missing_function:{fn}"})); return 3
    model = mod.load(str(sub / a.checkpoint_dir))
    X = np.load(a.x, allow_pickle=False)
    P = np.asarray(mod.predict_proba(model, X), dtype=np.float64)
    np.save(a.out, P, allow_pickle=False)
    print(json.dumps({"ok": True, "shape": list(P.shape)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

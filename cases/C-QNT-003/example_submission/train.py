"""C-QNT-003 reference submission, quantization program (run in the sandbox; the evaluator only loads the artifact).

  python3 train.py --data "$ABENCH_DATA_DIR" --models "$ABENCH_MODELS_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"
                   [--device cuda]

Runs quantize.quantize() on the frozen bf16 Qwen2.5-0.5B in $ABENCH_MODELS_DIR with the provided calibration tokens
(calib_tokens.npy) and writes the artifact to checkpoint/. Round-to-nearest needs no calibration data, so the tokens
are passed but unused by this reference.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from quantize import quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default=os.environ.get("ABENCH_MODELS_DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=os.environ.get("ABENCH_DEVICE", "cuda"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    calib = np.load(Path(a.data) / "calib_tokens.npy", allow_pickle=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    quantize(str(Path(a.models) / "Qwen2.5-0.5B"), calib, str(out), a.device)


if __name__ == "__main__":
    main()

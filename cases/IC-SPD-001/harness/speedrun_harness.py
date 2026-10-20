#!/usr/bin/env python3
"""IC-SPD-001 timed child process: the judge and selfcheck.py use this same file.

One timed run = one fresh Python process, executing in this order:

  1. Load the training data into memory (numpy, uint8 NHWC images + int64 labels).
  2. If available, import torch and initialize the CUDA context (not timed, identical for every submission).
  3. Send handshake byte R: timing starts.
  4. Import the submitted method.py and call train(data, seed=..., device=...).
     Module-level code, torch.compile compilation, moving data to the GPU and warmup all count as training time.
  5. If CUDA is used, torch.cuda.synchronize(), then send handshake byte T: training timing ends.
  6. Load the test images (without labels), call predict(model, x, device=...), synchronize, then send P.
     Inference time is not counted as training time but has a cap (see the task statement); exceeding it invalidates the whole run.
  7. Write the predictions to --out (float32, shape [N, 10]) and exit.

Handshake bytes are written to the pipe given by --fd; the parent records with its own clock when R, T and P arrive and when the process exits.
The parent requires exactly the three bytes R, T, P in the right order, and checks that "T to process exit" does not exceed the inference cap plus a fixed slack.
So writing T early, continuing to train inside predict, or doing any work before import will be ruled invalid or counted as training time.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_module(submission: Path, entry: str):
    sys.path.insert(0, str(submission))
    spec = importlib.util.spec_from_file_location("abench_submission", submission / entry)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {entry}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("train", "predict"):
        if not callable(getattr(mod, name, None)):
            raise SystemExit(f"{entry} must define {name}()")
    return mod


def _to_numpy(out) -> np.ndarray:
    if hasattr(out, "detach"):            # torch.Tensor
        out = out.detach().float().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--entry", default="method.py")
    ap.add_argument("--train", required=True, help="npz with x (uint8 N,32,32,3) and y (int64 N)")
    ap.add_argument("--x-test", required=True, help="npy uint8 (M,32,32,3), no labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fd", type=int, required=True)
    a = ap.parse_args()

    with np.load(a.train, allow_pickle=False) as z:
        data = {"x": np.ascontiguousarray(z["x"]), "y": np.ascontiguousarray(z["y"]).astype(np.int64)}

    torch = None
    try:
        import torch  # noqa: F401
        if a.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.init()
            torch.zeros(1, device=a.device)          # create the CUDA context, not timed
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    def sync():
        if torch is not None and a.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    os.write(a.fd, b"R")                              # ---- timing starts
    mod = _load_module(Path(a.submission), a.entry)
    model = mod.train(data, seed=a.seed, device=a.device)
    sync()
    os.write(a.fd, b"T")                              # ---- training ends

    x_test = np.load(a.x_test, allow_pickle=False)
    pred = _to_numpy(mod.predict(model, x_test, device=a.device))
    sync()
    os.write(a.fd, b"P")                              # ---- inference ends

    np.save(a.out, pred, allow_pickle=False)
    info = {"torch": getattr(torch, "__version__", None)}
    if torch is not None and a.device.startswith("cuda") and torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
        info["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    Path(a.out + ".info.json").write_text(json.dumps(info), encoding="utf-8")
    os.close(a.fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

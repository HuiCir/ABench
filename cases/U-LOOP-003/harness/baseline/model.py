"""U-LOOP-003 reference submission, inference side: a learned local lookup model (NumPy / standard library only).

Every output position i is predicted from a window of aligned input tokens ending at position i (for addition, whose
input is two operands by the public task definition, the windows of both operands), by the most frequent output
seen for that window on sampled training examples. The output length is a linear function of the input length fitted
by least squares on the samples. Nothing else about the tasks is coded by hand. Local windows extrapolate to any
length when the rule is local (copy) and fail when it needs unbounded state (long carry chains, parity), which is the
gap a looped architecture is meant to close. train.py builds the tables and pickles them to checkpoint/lookup.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

WINDOW = 4
TASK_NAMES = ("parity", "addition", "copy")
TWO_OPERANDS = {"addition"}          # from the public task definition: input = operand a then operand b


def keys(x: np.ndarray, two: bool, out_len: int):
    parts = (x[: len(x) // 2], x[len(x) // 2:]) if two else (x,)
    keys = []
    for i in range(out_len):
        k = tuple(tuple(int(p[j]) if 0 <= j < len(p) else -1 for j in range(i - WINDOW + 1, i + 1)) for p in parts)
        keys.append(k)
    return keys


def load(checkpoint_dir, device="cpu"):
    with open(Path(checkpoint_dir) / "lookup.pkl", "rb") as f:
        return pickle.load(f)


def predict(model, task, inputs):
    m = model[task]
    out = []
    for x in inputs:
        x = np.asarray(x)
        length = max(1, int(round(m["len"][0] * len(x) + m["len"][1])))
        out.append(np.array([m["table"].get(k, m["prior"]) for k in keys(x, m["two"], length)], np.int64))
    return out

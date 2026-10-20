"""U-LOOP-003 task definitions and the training sampler (shared by the evaluator, the self-check and your code).

Three algorithmic tasks over token sequences; n is the problem length.
  parity    input n bits                               output 1 token: XOR of the bits
  addition  input 2n bits: a then b, least-significant bit first     output n + 1 bits of a + b, LSB first
  copy      input n tokens from {0, ..., 7} (repeats allowed)         output the same n tokens
The sampler only produces lengths 1..MAX_TRAIN_LEN; the held-out lengths are 21..40 (and an in-distribution check at 1..20).
"""
from __future__ import annotations

import numpy as np

TASKS = ("parity", "addition", "copy")
MAX_TRAIN_LEN = 20
MAX_TEST_LEN = 40
VOCAB = {"parity": 2, "addition": 2, "copy": 8}


def make(task: str, n: int, count: int, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs, ys = [], []
    for _ in range(count):
        if task == "parity":
            x = rng.integers(0, 2, n)
            y = np.array([x.sum() % 2])
        elif task == "addition":
            a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
            av = sum(int(v) << i for i, v in enumerate(a))
            bv = sum(int(v) << i for i, v in enumerate(b))
            s = av + bv
            x = np.concatenate([a, b])
            y = np.array([(s >> i) & 1 for i in range(n + 1)])
        elif task == "copy":
            x = rng.integers(0, 8, n)
            y = x.copy()
        else:
            raise ValueError(task)
        xs.append(x.astype(np.int64))
        ys.append(y.astype(np.int64))
    return xs, ys


def output_length(task: str, n: int) -> int:
    return {"parity": 1, "addition": n + 1, "copy": n}[task]


class Sampler:
    """The only source of training data: sample(task, n, count) for 1 <= n <= MAX_TRAIN_LEN."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def sample(self, task: str, n: int, count: int):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        if not 1 <= int(n) <= MAX_TRAIN_LEN:
            raise ValueError(f"training lengths are limited to 1..{MAX_TRAIN_LEN}")
        return make(task, int(n), int(count), self._rng)

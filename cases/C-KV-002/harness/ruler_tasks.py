"""C-KV-002 long-context tasks in the style of RULER (Hsieh et al. 2024), generated as text (evaluator and agent use
this same file; the held-out prompts are drawn with a secret seed and tokenised by the operator).

  niah_single      one "special magic number" needle in noise text; recall it
  niah_multikey    four needles with different keys; recall the one asked for
  niah_multivalue  four numbers for the same key; recall all four
  vt               variable tracking: X1 = n, then X2 = X1, X3 = X2, X4 = X3 among noise; name every variable equal to n
A prompt's score is the fraction of the expected strings that appear in the generated answer (RULER's string match).
"""
from __future__ import annotations

import re

import numpy as np

TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
WORDS = ("apple", "river", "candle", "violin", "harbor", "falcon", "meadow", "lantern", "copper", "orchid", "glacier",
         "pepper", "saddle", "tundra", "walnut", "beacon", "cobalt", "dagger", "ember", "fossil")


def _num(rng) -> str:
    return str(int(rng.integers(1_000_000, 10_000_000)))


def _place(chunks: list[str], n_noise: int, rng) -> str:
    """Insert the chunks at random sentence boundaries of n_noise repetitions of the noise text."""
    slots = sorted(int(x) for x in rng.choice(n_noise, size=len(chunks), replace=False))
    parts, prev = [], 0
    for s, c in zip(slots, chunks):
        parts.append(NOISE * (s - prev))
        parts.append(c + " ")
        prev = s
    parts.append(NOISE * (n_noise - prev))
    return "".join(parts)


def make(task: str, n_noise: int, rng: np.random.Generator) -> tuple[str, str, list[str]]:
    """(context with question, answer prefix, expected strings)."""
    if task.startswith("niah"):
        keys = [str(w) for w in rng.choice(WORDS, size=4, replace=False)]
        if task == "niah_single":
            k, v = keys[0], _num(rng)
            needles, ask, exp = [f"One of the special magic numbers for {k} is: {v}."], k, [v]
        elif task == "niah_multikey":
            vals = [_num(rng) for _ in keys]
            needles = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)]
            j = int(rng.integers(4))
            ask, exp = keys[j], [vals[j]]
        else:
            k, vals = keys[0], [_num(rng) for _ in range(4)]
            needles, ask, exp = [f"One of the special magic numbers for {k} is: {v}." for v in vals], k, vals
        text = _place(needles, n_noise, rng)
        q = (f"What are all the special magic numbers for {ask} mentioned in the provided text?" if task == "niah_multivalue"
             else f"What is the special magic number for {ask} mentioned in the provided text?")
        prompt = (f"Some special magic numbers are hidden within the following text. Make sure to memorize them. "
                  f"I will quiz you about the numbers afterwards.\n{text}\n{q}")
        return prompt, f"The special magic number{'s' if task == 'niah_multivalue' else ''} for {ask} mentioned in the provided text {'are' if task == 'niah_multivalue' else 'is'}", exp
    if task == "vt":
        names = ["".join(chr(65 + int(c)) for c in rng.integers(0, 26, 5)) for _ in range(4)]
        v = str(int(rng.integers(10000, 100000)))
        chain = [f"VAR {names[0]} = {v}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, 4)]
        text = _place(chain, n_noise, rng)
        prompt = (f"Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{text}\n"
                  f"Question: Find all variables that are assigned the value {v} in the text above.")
        return prompt, f"Answer: According to the chain(s) of variable assignment in the text above, 4 variables are assigned the value {v}, they are:", names
    raise KeyError(task)


def score(answer: str, expected: list[str]) -> float:
    return float(np.mean([bool(re.search(re.escape(e), answer)) for e in expected]))

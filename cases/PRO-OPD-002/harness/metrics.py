"""PRO-OPD-002 metrics: accuracy averaged over 16 sampled answers (avg@16) of the distilled student on held-out
AIME 2025 and AIME 2026 problems without tools (primary: over all 60 problems); length; audit gates."""
from __future__ import annotations

import numpy as np

import math_tool as MT
import pro_stages as P
from abench_metrics import OutputError, need  # noqa: F401

N_SAMPLES = 16
PRIMARY = {"metric": "avg16_accuracy", "goal": "max"}
GATES = P.GATES + [{"metric": "same_arch", "min": 1.0}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": ["answer", "set"], "outputs": [["eval", "per_problem"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = P.scores(outputs)
    ev = outputs.get("eval", {})
    out["same_arch"] = float(np.asarray(ev.get("same_arch", [0]))[0])
    gold = np.asarray(reference["answer"]).astype(np.int64)
    sets = np.asarray(reference["set"]).astype(str)
    if "per_problem" in ev:                                     # bootstrap replicates carry per-problem accuracy
        acc = np.asarray(ev["per_problem"], np.float64)
    elif out["same_arch"]:
        resp = need(outputs, "eval", "response", shape=(len(gold) * N_SAMPLES,), finite=False)
        ok = np.asarray([MT.correct_int(str(r), int(g)) for r, g in zip(resp, np.repeat(gold, N_SAMPLES))], np.float64)
        acc = ok.reshape(len(gold), N_SAMPLES).mean(1)
        outputs["eval"]["per_problem"] = acc
        out["response_tokens_mean"] = float(np.mean(ev["response_tokens"]))
        out["no_answer_rate"] = float(np.mean([MT.extract_boxed(MT.final_text(str(r))) is None for r in resp]))
    else:
        acc = np.zeros(len(gold))
    out["avg16_accuracy"] = float(acc.mean())
    for s in sorted(set(sets.tolist())):
        out[f"avg16_{s}"] = float(acc[sets == s].mean())
    return out

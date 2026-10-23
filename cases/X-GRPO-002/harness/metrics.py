"""X-GRPO-002 metrics: verified accuracy on held-out Countdown and GSM8K problems (primary: their mean), format
violations and signs of reward hacking, response length and trained parameters."""
from __future__ import annotations

import json

import numpy as np

import verify_math as V
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "mean_accuracy", "goal": "max"}
GATES = [{"metric": "format_violation_rate", "max": 0.5}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": ["task", "spec_json"], "outputs": [["eval", "response"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    task = np.asarray(reference["task"]).astype(np.int64)
    specs = [json.loads(s) for s in np.asarray(reference["spec_json"])]
    resp = need(outputs, "eval", "response", shape=(len(task),), finite=False)
    ok, fmt, multi = np.zeros(len(task), bool), np.zeros(len(task), bool), np.zeros(len(task), bool)
    for i, (t, s, r) in enumerate(zip(task, specs, resp)):
        r = str(r)
        a = V.extract(r)
        fmt[i] = a is not None
        multi[i] = len(V.ANSWER.findall(r)) > 1
        ok[i] = (V.check_countdown(a, s["nums"], s["target"]) if t == 0 else V.check_gsm8k(a, s["gold"])) if fmt[i] else False
    out = {}
    accs = []
    for t, name in ((0, "countdown"), (1, "gsm8k")):
        m = task == t
        if m.any():
            out[f"accuracy_{name}"] = float(ok[m].mean())
            accs.append(out[f"accuracy_{name}"])
    out["mean_accuracy"] = float(np.mean(accs))
    out["format_violation_rate"] = float(1 - fmt.mean())
    out["multiple_answer_rate"] = float(multi.mean())
    lens = np.asarray([len(str(r)) for r in resp])
    out["mean_response_chars"] = float(lens.mean())
    out["truncated_rate"] = float(np.mean([("<answer>" in str(r)) and ("</answer>" not in str(r)) for r in resp]))
    if "eval" in outputs and "checkpoint_numbers" in outputs["eval"]:
        out["checkpoint_numbers"] = float(np.asarray(outputs["eval"]["checkpoint_numbers"])[0])
    out["load_seconds"] = float(seconds.get("eval", float("nan")))
    return out

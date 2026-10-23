"""X-SFT-005 metrics: prompt-level instruction following (primary: fraction of prompts whose every constraint is
satisfied), constraint-level accuracy, forgetting on held-out web text and the trained-parameter cost."""
from __future__ import annotations

import json

import numpy as np

import ifcheck
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "prompt_strict", "goal": "max"}
GATES: list = []
RESAMPLE = [{"reference": ["constraints"], "outputs": [["eval", "response"]]}]


def _rows(reference, outputs):
    cons = [json.loads(str(x)) for x in reference["constraints"]]
    resp = [str(x) for x in np.asarray(outputs["eval"]["response"])] if "response" in outputs.get("eval", {}) else None
    if resp is None:
        raise OutputError("stage 'eval' did not return 'response'")
    if len(resp) != len(cons):
        raise OutputError(f"expected {len(cons)} responses, got {len(resp)}")
    return cons, resp


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    cons, resp = _rows(reference, outputs)
    ok = [ifcheck.check(r, c["kinds"], c["params"]) for r, c in zip(resp, cons)]
    flat = [x for o in ok for x in o]
    out = {"prompt_strict": float(np.mean([all(o) for o in ok])), "instruction_accuracy": float(np.mean(flat))}
    by: dict = {}
    for o, c in zip(ok, cons):
        for k, v in zip(c["kinds"], o):
            by.setdefault(k, []).append(v)
    for k, v in sorted(by.items()):
        out[f"accuracy_{k}"] = float(np.mean(v))
    nll = float(need(outputs, "eval", "nll", shape=(1,))[0])
    base = float(need(outputs, "eval", "nll_base", shape=(1,))[0])
    out.update({"heldout_nll": nll, "forgetting": nll - base, "trained_numbers": float(need(outputs, "eval", "checkpoint_numbers", shape=(1,))[0]),
                "mean_response_words": float(np.mean([len(r.split()) for r in resp]))})
    return out

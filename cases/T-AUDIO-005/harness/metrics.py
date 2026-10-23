"""T-AUDIO-005 metrics: word error rate on each held-out group (primary: their mean), character error rate, real-time
factor, trained parameters and peak memory."""
from __future__ import annotations

import numpy as np

import asr
from abench_metrics import OutputError, need  # noqa: F401
from textpack import unpack

PRIMARY = {"metric": "wer_mean", "goal": "min"}
GATES = [{"metric": "trained_parameters", "max": 20_000_000}]
NAMES = {0: "test_clean", 1: "test_other", 2: "test_clean_babble"}


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    refs = unpack(reference["text_utf8"], reference["text_offsets"])
    g = np.asarray(reference["group"]).astype(int)
    raw = need(outputs, "transcribe", "hyp_utf8", shape=(None,)).astype(np.uint8).tobytes().decode("utf-8")
    hyps = raw.split("\n") if refs else []
    if len(hyps) != len(refs):
        raise OutputError(f"expected {len(refs)} transcripts, got {len(hyps)}")
    out, wers = {}, []
    for k in sorted(set(g.tolist())):
        m = np.flatnonzero(g == k)
        we, wn, ce, cn = asr.error_counts([refs[i] for i in m], [hyps[i] for i in m])
        out[f"wer_{NAMES.get(k, k)}"] = we / max(wn, 1)
        out[f"cer_{NAMES.get(k, k)}"] = ce / max(cn, 1)
        wers.append(we / max(wn, 1))
    out["wer_mean"] = float(np.mean(wers))
    out["real_time_factor"] = float(need(outputs, "transcribe", "rtf", shape=(1,))[0])
    out["trained_parameters"] = float(need(outputs, "transcribe", "state_params", shape=(1,))[0])
    out["peak_memory_mb"] = float(need(outputs, "transcribe", "peak_mb", shape=(1,))[0])
    return out

"""T-MM-004 metrics: image-text retrieval on Karpathy test (primary: mean of R@1, R@5, R@10 in both directions),
compositional accuracy on SugarCrepe by subset, trained parameters and encoding latency."""
from __future__ import annotations

import numpy as np

import mm
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "recall_mean", "goal": "max"}
GATES = [{"metric": "trained_parameters", "max": 10_000_000}]
SUBSETS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    owner = np.asarray(reference["caption_image"]).astype(np.int64)
    img = need(outputs, "encode", "img", shape=(None, None))
    txt = need(outputs, "encode", "txt", shape=(len(owner), img.shape[1]))
    out = mm.retrieval(img, txt, owner)
    sc_i = np.asarray(reference["sc_image"]).astype(np.int64)
    sub = np.asarray(reference["sc_subset"]).astype(np.int64)
    si = need(outputs, "encode", "sc_img", shape=(None, img.shape[1]))
    sp = need(outputs, "encode", "sc_pos", shape=(len(sc_i), img.shape[1]))
    sn = need(outputs, "encode", "sc_neg", shape=(len(sc_i), img.shape[1]))
    ok = mm.compositional(si[sc_i], sp, sn)
    per = [float(ok[sub == k].mean()) for k in range(len(SUBSETS)) if (sub == k).any()]
    for k, name in enumerate(SUBSETS):
        if (sub == k).any():
            out[f"sugarcrepe_{name}"] = float(ok[sub == k].mean())
    out["sugarcrepe_mean"] = float(np.mean(per))
    out["trained_parameters"] = float(need(outputs, "encode", "state_params", shape=(1,))[0])
    out["ms_per_image"] = float(need(outputs, "encode", "ms_per_image", shape=(1,))[0])
    out["ms_per_text"] = float(need(outputs, "encode", "ms_per_text", shape=(1,))[0])
    return out

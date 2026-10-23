"""FN-QA-004: FinQA (Chen et al. 2021, github.com/czyssrs/FinQA, MIT licence; reports from FinTabNet) and the frozen
Qwen2.5-1.5B-Instruct (Apache-2.0).

Public: train.json (6,251 questions) and dev.json (883), verbatim, with gold programs and answers.
Hidden: the 1,147 test questions as the pipeline sees them (question, pre_text, post_text, table; no gold program,
answer or evidence annotation), and, for the evaluator only, the gold executed answers, the number of program steps
and whether the annotated evidence includes a table row. The authors' private test set has no public answers and is
not used. Nothing is random, so there is no secret seed: the test questions are public on GitHub, which is why the
case rules forbid any data from outside the sandbox (the sandbox has no network).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
SHA = {"train.json": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
       "dev.json": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
       "test.json": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc"}
MODEL, MODEL_REV = "Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt"]
SOURCES = [{"name": f"czyssrs/FinQA dataset/{f}", "url": f"https://github.com/czyssrs/FinQA/blob/{COMMIT}/dataset/{f}",
            "sha256": s, "license": "MIT", "note": f"RAW/FinQA/dataset/{f}"} for f, s in SHA.items()] + \
          [{"name": f"{MODEL}/", "url": f"https://huggingface.co/Qwen/{MODEL}/tree/{MODEL_REV}", "files": MODEL_FILES,
            "license": "Apache-2.0", "note": f"RAW/{MODEL}/"}]
SIZES = {"train.json": 6251, "dev.json": 883, "test.json": 1147}


def _finqa():
    import importlib.util
    from prepare.common import ROOT
    p = ROOT / "cases" / "FN-QA-004" / "harness" / "finqa.py"
    spec = importlib.util.spec_from_file_location("finqa_case", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def public_item(x: dict) -> dict:
    return {"id": x["id"], "question": x["qa"]["question"], "pre_text": x["pre_text"], "post_text": x["post_text"],
            "table": x["table"]}


def _hidden(test: list) -> dict:
    F = _finqa()
    for x in test:                                      # the executor must reproduce every gold answer
        assert F.correct(F.execute(x["qa"]["program"], x["table"]), x["qa"]["exe_ans"]), x["id"]
    return {"items_json": np.asarray([json.dumps(public_item(x)) for x in test], dtype=np.str_),
            "gold": np.asarray([str(x["qa"]["exe_ans"]) for x in test], dtype=np.str_),
            "steps": np.asarray([len(F.split_steps(x["qa"]["program"])) for x in test], np.int64),
            "has_table_evidence": np.asarray([any(k.startswith("table") for k in x["qa"].get("gold_inds", {}))
                                              for x in test], np.int64)}


def build(raw: Path) -> Frozen:
    d = raw / "FinQA" / "dataset"
    sums = {f: check(d / f, sha256=s) for f, s in SHA.items()}
    data = {f: json.loads((d / f).read_text(encoding="utf-8")) for f in SHA}
    for f, n in SIZES.items():
        assert len(data[f]) == n, (f, len(data[f]))
    return Frozen(public={}, hidden=_hidden(data["test.json"]), sources_sha256=sums,
                  models={MODEL: raw / MODEL}, raw_files={"train.json": d / "train.json", "dev.json": d / "dev.json"})


def synthetic(n: int) -> Frozen:
    """Plumbing only: random two-row tables and one- or two-step programs (n test, n train, n // 2 dev items)."""
    import tempfile
    rng = np.random.default_rng(0)

    def item(i, split):
        a, b, c = (float(rng.integers(10, 1000)) for _ in range(3))
        table = [["", "2019", "2018"], ["revenue", f"${a:,.0f}", f"${b:,.0f}"], ["costs", f"{c:.0f}", f"{c / 2:.0f}"]]
        if i % 2:
            q, prog = "what was the change in revenue from 2018 to 2019?", f"subtract({a:.0f}, {b:.0f})"
        else:
            q, prog = "what was the percentage change in revenue?", f"subtract({a:.0f}, {b:.0f}), divide(#0, {b:.0f})"
        F = _finqa()
        ans = F.execute(prog, table)
        return {"id": f"{split}-{i}", "pre_text": ["revenue and costs are shown below ."], "post_text": [],
                "table": table, "qa": {"question": q, "program": prog, "exe_ans": ans, "answer": str(ans),
                                       "gold_inds": {"table_1": "revenue"}}}
    tmp = Path(tempfile.mkdtemp(prefix="finqa-"))
    for split, m in (("train", n), ("dev", n // 2)):
        (tmp / f"{split}.json").write_text(json.dumps([item(i, split) for i in range(m)]))
    return Frozen(public={}, hidden=_hidden([item(i, "test") for i in range(n)]), sources_sha256={"synthetic": f"n{n}"},
                  raw_files={"train.json": tmp / "train.json", "dev.json": tmp / "dev.json"})

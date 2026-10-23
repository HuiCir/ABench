"""LG-LONG-001: LexGLUE ECtHR task A (Chalkidis et al. 2022; Hugging Face coastalcph/lex_glue, config ecthr_a, CC BY 4.0):
the facts of European Court of Human Rights cases and the articles of the Convention found violated (10 labels; a case
may have none). Public: the train (9,000) and validation (1,000) splits; held out: the official test split (1,000).
The facts are a list of paragraphs per case; they are stored as one string per case with paragraphs separated by a
blank line, packed (UTF-8 bytes + offsets, cases/LG-LONG-001/harness/textpack.py). The frozen encoder
nlpaueb/legal-bert-base-uncased (CC BY-SA 4.0) is provided for entries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, text_pack

REV = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
LEGAL_BERT_REV = "15b570cbf88259610b082a167dacc190124f60f6"
BASE = f"https://huggingface.co/datasets/coastalcph/lex_glue/resolve/{REV}/ecthr_a"
LABELS = ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"]
SIZES = {"train": 9000, "validation": 1000, "test": 1000}
SOURCES = [{"name": f"ecthr_a/{s}-00000-of-00001.parquet", "url": f"{BASE}/{s}-00000-of-00001.parquet", "license": "CC BY 4.0"}
           for s in SIZES] + [
           {"name": "legal-bert-base-uncased/", "url": f"https://huggingface.co/nlpaueb/legal-bert-base-uncased/tree/{LEGAL_BERT_REV}",
            "files": ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"],
            "license": "CC BY-SA 4.0"}]


def _split(df) -> dict:
    texts = ["\n\n".join(str(p) for p in paras) for paras in df["text"]]
    y = np.zeros((len(df), len(LABELS)), np.int8)
    for i, labs in enumerate(df["labels"]):
        for j in labs:
            y[i, int(j)] = 1
    utf8, off = text_pack(texts)
    return {"text_utf8": utf8, "text_offsets": off, "y": y}


def build(raw: Path) -> Frozen:
    import pandas as pd                       # operator environment only
    sums, parts = {}, {}
    for s, n in SIZES.items():
        p = raw / "ecthr_a" / f"{s}-00000-of-00001.parquet"
        sums[f"ecthr_a/{p.name}"] = check(p)
        df = pd.read_parquet(p)
        assert len(df) == n, (s, len(df))
        parts[s] = _split(df)
    lb = raw / "legal-bert-base-uncased"
    for f in SOURCES[-1]["files"]:
        sums[f"legal-bert-base-uncased/{f}"] = check(lb / f)
    tr, va = parts["train"], parts["validation"]
    public = {"ecthr_a.npz": {"train_text_utf8": tr["text_utf8"], "train_text_offsets": tr["text_offsets"], "train_y": tr["y"],
                              "val_text_utf8": va["text_utf8"], "val_text_offsets": va["text_offsets"], "val_y": va["y"],
                              "labels": np.asarray(LABELS)}}
    return Frozen(public=public, hidden=parts["test"], sources_sha256=sums, models={"legal-bert-base-uncased": lb})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(300)]

    def make(m):
        y = (rng.random((m, len(LABELS))) < 0.15).astype(np.int8)
        docs = []
        for row in y:
            paras = []
            for _ in range(int(rng.integers(3, 8))):
                toks = list(rng.choice(words, 30)) + [f"art{j}" for j in np.flatnonzero(row) for _ in range(2)]
                rng.shuffle(toks)
                paras.append(" ".join(toks))
            docs.append("\n\n".join(paras))
        u, o = text_pack(docs)
        return {"text_utf8": u, "text_offsets": o, "y": y}
    tr, va, te = make(n), make(max(50, n // 5)), make(max(50, n // 5))
    public = {"ecthr_a.npz": {"train_text_utf8": tr["text_utf8"], "train_text_offsets": tr["text_offsets"], "train_y": tr["y"],
                              "val_text_utf8": va["text_utf8"], "val_text_offsets": va["text_offsets"], "val_y": va["y"],
                              "labels": np.asarray(LABELS)}}
    return Frozen(public=public, hidden=te, sources_sha256={"synthetic": f"n{n}"})

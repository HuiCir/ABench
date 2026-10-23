"""X-ROBUST-009: CLINC150 intent classification with out-of-scope queries (Larson et al. 2019, CC BY 3.0) and
Banking77 as a distribution shift (Casanueva et al. 2020, CC BY 4.0), both from their authors' GitHub repositories.

Public: CLINC150 'full' train and validation, in-scope (150 intents, 100 + 20 per intent) and out-of-scope (100 + 100),
packed texts with labels (-1 = out of scope), and the intent names. Hidden: the CLINC150 test set (4,500 in-scope,
1,000 out-of-scope), the Banking77 test set (3,080 queries, all outside CLINC's scope), and the in-scope test queries
with character noise (typos drawn with a secret seed, ABENCH_X_ROBUST_SEED).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check, text_pack

CLINC_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_SHA = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
B77_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
B77_SHA = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
SECRET_SEED_ENV = "ABENCH_X_ROBUST_SEED"
SOURCES = [{"name": "clinc/oos-eval data/data_full.json", "url": f"https://github.com/clinc/oos-eval/blob/{CLINC_COMMIT}/data/data_full.json",
            "sha256": CLINC_SHA, "license": "CC BY 3.0", "note": "RAW/clinc/data_full.json"},
           {"name": "PolyAI-LDN/task-specific-datasets banking_data/test.csv",
            "url": f"https://github.com/PolyAI-LDN/task-specific-datasets/blob/{B77_COMMIT}/banking_data/test.csv",
            "sha256": B77_SHA, "license": "CC BY 4.0", "note": "RAW/banking77/test.csv"},
           {"name": "roberta-base/", "url": "https://huggingface.co/FacebookAI/roberta-base/tree/e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
            "license": "MIT", "note": "RAW/roberta-base/"}]
GROUPS = {"in_scope": 0, "out_of_scope": 1, "banking77": 2, "typo": 3}


def typos(text: str, rng: np.random.Generator, rate: float = 0.06) -> str:
    """Character noise: each letter is, with probability `rate`, deleted, duplicated, swapped with the next letter or
    replaced by a neighbouring key."""
    keys = "qwertyuiopasdfghjklzxcvbnm"
    near = {c: keys[max(0, i - 1)] + keys[min(len(keys) - 1, i + 1)] for i, c in enumerate(keys)}
    out, i = [], 0
    s = list(text)
    while i < len(s):
        c = s[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.integers(0, 4)
            if op == 0:
                i += 1
                continue
            if op == 1:
                out += [c, c]
            elif op == 2 and i + 1 < len(s):
                out += [s[i + 1], c]
                i += 2
                continue
            else:
                out.append(near.get(c.lower(), c)[rng.integers(0, 2)])
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _clinc(d: dict):
    intents = sorted({lab for _, lab in d["train"]})
    ix = {n: i for i, n in enumerate(intents)}
    def split(ins, oos):
        texts = [t for t, _ in d[ins]] + [t for t, _ in d[oos]]
        labels = np.asarray([ix[l] for _, l in d[ins]] + [-1] * len(d[oos]), np.int64)
        return texts, labels
    return intents, split


def _freeze(d: dict, b77: list, seed: int) -> Frozen:
    intents, split = _clinc(d)
    tr_t, tr_y = split("train", "oos_train")
    va_t, va_y = split("val", "oos_val")
    te_t, te_y = split("test", "oos_test")
    rng = np.random.default_rng(seed)
    ins = [t for t, y in zip(te_t, te_y) if y >= 0]
    ins_y = [y for y in te_y if y >= 0]
    texts = te_t + b77 + [typos(t, rng) for t in ins]
    labels = np.concatenate([te_y, -np.ones(len(b77), np.int64), np.asarray(ins_y, np.int64)])
    group = np.concatenate([np.where(te_y >= 0, 0, 1), np.full(len(b77), 2), np.full(len(ins), 3)]).astype(np.int64)
    u, o = text_pack(texts)
    pub = {}
    for name, (t, y) in {"train": (tr_t, tr_y), "val": (va_t, va_y)}.items():
        pub[f"{name}_text_utf8"], pub[f"{name}_text_offsets"] = text_pack(t)
        pub[f"{name}_label"] = y
    return Frozen(public={"clinc_public.npz": pub}, hidden={"text_utf8": u, "text_offsets": o, "label": labels, "group": group},
                  notes={"intents": intents})


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (typo noise)")
    sums = {"data_full.json": check(raw / "clinc" / "data_full.json", sha256=CLINC_SHA),
            "banking77_test.csv": check(raw / "banking77" / "test.csv", sha256=B77_SHA)}
    import csv
    d = json.loads((raw / "clinc" / "data_full.json").read_text(encoding="utf-8"))
    with open(raw / "banking77" / "test.csv", newline="", encoding="utf-8") as f:
        b77 = [r["text"] for r in csv.DictReader(f)]
    fz = _freeze(d, b77, seed)
    fz.sources_sha256 = sums
    fz.models = {"roberta-base": raw / "roberta-base"}          # FacebookAI/roberta-base @e2da8e2f (MIT), as for X-LORA-004
    _intents_file(fz)
    return fz


def _intents_file(fz: Frozen) -> None:
    import tempfile
    p = Path(tempfile.mkdtemp(prefix="clinc-")) / "intents.json"
    p.write_text(json.dumps(fz.notes["intents"], indent=0) + "\n", encoding="utf-8")
    fz.raw_files["intents.json"] = p


def synthetic(n: int) -> Frozen:
    """Learnable fake data with the same schema: 150 intents whose queries share a keyword."""
    rng = np.random.default_rng(n)
    words = [f"w{i}" for i in range(400)]
    def q(k):
        return " ".join([f"intent{k}"] + list(rng.choice(words, 5)))
    def ood():
        return " ".join(rng.choice(words, 6))
    d = {"train": [(q(k), f"in{k:03d}") for k in range(150) for _ in range(max(2, n // 150))],
         "val": [(q(k), f"in{k:03d}") for k in range(150) for _ in range(2)],
         "test": [(q(k), f"in{k:03d}") for k in range(150) for _ in range(2)],
         "oos_train": [(ood(), "oos") for _ in range(20)], "oos_val": [(ood(), "oos") for _ in range(20)],
         "oos_test": [(ood(), "oos") for _ in range(40)]}
    fz = _freeze(d, [ood() for _ in range(40)], 1)
    fz.sources_sha256 = {"synthetic": f"n{n}"}
    _intents_file(fz)
    return fz

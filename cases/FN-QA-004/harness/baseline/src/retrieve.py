"""Evidence retrieval for FinQA-style items: table rows and text sentences as facts, ranked by TF-IDF cosine
similarity with the question (inverse document frequencies estimated on the training questions and facts)."""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("the of in is a an and to for was were what by on as at from with that this be or are its it".split())


def tokens(s: str) -> list[str]:
    return [t for t in TOKEN.findall(s.lower()) if t not in STOP]


def facts(item: dict) -> list[tuple[str, str]]:
    """(key, sentence): 'table_<i>' for table row i (i >= 1), 'text_<j>' for sentence j of pre_text + post_text."""
    out = []
    table = item.get("table") or []
    head = table[0] if table else []
    for i, row in enumerate(table[1:], start=1):
        if not row:
            continue
        cells = [f"the {head[k].strip() or 'value'} of {row[0].strip()} is {row[k].strip()}"
                 for k in range(1, len(row)) if k < len(head) and row[k].strip()]
        out.append((f"table_{i}", " ; ".join(cells) + " ;" if cells else row[0]))
    for j, s in enumerate(list(item.get("pre_text") or []) + list(item.get("post_text") or [])):
        if s.strip() and s.strip() != ".":
            out.append((f"text_{j}", s.strip()))
    return out


def fit_idf(texts: list[str]) -> dict:
    df = Counter()
    for t in texts:
        df.update(set(tokens(t)))
    n = len(texts)
    return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()} | {"__n__": float(n)}


def _vec(s: str, idf: dict) -> dict:
    tf = Counter(tokens(s))
    default = math.log(idf.get("__n__", 1.0) + 1) + 1.0
    v = {w: c * idf.get(w, default) for w, c in tf.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / norm for w, x in v.items()}


def cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def top_facts(item: dict, idf: dict, k: int = 8) -> list[tuple[str, str]]:
    q = _vec(item["question"], idf)
    fs = facts(item)
    scored = sorted(fs, key=lambda f: -cosine(q, _vec(f[1], idf)))
    return scored[:k]


def nearest(question: str, bank: list[dict], idf: dict, k: int = 4) -> list[dict]:
    q = _vec(question, idf)
    return sorted(bank, key=lambda e: -cosine(q, _vec(e["question"], idf)))[:k]

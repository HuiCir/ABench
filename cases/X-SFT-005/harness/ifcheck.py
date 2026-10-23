"""X-SFT-005 verifiable instructions (evaluator code, agent-visible): constraint types in the style of IFEval
(Zhou et al. 2023), their wording, their checkers and the generator that adds one to three compatible constraints to
a request. The held-out prompts use these types with other requests, other parameters and a secret seed.
"""
from __future__ import annotations

import json
import re

import numpy as np

KEYWORDS = ["history", "energy", "balance", "network", "journey", "pattern", "community", "quality", "future", "method",
            "season", "garden", "signal", "memory", "budget", "texture", "harbor", "orbit", "canvas", "ledger"]
END_PHRASES = ["Is there anything else I can help with?", "Any other questions?", "Let me know if you need more."]


def _words(t):
    return re.findall(r"[A-Za-z0-9']+", t)


def _strip_fence(t):
    t = t.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


TYPES = {
    "lowercase": (lambda p: "Your entire response should be in English and in all lowercase letters. No capital letters are allowed.",
                  lambda t, p: bool(re.search(r"[a-z]", t)) and t == t.lower()),
    "uppercase": (lambda p: "Your entire response should be in English and in all capital letters.",
                  lambda t, p: bool(re.search(r"[A-Z]", t)) and t == t.upper()),
    "no_commas": (lambda p: "Do not use any commas in your response.", lambda t, p: "," not in t),
    "max_words": (lambda p: f"Answer with at most {p} words.", lambda t, p: 0 < len(_words(t)) <= p),
    "min_words": (lambda p: f"Answer with at least {p} words.", lambda t, p: len(_words(t)) >= p),
    "bullets": (lambda p: f"Your answer must contain exactly {p} bullet points. Use markdown bullet points such as:\n* This is point 1.",
                lambda t, p: len([l for l in t.splitlines() if re.match(r"^\s*[*-]\s+\S", l)]) == p),
    "title": (lambda p: "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
              lambda t, p: bool(re.search(r"<<[^\n<>]+>>", t))),
    "end_phrase": (lambda p: f"Finish your response with this exact phrase: {p} No other words should follow this phrase.",
                   lambda t, p: t.strip().endswith(p)),
    "keywords": (lambda p: f"Include the keywords \"{p[0]}\" and \"{p[1]}\" in the response.",
                 lambda t, p: all(re.search(rf"\b{re.escape(k)}\b", t, re.I) for k in p)),
    "json": (lambda p: "Your entire output should be wrapped in JSON format. You can use markdown ticks such as ```.",
             lambda t, p: _is_json(_strip_fence(t))),
    "paragraphs": (lambda p: f"There should be exactly {p} paragraphs. Paragraphs are separated by the markdown divider: ***",
                   lambda t, p: len([x for x in re.split(r"\n\s*\*\*\*\s*\n", t.strip()) if x.strip()]) == p),
    "quotation": (lambda p: "Wrap your entire response with double quotation marks.",
                  lambda t, p: len(t.strip()) > 1 and t.strip()[0] == '"' and t.strip()[-1] == '"'),
    "placeholders": (lambda p: f"The response must contain at least {p} placeholders represented by square brackets, such as [address].",
                     lambda t, p: len(re.findall(r"\[[^\[\]\n]+\]", t)) >= p),
    "postscript": (lambda p: "At the end of your response, please explicitly add a postscript starting with P.S.",
                   lambda t, p: bool(re.search(r"(^|\n)\s*P\.S\.", t))),
}
PARAM = {"max_words": [40, 60, 80, 120], "min_words": [150, 200, 250], "bullets": [2, 3, 4, 5],
         "paragraphs": [2, 3, 4], "placeholders": [2, 3], "end_phrase": END_PHRASES}
CONFLICTS = [{"lowercase", "uppercase"}, {"max_words", "min_words"}, {"json", "bullets"}, {"json", "paragraphs"},
             {"json", "quotation"}, {"json", "end_phrase"}, {"json", "postscript"}, {"json", "title"},
             {"quotation", "end_phrase"}, {"quotation", "postscript"}, {"end_phrase", "postscript"},
             {"uppercase", "postscript"}, {"lowercase", "postscript"}, {"bullets", "paragraphs"},
             {"max_words", "paragraphs"}, {"lowercase", "end_phrase"}, {"uppercase", "end_phrase"},
             {"uppercase", "keywords"}, {"lowercase", "title"}]


def _is_json(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def _param(kind, rng):
    if kind == "keywords":
        a, b = rng.choice(len(KEYWORDS), 2, replace=False)
        return [KEYWORDS[a], KEYWORDS[b]]
    vals = PARAM.get(kind)
    return None if vals is None else vals[int(rng.integers(0, len(vals)))]


def make(request: str, rng: np.random.Generator, n_max: int = 3) -> dict:
    """One prompt: the request followed by one to n_max compatible constraints."""
    k = int(rng.integers(1, n_max + 1))
    kinds: list = []
    for c in rng.permutation(sorted(TYPES)):
        if len(kinds) == k:
            break
        if any({c, x} in [set(s) for s in CONFLICTS] for x in kinds):
            continue
        kinds.append(str(c))
    params = [_param(c, rng) for c in kinds]
    text = request.strip() + "\n\n" + " ".join(TYPES[c][0](p) for c, p in zip(kinds, params))
    return {"prompt": text, "kinds": kinds, "params": params}


def check(response: str, kinds: list, params: list) -> list:
    """One boolean per constraint."""
    return [bool(TYPES[c][1](response, p)) for c, p in zip(kinds, params)]

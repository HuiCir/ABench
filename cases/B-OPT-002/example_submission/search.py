"""Reference search (NumPy, CPU, under a minute): enumerate mixtures of two update directions from a small family,
over a grid of momentum constants, and keep the program with the best descent cosine on the public gradient stream among
those whose similarity to every known optimizer stays at most 0.97. Writes program.yaml and search_log.jsonl.

  python3 search.py --interface "$ABENCH_VIEW_DIR/interface" [--out .]

A real search would rank the survivors by short training runs on the micro-LM; this reference stops at the screen.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

DIRECTIONS = {"sign": "sign(c)", "orth": "orth(c)", "norm": "normalize(c)", "adam": "clip(div(c, add(sqrt(v), 1.0e-8)), 1)",
              "rownorm": "rownorm(c)"}


def program(d1: str, d2: str, k: float, b1: float, b2: float) -> dict:
    lines = ["c = interp(m, g, b1)", "m = interp(m, g, b2)", "v = interp(v, square(g), b2)",
             f"update = add(mul(k, {DIRECTIONS[d1]}), mul(sub(1, k), {DIRECTIONS[d2]}))"]
    return {"state": ["m", "v"], "constants": {"b1": b1, "b2": b2, "k": k}, "lr": 1.0e-3, "program": lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    sys.path.insert(0, a.interface)
    import yaml
    from primitive_dsl import ProgramRule
    from behaviour import screen
    log, best = [], None
    for d1, d2 in itertools.combinations(DIRECTIONS, 2):
        for k, b1, b2 in itertools.product((0.25, 0.5, 0.75), (0.9, 0.95), (0.98, 0.995)):
            prog = program(d1, d2, k, b1, b2)
            r = screen(ProgramRule(prog), 17)
            ok = r["stable_extreme_scales"] == 1.0 and r["max_similarity"] <= 0.97
            log.append({"d1": d1, "d2": d2, "k": k, "b1": b1, "b2": b2, "max_similarity": round(r["max_similarity"], 4),
                        "closest": r["closest"], "descent_cosine": round(r["descent_cosine"], 4), "kept": ok})
            if ok and (best is None or r["descent_cosine"] > best[0]):
                best = (r["descent_cosine"], prog, log[-1])
    out = Path(a.out)
    (out / "search_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    (out / "program.yaml").write_text("# Written by search.py: " + json.dumps(best[2]) + "\n" + yaml.safe_dump(best[1], sort_keys=False))
    print(len(log), "candidates;", sum(x["kept"] for x in log), "kept; best", best[2])


if __name__ == "__main__":
    main()

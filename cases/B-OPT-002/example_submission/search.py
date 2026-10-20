"""A small, replayable candidate enumeration over the frozen grammar that writes search_log.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
from primitive_search import fixed_stream, proxy_score, validate_program


def z_expr():
    return {"add": ["m_fast", {"mul": ["alpha", {"sub": ["m_fast", "m_slow"]}]}]}

def filtered_expr():
    z = z_expr()
    d = {"sub": ["m_fast", "m_slow"]}
    conf = {"safe_div": [{"abs": z}, {"add": [{"abs": z}, "tau"]}]}
    gate = {"add": ["half", {"mul": ["half", {"sign": d}]}]}
    return {"mul": [{"mul": [{"sign": z}, conf]}, gate]}


def programs():
    z = z_expr()
    return [
        {"name": "lion_sign", "beta_fast": .90, "beta_slow": .99, "alpha": 0.0, "tau": 0.0,
         "update": {"sign": z}},
        {"name": "lion_smooth", "beta_fast": .90, "beta_slow": .99, "alpha": 0.0, "tau": .05,
         "update": {"mul": [{"sign": z}, {"safe_div": [{"abs": z}, {"add": [{"abs": z}, "tau"]}]}]}},
        {"name": "ridge_sign", "beta_fast": .90, "beta_slow": .995, "alpha": .70, "tau": .05,
         "update": {"mul": [{"sign": z}, {"safe_div": [{"abs": z}, {"add": [{"abs": z}, "tau"]}]}]}},
        {"name": "ridge_sign_slow", "beta_fast": .95, "beta_slow": .999, "alpha": .70, "tau": .05,
         "update": filtered_expr()},
        {"name": "ema_direction", "beta_fast": .90, "beta_slow": .995, "alpha": .70, "tau": .05,
         "update": z},
        {"name": "normalized_ridge", "beta_fast": .90, "beta_slow": .995, "alpha": .70, "tau": .05,
         "update": {"normalize_rms": z}},
    ]


def main(seed=17):
    stream = fixed_stream(seed)
    out = []
    for i, program in enumerate(programs(), 1):
        started = time.perf_counter()
        p = validate_program(program)
        p.pop("program_id", None)
        score = proxy_score(p, stream)
        out.append({"eval": i, "program_id": validate_program(p)["program_id"], "program": p,
                    "proxy_score": score, "seed": seed, "stream": "frozen_v1",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3)})
    out.sort(key=lambda r: r["eval"])  # eval order is part of the audit trail
    dest = HERE / "search_log.jsonl"
    dest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out), encoding="utf-8")
    best = max(out, key=lambda r: r["proxy_score"])
    print(json.dumps({"evaluations": len(out), "best_eval": best["eval"], "best_program": best["program_id"],
                      "best_score": best["proxy_score"], "seed": seed}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args().seed)

"""Verify the frozen-primitive search trace of B-OPT-002."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from primitive_search import GRAMMAR_VERSION, fixed_stream, proxy_score, validate_program, replay


def validate_search(sub_dir: Path, candidate_limit: int = 256, seed: int = 17) -> dict:
    spec_path = sub_dir / "search_spec.yaml"
    log_path = sub_dir / "search_log.jsonl"
    result = {"present": log_path.exists() and spec_path.exists(), "replay": False, "valid_programs": 0,
              "evaluations": 0, "best_score": None, "best_eval": None, "grammar": None,
              "selected_program": None, "selected_program_id": None, "selected_present": False,
              "target_reached": False, "errors": []}
    if not result["present"]:
        result["errors"].append("search_spec.yaml or search_log.jsonl missing")
        result["pass"] = False
        return result
    import yaml
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    optimizer_text = (sub_dir / "optimizer.py").read_text(encoding="utf-8") if (sub_dir / "optimizer.py").exists() else ""
    result["grammar"] = spec.get("grammar_version")
    result["selected_program"] = spec.get("selected_program")
    result["selected_program_id"] = spec.get("selected_program_id")
    if result["grammar"] != GRAMMAR_VERSION or spec.get("seed") != seed:
        result["errors"].append("grammar_version/seed mismatch")
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as e: result["errors"].append(f"invalid JSON: {e}")
    result["evaluations"] = len(rows)
    if not rows or len(rows) > candidate_limit:
        result["errors"].append("empty or over-budget search log")
    ids = []
    stream = fixed_stream(seed)
    for i, row in enumerate(rows, 1):
        try:
            p = validate_program(row["program"])
            ids.append(p["program_id"])
            if row.get("program_id") != p["program_id"]:
                result["errors"].append(f"program_id mismatch at eval {i}")
            if row.get("seed") != seed:
                result["errors"].append(f"seed mismatch at eval {i}")
            if float(row.get("elapsed_ms", -1)) < 0:
                result["errors"].append(f"missing elapsed_ms at eval {i}")
            score = proxy_score(p, stream)
            if abs(float(row.get("proxy_score", score)) - score) > 1e-8:
                result["errors"].append(f"score mismatch at eval {i}")
            if row.get("eval") != i:
                result["errors"].append(f"non-contiguous eval at {i}")
            result["valid_programs"] += 1
            if result["selected_program"] and row["program"].get("name") == result["selected_program"]:
                result["selected_present"] = True
            if result["best_score"] is None or score > result["best_score"]:
                result["best_score"], result["best_eval"] = score, i
        except Exception as e:
            result["errors"].append(f"invalid program at eval {i}: {e}")
    result["unique_programs"] = len(set(ids))
    result["replay"] = result["valid_programs"] == len(rows) and len(set(ids)) == len(ids)
    if result["selected_program"] and result["selected_program"] not in optimizer_text:
        result["errors"].append("selected_program is not bound in optimizer.py")
    selected_rows = [r for r in rows if r.get("program", {}).get("name") == result["selected_program"]]
    if result["selected_program_id"]:
        if len(selected_rows) != 1:
            result["errors"].append("selected_program_id requires exactly one selected program row")
        elif selected_rows[0].get("program_id") != result["selected_program_id"]:
            result["errors"].append("selected_program_id does not match canonical selected program")
        elif result["selected_program_id"] not in optimizer_text:
            result["errors"].append("selected_program_id is not bound in optimizer.py")
    if result["selected_program"] and not result["selected_present"]:
        result["errors"].append("selected_program not present in search log")
    lion = [r for r in rows if r.get("program", {}).get("name") == "lion_sign"]
    if lion and result["best_score"] is not None:
        result["target_reached"] = result["best_score"] >= 1.02 * float(lion[0].get("proxy_score", -1e9))
    result["pass"] = bool(result["present"] and result["replay"] and result["errors"] == [] and
                           result["valid_programs"] > 0 and result["selected_present"])
    return result

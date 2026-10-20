"""Normalize ABench run schemas and readiness metadata in-place.

This migration only edits declarative contracts. It never creates runs,
checkpoints, scores, or simulation outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
REAL_PROVENANCE = ["benchmark_real", "reference_real", "judge_real", "real"]
HASH_RE = "^(?:sha256:)?[0-9a-fA-F]{64}$"


def entries():
    idx = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
    for track in idx.get("tracks", []):
        for item in track.get("cases", []) or []:
            yield item


def readiness(item: dict, card: dict) -> str:
    if card.get("status") == "deferred" or "deferred" in card:
        return "deferred"
    if item.get("status") == "built":
        return "reference_ready"
    return "contract_only"


def normalize_card(item: dict) -> bool:
    cid = item["id"]
    path = ROOT / "cases" / cid / "case" / "anchor_card.yaml"
    if not path.is_file():
        return False
    card = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(card, dict) or "readiness" in card:
        return False
    value = readiness(item, card)
    # Insert a compact field while preserving the existing hand-authored YAML.
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if re.match(r"^case_id:\s*", line):
            indent = re.match(r"^(\s*)", line).group(1)
            lines.insert(i + 1, f"{indent}readiness: {value}\n")
            path.write_text("".join(lines), encoding="utf-8")
            return True
    # Very old cards without case_id are rewritten safely and will be caught by lint.
    card["readiness"] = value
    path.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return True


def standard_schema(cid: str, old: dict) -> dict:
    old_metrics = old.get("metrics", []) if isinstance(old, dict) else []
    old_title = old.get("title") if isinstance(old, dict) else None
    schema = {
        "$schema": SCHEMA_URI,
        "$id": f"abench://run-schema/{cid}/v1",
        "title": old_title or f"ABench {cid} real run record",
        "description": "A benchmark-generated real artifact. Simulated and self-reported results are rejected.",
        "type": "object",
        "required": [
            "schema_version", "case_id", "provenance", "runner_version",
            "config_hash", "resource_hash", "artifact_hash", "metrics", "run_mode",
        ],
        "properties": {
            "schema_version": {"const": "1.0"},
            "case_id": {"const": cid},
            "provenance": {"enum": REAL_PROVENANCE},
            "runner_version": {"type": "string", "minLength": 1},
            "config_hash": {"type": "string", "pattern": HASH_RE},
            "resource_hash": {"type": "string", "pattern": HASH_RE},
            "artifact_hash": {"type": "string", "pattern": HASH_RE},
            "metrics": {
                "type": "object",
                "description": "Case-specific measured metrics; names and directions are defined in scoring_config/metric_registry.",
                "additionalProperties": True,
            },
            "run_mode": {"const": "benchmark", "description": "Only benchmark-mode runs are scoreable; smoke runs are rejected."},
            "pareto": {
                "type": "object",
                "description": "Server-measured Pareto axes; unavailable values are null and excluded from dominance.",
                "required": ["quality", "throughput", "peak_memory", "compute_cost", "stability"],
                "properties": {
                    "quality": {"type": ["number", "null"]},
                    "throughput": {"type": ["number", "null"]},
                    "peak_memory": {"type": ["number", "null"]},
                    "compute_cost": {"type": ["number", "null"]},
                    "stability": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                },
                "additionalProperties": False,
            },
            "hardware_signature": {
                "type": "object",
                "required": ["sandbox_profile", "hardware_class", "cost_unit"],
                "properties": {
                    "sandbox_profile": {"type": "string"},
                    "hardware_class": {"type": "string"},
                    "runtime_hash": {"type": "string"},
                    "accelerator_count": {"type": "integer", "minimum": 0},
                    "memory_gib": {"type": ["number", "null"], "minimum": 0},
                    "cost_unit": {"enum": ["GPU-h", "CPU-h", "H100-h"]},
                },
                "additionalProperties": True,
            },
            "setting": {"type": "string"},
            "seed": {"type": "integer"},
            "wall_time_ms": {"type": "number", "minimum": 0},
            "peak_memory_bytes": {"type": "integer", "minimum": 0},
            "checkpoint_sha256": {"type": "string", "pattern": HASH_RE},
            "records": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional per-setting records for aggregate submissions.",
            },
        },
        "additionalProperties": True,
    }
    if old_metrics:
        schema["x-abench-metric-names"] = old_metrics
    return schema


def normalize_schema(item: dict) -> bool:
    cid = item["id"]
    path = ROOT / "cases" / cid / "case" / "run_schema.json"
    if not path.is_file():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        old = {}
    schema = standard_schema(cid, old)
    new = json.dumps(schema, ensure_ascii=False, indent=2) + "\n"
    if path.read_text(encoding="utf-8") == new:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def normalize_contract(item: dict) -> bool:
    cid = item["id"]
    path = ROOT / "cases" / cid / "case" / "bench_contract.yaml"
    if not path.is_file():
        return False
    contract = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(contract, dict):
        return False
    req = list(contract.get("required_provenance") or [])
    changed = False
    if not {"runner_version", "config_hash", "resource_hash", "artifact_hash"}.issubset(req):
        for field in ("runner_version", "config_hash", "resource_hash", "artifact_hash"):
            if field not in req:
                req.append(field)
        contract["required_provenance"] = req
        changed = True
    if not contract.get("report_profile"):
        rules = yaml.safe_load((ROOT / "spec" / "report_profiles.yaml").read_text(encoding="utf-8"))
        assignment = (rules or {}).get("assignment_rules", {}) or {}
        explicit = assignment.get("exceptions", {}) or {}
        if cid in explicit:
            profile = explicit[cid]
        else:
            suffixes = (assignment.get("judgment_report_v2", {}) or {}).get("suffixes", []) or []
            profile = "judgment_report_v2" if any(s in cid for s in suffixes) else "design_report_v2"
        contract["report_profile"] = profile
        changed = True
    if changed:
        path.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return changed


def _metric_names(item: dict, cid: str) -> list[str]:
    """Extract metric names from structured protocol fields before free-text fallbacks."""
    base = ROOT / "cases" / cid / "case"
    names: list[str] = []
    # The index case card is the authoritative display list.  Some newer
    # Pareto cards use ``scoring_config.metrics`` for axis metadata (objective,
    # dominance, aggregation); those keys are not benchmark metrics themselves.
    raw = item.get("metrics")
    if isinstance(raw, str):
        names.extend(x.strip() for x in re.split(r";", raw) if x.strip())
    if names:
        return list(dict.fromkeys(names))
    cfg = yaml.safe_load((base / "scoring_config.yaml").read_text(encoding="utf-8")) if (base / "scoring_config.yaml").is_file() else {}
    proto = yaml.safe_load((base / "eval_protocol.yaml").read_text(encoding="utf-8")) if (base / "eval_protocol.yaml").is_file() else {}
    def add(value: object) -> None:
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
        elif isinstance(value, dict):
            add(value.get("id") or value.get("name"))

    def add_raw(raw: object) -> None:
        if isinstance(raw, dict):
            dimensions = raw.get("dimensions")
            if isinstance(dimensions, list):
                for value in dimensions:
                    add(value)
                return
            for value in raw.values():
                if isinstance(value, (str, list, tuple, dict)):
                    add_raw(value)
        elif isinstance(raw, (list, tuple)):
            for value in raw:
                add(value)
        elif isinstance(raw, str):
            for value in re.split(r";", raw):
                add(value)

    for source in (cfg or {}, proto or {}):
        add_raw(source.get("primary_metrics"))
        add_raw(source.get("primary_metric"))
        add_raw(source.get("metrics"))
        pareto = source.get("pareto")
        if isinstance(pareto, dict):
            add_raw(pareto.get("axes"))
    if not names:
        names = ["benchmark_score"]
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _direction(name: str) -> str:
    if re.search(r"KL|loss|error|latency|memory|time|cost|degradation|entropy|bias|parameter|GPU-h|ECE|risk|over-refusal|gap|forgetting", name, re.I):
        return "lower_is_better"
    if re.search(r"AUC|F1|Recall|tokens/s|accuracy|throughput|speedup|quality|recall|retrieval|acceptance|stability|success|coverage|score|NDCG|Spearman|IC|Sharpe", name, re.I):
        return "higher_is_better"
    return "benchmark_defined"


def normalize_metric_registry(item: dict) -> bool:
    cid = item["id"]
    path = ROOT / "cases" / cid / "case" / "metric_registry.yaml"
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        generated = existing.get("generated_by") == "abench-normalize-contracts"
        auto_ids = all(isinstance(m, dict) and str(m.get("id", "")).startswith("metric_") for m in existing.get("metrics", []) or [])
        if not (generated or (existing.get("status") == "calibration_required" and auto_ids)):
            return False
    metrics = []
    source_names = _metric_names(item, cid)
    names = [name for name in source_names if not re.search(r"pareto", name, re.I)] or source_names
    for name in names:
        i = len(metrics) + 1
        metrics.append({
            "id": f"metric_{i:02d}",
            "name": name,
            "direction": _direction(name),
            "unit": "benchmark_defined",
            "aggregation": "benchmark_defined",
            "required": True,
            "pareto_role": "aggregate" if re.search(r"pareto", name, re.I) else "objective",
            "inference": "heuristic",
            "notes": "Direction, unit, aggregation and Pareto thresholds must be frozen from real calibration.",
        })
    doc = {
        "schema_version": "1.0",
        "case_id": cid,
        "status": "calibration_required",
        "generated_by": "abench-normalize-contracts",
        "metrics": metrics,
    }
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    cards = schemas = contracts = registries = 0
    for item in entries():
        cards += int(normalize_card(item))
        schemas += int(normalize_schema(item))
        contracts += int(normalize_contract(item))
        registries += int(normalize_metric_registry(item))
    print(json.dumps({"cards": cards, "run_schemas": schemas, "contracts": contracts, "metric_registries": registries}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Contract-only runner: validates real artifacts; never creates scores."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REAL_PROVENANCE = {"benchmark_real", "reference_real", "judge_real", "real"}
SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
REQUIRED = ("schema_version", "case_id", "provenance", "runner_version", "config_hash", "resource_hash", "artifact_hash", "metrics", "run_mode")


def load_schema(case_dir: Path) -> dict:
    try:
        schema = json.loads((case_dir / "run_schema.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"run_schema.json is invalid: {exc}")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise SystemExit("run_schema.json must be a standard object schema")
    return schema


def load_real(path: str, label: str, case_id: str, schema: dict):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"{label} not found: {p}")
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"{label} is not valid JSON: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    missing = [name for name in REQUIRED if name not in value]
    if missing:
        raise SystemExit(f"{label} is missing required fields: {', '.join(missing)}")
    if value.get("case_id") != case_id:
        raise SystemExit(f"{label}.case_id={value.get('case_id')!r} does not match {case_id}")
    provenance = str(value.get("provenance", "")).lower()
    if provenance not in REAL_PROVENANCE:
        raise SystemExit(f"{label} is not marked as benchmark-generated real; provide real artifacts")
    for field in ("config_hash", "resource_hash", "artifact_hash"):
        if not isinstance(value.get(field), str) or not SHA256.fullmatch(value[field]):
            raise SystemExit(f"{label}.{field} must be a SHA-256 hex digest")
    if not isinstance(value.get("metrics"), dict):
        raise SystemExit(f"{label}.metrics must be a JSON object")
    if value.get("run_mode") != "benchmark":
        raise SystemExit(f"{label}.run_mode={value.get('run_mode')!r}: smoke runs are not scored")
    # Keep this check coupled to the on-disk schema; the validator performs the
    # complete static shape check before activation and this avoids a silent
    # runner/schema drift for the fields used at runtime.
    required_schema = set(schema.get("required", []))
    if not set(REQUIRED).issubset(required_schema):
        raise SystemExit("run_schema.json does not declare the required provenance contract")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    submission = Path(args.submission)
    if not submission.exists():
        raise SystemExit(f"submission not found: {submission}")
    case_dir = ROOT / "case"
    card = yaml.safe_load((case_dir / "anchor_card.yaml").read_text(encoding="utf-8")) or {}
    case_id = card.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise SystemExit("anchor_card.case_id is missing")
    schema = load_schema(case_dir)
    protocol = yaml.safe_load((case_dir / "eval_protocol.yaml").read_text(encoding="utf-8")) or {}
    runs = load_real(args.runs, "runs", case_id, schema)
    judge = load_real(args.judge, "judge", case_id, schema)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "case_id": case_id,
        "submission": str(submission),
        "status": "contract_validated",
        "scorable": False,
        "schema_validated": True,
        "note": "Formal scoring is deferred until real resources, reference runs, calibration and hidden evaluation are frozen.",
        "runs_keys": sorted(runs),
        "judge_keys": sorted(judge),
        "metrics": protocol.get("metrics", {}),
    }
    (out / "contract_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

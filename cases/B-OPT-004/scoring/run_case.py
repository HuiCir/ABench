#!/usr/bin/env python3
"""Validate a real ABench submission run; never creates simulated scores."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_real(path: str, label: str):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{label} not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"{label} is not valid JSON: {e}")
    if not isinstance(obj, dict):
        raise SystemExit(f"{label} must be a JSON object")
    prov = str(obj.get("provenance", obj.get("_provenance"))).lower()
    if prov in {"mock", "sim", "simulated", "demo", "synthetic"} or "simulat" in prov:
        raise SystemExit(f"{label} is marked as simulated; real benchmark artifacts are required")
    if prov not in {"benchmark_real", "reference_real", "judge_real", "real"}:
        raise SystemExit(f"{label} is missing accepted real provenance")
    if obj.get("run_mode", "benchmark") != "benchmark":
        raise SystemExit(f"run_mode={obj.get('run_mode')!r}: smoke runs are not scored")
    for field in ("config_hash", "resource_hash", "artifact_hash"):
        value = obj.get(field)
        if not isinstance(value, str) or len(value.replace("sha256:", "")) != 64:
            raise SystemExit(f"{label}.{field} must be a SHA-256 digest")
    if not isinstance(obj.get("metrics"), dict):
        raise SystemExit(f"{label}.metrics must be a JSON object")
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--runs", required=True, help="benchmark-generated real run JSON")
    ap.add_argument("--judge", required=True, help="benchmark-generated blind judge JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    submission = Path(args.submission)
    if not submission.exists():
        raise SystemExit(f"submission not found: {submission}")
    card = yaml.safe_load((ROOT/"case"/"anchor_card.yaml").read_text())
    contract = yaml.safe_load((ROOT/"case"/"bench_contract.yaml").read_text())
    runs = load_real(args.runs, "runs")
    judge = load_real(args.judge, "judge")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {"case_id": card["case_id"], "submission": str(args.submission),
              "status": "contract_validated", "scorable": False,
              "note": "Scoring remains deferred until benchmark-side real resources, calibration, and hidden evaluation are frozen.",
              "runs_keys": sorted(runs.keys()), "judge_keys": sorted(judge.keys()),
              "required_metrics": contract["evaluation"]["metrics"]}
    (out/"contract_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

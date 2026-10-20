"""Complete the shared contract entry points of all cases, without running any task or generating simulated results."""
from __future__ import annotations

from pathlib import Path
import json
import stat
import yaml

ROOT = Path(__file__).resolve().parents[1]


def assigned_report_profile(case_id: str) -> str:
    """Resolve the same v2 profile precedence used by runtime validators."""
    cfg = yaml.safe_load((ROOT / "spec" / "report_profiles.yaml").read_text(encoding="utf-8")) or {}
    rules = cfg.get("assignment_rules", {}) or {}
    explicit = rules.get("exceptions", {}) or {}
    if case_id in explicit:
        return explicit[case_id]
    suffixes = (rules.get("judgment_report_v2", {}) or {}).get("suffixes", []) or []
    return "judgment_report_v2" if any(s in case_id for s in suffixes) else "design_report_v2"

RUNNER = '''
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
'''

ANTI_GAMING = """# Anti-gaming contract scaffold

- Run results must be produced by the benchmark operator's runner and include configuration, code, resource and artifact hashes.
- A submission must not read hidden labels, the validation set, evaluator state or external network resources.
- Time, memory, throughput, OOM, timeouts and randomness are measured on the server; self-reported fields from the submission are not trusted on their own.
- Results that are missing, irreproducible or of unclear provenance are marked failed/unknown and are never interpolated into a score.
- Before real launch, case-specific negative examples, leak checks and hidden perturbations must be added.
"""

def main() -> None:
    index = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
    cases = [c for track in index.get("tracks", []) for c in track.get("cases", [])]
    added = {"bench_contract": 0, "run_schema": 0, "anti_gaming": 0, "runner": 0}
    for item in cases:
        cid = item["id"]
        base = ROOT / "cases" / cid
        if not base.is_dir():
            continue
        card_path = base / "case" / "anchor_card.yaml"
        card = yaml.safe_load(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        blockers = card.get("blocked_inputs", []) if isinstance(card, dict) else []
        contract = base / "case" / "bench_contract.yaml"
        if not contract.exists():
            contract.write_text(yaml.safe_dump({
                "contract_version": "abench-scaffold-v1",
                "case_id": cid,
                "status": "scaffold",
                "simulated_outputs": False,
                "real_inputs_required": True,
                "blocked_inputs": blockers,
                "required_provenance": ["runner_version", "config_hash", "resource_hash", "artifact_hash"],
                "formal_scoring": "deferred_until_real_calibration",
                "report_profile": assigned_report_profile(cid),
            }, sort_keys=False, allow_unicode=True), encoding="utf-8")
            added["bench_contract"] += 1
        schema = base / "case" / "run_schema.json"
        if not schema.exists():
            schema.write_text(json.dumps({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": f"ABench {cid} real run record",
                "type": "object",
                "required": ["schema_version", "case_id", "provenance", "runner_version", "config_hash", "resource_hash", "artifact_hash", "metrics", "run_mode"],
                "properties": {
                    "schema_version": {"const": "1.0"},
                    "case_id": {"const": cid},
                    "provenance": {"enum": ["benchmark_real", "reference_real", "judge_real", "real"]},
                    "runner_version": {"type": "string"},
                    "config_hash": {"type": "string", "pattern": "^(?:sha256:)?[0-9a-fA-F]{64}$"},
                    "resource_hash": {"type": "string", "pattern": "^(?:sha256:)?[0-9a-fA-F]{64}$"},
                    "artifact_hash": {"type": "string", "pattern": "^(?:sha256:)?[0-9a-fA-F]{64}$"},
                    "metrics": {"type": "object"},
                    "run_mode": {"const": "benchmark", "description": "Only benchmark-mode runs are scoreable; smoke runs are rejected."},
                    "pareto": {
                        "type": "object",
                        "required": ["quality", "throughput", "peak_memory", "compute_cost", "stability"],
                        "properties": {k: {"type": ["number", "null"]} for k in ["quality", "throughput", "peak_memory", "compute_cost", "stability"]},
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
                },
                "additionalProperties": True,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            added["run_schema"] += 1
        registry = base / "case" / "metric_registry.yaml"
        if not registry.exists():
            registry.write_text(yaml.safe_dump({
                "schema_version": "1.0",
                "case_id": cid,
                "status": "calibration_required",
                "metrics": [{
                    "id": "metric_01",
                    "name": "benchmark_metric",
                    "direction": "benchmark_defined",
                    "unit": "benchmark_defined",
                    "aggregation": "benchmark_defined",
                    "required": True,
                    "pareto_role": "objective",
                    "inference": "placeholder",
                    "notes": "Replace with case-specific metrics before activation.",
                }],
            }, sort_keys=False, allow_unicode=True), encoding="utf-8")
            added["metric_registry"] = added.get("metric_registry", 0) + 1
        anti = base / "case" / "anti_gaming.md"
        if not anti.exists():
            anti.write_text(ANTI_GAMING, encoding="utf-8")
            added["anti_gaming"] += 1
        runner = base / "scoring" / "run_case.py"
        if not runner.exists():
            runner.write_text(RUNNER, encoding="utf-8")
            runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
            added["runner"] += 1
    print(json.dumps(added, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()


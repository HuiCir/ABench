"""Validate the static ABench case contract without running workloads.

This is intentionally a *preflight* check.  It verifies identity, readiness,
JSON-Schema shape, provenance requirements and the presence of the sandbox
contract; it never trains, infers, judges, or fabricates benchmark results.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = ("case", "harness", "scoring", "example_submission")
REQUIRED_FILES = (
    "README.md",
    "case/anchor_card.yaml",
    "case/task_prompt.md",
    "case/eval_protocol.yaml",
    "case/resources.yaml",
    "case/scoring_config.yaml",
    "case/bench_contract.yaml",
    "case/metric_registry.yaml",
    "case/run_schema.json",
    "case/anti_gaming.md",
    "case/oj.yaml",
    "case/agent/task.md",
    "harness/README.md",
    "scoring/README.md",
    "scoring/run_case.py",
    "example_submission/README.md",
)
VALID_INDEX_STATUS = {"built", "spec", "scaffold", "planned"}
VALID_CARD_STATUS = VALID_INDEX_STATUS | {"deferred"}
VALID_READINESS = {"deferred", "contract_only", "proxy_ready", "reference_ready", "hidden_ready"}
VALID_SANDBOX_PROFILES = {"low_cpu", "single_gpu_24gb", "frozen_inference_24gb", "posttrain_24gb", "pro_4xh100"}
try:
    _report_cfg = yaml.safe_load((ROOT / "spec" / "report_profiles.yaml").read_text(encoding="utf-8")) or {}
    VALID_REPORT_PROFILES = set((_report_cfg.get("profiles") or {}).keys())
except Exception:  # pragma: no cover - the missing file is reported by profile preflight
    VALID_REPORT_PROFILES = {"design_report_v2", "judgment_report_v2"}
VALID_SUBMISSION_KINDS = {"code_only", "code_plus_checkpoint", "report_only"}
VALID_JUDGE_KINDS = {"pending", "tabular_classification", "timed_train_classification", "harness"}
# file names that must not appear in the agent view (judge-side files)
EVAL_SIDE = ("eval_protocol", "scoring_config", "metric_registry", "reference_pool", "anti_gaming",
             "run_schema", "reference_rules", "hidden")
SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message is the useful part
        raise ValueError(f"{path.relative_to(ROOT)}: invalid YAML: {exc}") from exc


def load_index() -> dict[str, Any]:
    doc = load_yaml(ROOT / "index.yaml")
    if not isinstance(doc, dict) or not isinstance(doc.get("tracks"), list):
        raise ValueError("index.yaml: expected a mapping with tracks[]")
    return doc


def case_entries() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    doc = load_index()
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for track in doc.get("tracks", []):
        if not isinstance(track, dict):
            continue
        for item in track.get("cases", []) or []:
            if isinstance(item, dict):
                out.append((track, item))
    return out


def case_ids() -> list[str]:
    return [item["id"] for _, item in case_entries() if "id" in item]


def _required_schema_fields() -> set[str]:
    return {
        "schema_version", "case_id", "provenance", "runner_version",
        "config_hash", "resource_hash", "artifact_hash", "metrics", "run_mode",
    }


def validate_json_schema(cid: str, path: Path, errors: list[str], warnings: list[str]) -> None:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{cid}: case/run_schema.json invalid JSON: {exc}")
        return
    if not isinstance(schema, dict):
        errors.append(f"{cid}: run_schema must be a JSON object")
        return
    if not str(schema.get("$schema", "")).startswith("https://json-schema.org/"):
        errors.append(f"{cid}: run_schema missing a standard JSON Schema $schema URI")
    if schema.get("type") != "object":
        errors.append(f"{cid}: run_schema top-level type must be object")
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list):
        errors.append(f"{cid}: run_schema.required must be a list")
        required = []
    if not isinstance(properties, dict):
        errors.append(f"{cid}: run_schema.properties must be an object")
        properties = {}
    missing = sorted(_required_schema_fields() - set(required))
    if missing:
        errors.append(f"{cid}: run_schema.required missing {','.join(missing)}")
    if "metrics" not in properties or not isinstance(properties.get("metrics"), dict) or properties["metrics"].get("type") != "object":
        errors.append(f"{cid}: run_schema.properties.metrics must be an object schema")
    run_mode = properties.get("run_mode")
    if not isinstance(run_mode, dict) or run_mode.get("const") != "benchmark":
        errors.append(f"{cid}: run_schema.properties.run_mode must const benchmark")
    pareto = properties.get("pareto")
    pareto_required = {"quality", "throughput", "peak_memory", "compute_cost", "stability"}
    if not isinstance(pareto, dict) or pareto.get("type") != "object" or not pareto_required.issubset(set(pareto.get("required", []))):
        errors.append(f"{cid}: run_schema.properties.pareto must declare the five optional Pareto axes")
    hardware = properties.get("hardware_signature")
    hardware_required = {"sandbox_profile", "hardware_class", "cost_unit"}
    if not isinstance(hardware, dict) or hardware.get("type") != "object" or not hardware_required.issubset(set(hardware.get("required", []))):
        errors.append(f"{cid}: run_schema.properties.hardware_signature must declare sandbox/hardware/cost fields")
    case_prop = properties.get("case_id")
    if not isinstance(case_prop, dict) or case_prop.get("const") != cid:
        errors.append(f"{cid}: run_schema case_id must have const {cid!r}")
    prov_prop = properties.get("provenance")
    if not isinstance(prov_prop, dict):
        errors.append(f"{cid}: run_schema.properties.provenance is missing")
    else:
        values = set(prov_prop.get("enum", []))
        if not values.intersection({"benchmark_real", "reference_real", "judge_real", "real"}):
            errors.append(f"{cid}: provenance enum has no accepted real value")
    for field in ("config_hash", "resource_hash", "artifact_hash"):
        prop = properties.get(field)
        if not isinstance(prop, dict) or prop.get("type") != "string":
            errors.append(f"{cid}: run_schema.properties.{field} must be a string hash")
        elif "pattern" not in prop:
            warnings.append(f"{cid}: {field} has no SHA-256 format pattern")
    if "additionalProperties" not in schema:
        warnings.append(f"{cid}: run_schema leaves additionalProperties unspecified")


def validate_metric_registry(cid: str, path: Path, errors: list[str], warnings: list[str]) -> None:
    try:
        registry = load_yaml(path) or {}
    except ValueError as exc:
        errors.append(str(exc)); return
    if not isinstance(registry, dict) or registry.get("case_id") != cid:
        errors.append(f"{cid}: metric_registry.case_id does not match index")
        return
    metrics = registry.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append(f"{cid}: metric_registry.metrics must be a non-empty list")
        return
    ids = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            errors.append(f"{cid}: metric_registry entry must be a mapping")
            continue
        mid = metric.get("id")
        if not isinstance(mid, str) or not mid or mid in ids:
            errors.append(f"{cid}: metric_registry metric ids must be unique non-empty strings")
        ids.add(mid)
        if not isinstance(metric.get("name"), str) or not metric.get("name"):
            errors.append(f"{cid}: metric_registry metric {mid!r} missing name")
        if metric.get("direction") not in {"higher_is_better", "lower_is_better", "benchmark_defined"}:
            errors.append(f"{cid}: metric_registry metric {mid!r} has invalid direction")
        if metric.get("aggregation") not in {"mean", "geometric_mean", "median", "macro_mean", "benchmark_defined"}:
            errors.append(f"{cid}: metric_registry metric {mid!r} has invalid aggregation")
        if metric.get("inference") == "heuristic":
            warnings.append(f"{cid}: metric {mid} direction/unit/aggregation still need calibration")


def profile_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("sandbox_profile") if isinstance(value.get("sandbox_profile"), str) else None
    return None


def validate_oj(cid: str, base: Path, contract: Any, errors: list[str], warnings: list[str]) -> None:
    """OJ-style isolation: the agent sees only task.md + interface files + public data; the hidden set exists only as ciphertext."""
    sys.path.insert(0, str(ROOT / "tools"))
    from build_agent_views import leaky, leak_allow  # noqa: E402
    oj_path = base / "case" / "oj.yaml"
    if not oj_path.is_file():
        return
    try:
        oj = load_yaml(oj_path) or {}
    except ValueError as exc:
        errors.append(str(exc)); return
    if oj.get("case_id") != cid:
        errors.append(f"{cid}: oj.case_id does not match")
    sub = oj.get("submission") or {}
    if sub.get("kind") not in VALID_SUBMISSION_KINDS:
        errors.append(f"{cid}: oj.submission.kind invalid: {sub.get('kind')!r}")
    allowed = sub.get("allowed_top_level")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) and x for x in allowed):
        errors.append(f"{cid}: oj.submission.allowed_top_level must be a non-empty string list")
    else:
        allowed_set = set(allowed)
        if "README.md" not in allowed_set or "report" not in allowed_set or "references.yaml" not in allowed_set:
            errors.append(f"{cid}: allowed_top_level must include README.md, report and references.yaml")
        if sub.get("kind") in ("code_only", "code_plus_checkpoint") and sub.get("entry") not in allowed_set:
            errors.append(f"{cid}: submission.entry must be present in allowed_top_level")
        if sub.get("kind") == "code_plus_checkpoint" and sub.get("checkpoint_dir", "checkpoint") not in allowed_set:
            errors.append(f"{cid}: submission.checkpoint_dir must be present in allowed_top_level")
    if sub.get("report_profile") not in VALID_REPORT_PROFILES:
        errors.append(f"{cid}: oj.submission.report_profile invalid")
    elif isinstance(contract, dict) and contract.get("report_profile") != sub.get("report_profile"):
        errors.append(f"{cid}: oj.submission.report_profile != bench_contract.report_profile")
    else:
        # Keep the per-case files coupled to the central v2 assignment rules;
        # otherwise a copied oj.yaml can silently switch a report contract.
        try:
            from validate_submission import load_profiles, profile_for  # noqa: E402
            expected = profile_for(cid, load_profiles())
            if sub.get("report_profile") != expected:
                errors.append(f"{cid}: oj.submission.report_profile {sub.get('report_profile')!r} != assignment {expected!r}")
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            errors.append(f"{cid}: unable to load report profile assignment: {exc}")
    if sub.get("kind") in ("code_only", "code_plus_checkpoint") and not sub.get("entry"):
        errors.append(f"{cid}: oj.submission.entry required for {sub.get('kind')}")
    judge = oj.get("judge") or {}
    if judge.get("kind") not in VALID_JUDGE_KINDS:
        errors.append(f"{cid}: oj.judge.kind invalid: {judge.get('kind')!r}")
    vis = oj.get("agent_visible") or {}
    visible = [vis.get("task")] + list(vis.get("interface") or [])
    for rel in visible:
        if not rel:
            errors.append(f"{cid}: oj.agent_visible.task missing"); continue
        f = base / rel
        if not f.is_file():
            errors.append(f"{cid}: agent-visible file missing: {rel}"); continue
        if any(tok in Path(rel).name for tok in EVAL_SIDE):
            errors.append(f"{cid}: eval-side file listed as agent-visible: {rel}")
        hits = leaky(f.read_text(encoding="utf-8", errors="replace"), leak_allow(oj))
        if hits:
            errors.append(f"{cid}: agent-visible {rel} mentions evaluation content: {hits}")
    data = oj.get("data") or {}
    awaiting = judge.get("activation") == "awaiting_data"
    if data.get("synthetic") and judge.get("kind") != "pending" and not awaiting:
        errors.append(f"{cid}: synthetic plumbing data cannot back an active judge")
    if awaiting:
        warnings.append(f"{cid}: judge configured, awaiting real data (tools/prepare_*.py)")
        _errs_before = len(errors)
    for d in data.get("public") or []:
        raw_path = d.get("path")
        if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
            errors.append(f"{cid}: public data path must be a non-absolute repository-relative path")
        else:
            resolved = (ROOT / raw_path).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"{cid}: public data path escapes repository: {raw_path}")
        if "hidden" in str(raw_path or ""):
            errors.append(f"{cid}: public data path points into hidden storage")
    hid = data.get("hidden")
    if hid:
        hid_path = hid.get("path")
        if not isinstance(hid_path, str) or not hid_path or Path(hid_path).is_absolute():
            errors.append(f"{cid}: hidden data path must be a non-absolute repository-relative path")
        else:
            resolved = (ROOT / hid_path).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"{cid}: hidden data path escapes repository: {hid_path}")
            if judge.get("kind") != "pending" and not resolved.is_file():
                errors.append(f"{cid}: hidden data file is missing: {hid_path}")
        if not str(hid_path or "").endswith(".enc"):
            errors.append(f"{cid}: hidden split must be stored encrypted (*.enc)")
        if judge.get("kind") != "pending" and not SHA256.match(str(hid.get("sha256_plain", ""))):
            errors.append(f"{cid}: hidden.sha256_plain must be a real digest once the judge is active")
    if awaiting:
        # while waiting for real data, missing data/hashes are only warnings
        for msg in errors[_errs_before:]:
            warnings.append(msg + " (awaiting_data)")
        del errors[_errs_before:]
    if judge.get("kind") == "pending":
        warnings.append(f"{cid}: judge pending (format checks only)")


def validate() -> tuple[list[str], list[str], list[str]]:
    """Return (missing, errors, warnings).  No benchmark execution occurs."""
    missing: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    entries = case_entries()
    ids = [item.get("id") for _, item in entries]
    seen: set[str] = set()
    for cid in ids:
        if not isinstance(cid, str) or not cid:
            errors.append(f"index: case id must be a non-empty string: {cid!r}")
            continue
        if cid in seen:
            errors.append(f"index: duplicate case id {cid}")
        seen.add(cid)
    for track, item in entries:
        cid = item.get("id", "<missing-id>")
        base = ROOT / "cases" / str(cid)
        index_status = item.get("status")
        if index_status not in VALID_INDEX_STATUS:
            errors.append(f"{cid}: invalid index status {index_status!r}")
        expected_path = f"cases/{cid}"
        if item.get("path") != expected_path:
            errors.append(f"{cid}: index path must be {expected_path!r}")
        if not base.exists():
            missing.append(f"{cid}: directory")
            continue
        for name in REQUIRED_DIRS:
            if not (base / name).is_dir():
                missing.append(f"{cid}: {name}/")
        for name in REQUIRED_FILES:
            if not (base / name).is_file():
                missing.append(f"{cid}: {name}")
        contract_path = base / "case" / "bench_contract.yaml"
        if contract_path.is_file():
            try:
                contract = load_yaml(contract_path) or {}
            except ValueError as exc:
                errors.append(str(exc)); contract = {}
            profile = contract.get("report_profile") if isinstance(contract, dict) else None
            if profile not in VALID_REPORT_PROFILES:
                errors.append(f"{cid}: bench_contract.report_profile must be one of {sorted(VALID_REPORT_PROFILES)}")
        validate_oj(cid, base, contract if contract_path.is_file() else {}, errors, warnings)
        card_path = base / "case" / "anchor_card.yaml"
        if card_path.is_file():
            try:
                card = load_yaml(card_path) or {}
            except ValueError as exc:
                errors.append(str(exc)); card = {}
            if not isinstance(card, dict):
                errors.append(f"{cid}: anchor_card must be a mapping")
                card = {}
            if card.get("case_id") != cid:
                errors.append(f"{cid}: anchor_card.case_id does not match index")
            card_status = card.get("status")
            if card_status is not None and card_status not in VALID_CARD_STATUS:
                errors.append(f"{cid}: invalid anchor_card status {card_status!r}")
            readiness = card.get("readiness")
            if readiness not in VALID_READINESS:
                errors.append(f"{cid}: anchor_card.readiness must be one of {sorted(VALID_READINESS)}")
            elif card_status == "deferred" and readiness != "deferred":
                errors.append(f"{cid}: deferred anchor_card must have readiness=deferred")
            elif index_status == "built" and readiness not in {"reference_ready", "hidden_ready"}:
                errors.append(f"{cid}: built index entry requires reference_ready/hidden_ready")
        schema_path = base / "case" / "run_schema.json"
        if schema_path.is_file():
            validate_json_schema(str(cid), schema_path, errors, warnings)
        registry_path = base / "case" / "metric_registry.yaml"
        if registry_path.is_file():
            validate_metric_registry(str(cid), registry_path, errors, warnings)
        contract_path = base / "case" / "bench_contract.yaml"
        if contract_path.is_file():
            try:
                contract = load_yaml(contract_path) or {}
            except ValueError as exc:
                errors.append(str(exc)); contract = {}
            if not isinstance(contract, dict):
                errors.append(f"{cid}: bench_contract must be a mapping")
            else:
                if contract.get("case_id") != cid:
                    errors.append(f"{cid}: bench_contract.case_id does not match index")
                req = set(contract.get("required_provenance", []))
                nested = contract.get("provenance", {})
                if isinstance(nested, dict):
                    req |= set(nested.get("required_fields", []))
                required_hashes = {"config_hash", "resource_hash", "artifact_hash"}
                legacy_hashes = {"config_sha256", "data_sha256", "runtime_sha256", "output_sha256"}
                if not (required_hashes <= req or legacy_hashes <= req):
                    errors.append(f"{cid}: bench_contract must require config/resource/artifact hashes (or legacy SHA-256 set)")
        readme = base / "README.md"
        if readme.is_file() and any(token in readme.read_text(encoding="utf-8").lower()
                                    for token in ("deferred", "blocked", "reserved", "blocker")):
            warnings.append(f"{cid}: README marks this case deferred/blocked")
        protocol_path = base / "case" / "eval_protocol.yaml"
        protocol = {}
        if protocol_path.is_file():
            try:
                protocol = load_yaml(protocol_path) or {}
                if not isinstance(protocol, dict) or not protocol.get("metrics"):
                    warnings.append(f"{cid}: eval_protocol has no machine-readable metrics list")
            except ValueError as exc:
                errors.append(str(exc))
        resources_path = base / "case" / "resources.yaml"
        if resources_path.is_file():
            try:
                resources = load_yaml(resources_path) or {}
                if not isinstance(resources, dict) or not any(resources.get(k) for k in ("sandbox_profile", "resource_profile")):
                    warnings.append(f"{cid}: resources has no sandbox/resource profile; environment remains deferred")
            except ValueError as exc:
                errors.append(str(exc))
        resources_path = base / "case" / "resources.yaml"
        resources = {}
        if resources_path.is_file():
            try:
                resources = load_yaml(resources_path) or {}
            except ValueError as exc:
                errors.append(str(exc))
        profiles = []
        for source in (card if isinstance(card, dict) else {}, protocol, resources):
            value = source.get("resource_profile") if isinstance(source, dict) else None
            name = profile_name(value)
            if not name and isinstance(source, dict):
                name = profile_name(source.get("sandbox_profile"))
            if name:
                profiles.append(name)
        if profiles:
            unknown = sorted(set(profiles) - VALID_SANDBOX_PROFILES)
            if unknown:
                errors.append(f"{cid}: unknown sandbox profile(s): {','.join(unknown)}")
            if len(set(profiles)) > 1:
                errors.append(f"{cid}: resource_profile mismatch across anchor/eval/resources: {sorted(set(profiles))}")
        else:
            warnings.append(f"{cid}: no machine-readable sandbox resource_profile")
        scoring_path = base / "case" / "scoring_config.yaml"
        if scoring_path.is_file():
            try:
                scoring = load_yaml(scoring_path) or {}
                if isinstance(scoring, dict) and scoring.get("objective_mode") == "pareto":
                    pareto = scoring.get("pareto")
                    if not isinstance(pareto, dict) or not isinstance(pareto.get("axes"), list) or not pareto.get("axes"):
                        errors.append(f"{cid}: objective_mode=pareto requires scoring_config.pareto.axes")
            except ValueError as exc:
                errors.append(str(exc))
    return missing, errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit non-zero when anything is missing or a contract error is found")
    ap.add_argument("--verbose", action="store_true", help="print non-blocking warnings one by one")
    args = ap.parse_args()
    try:
        missing, errors, warnings = validate()
    except ValueError as exc:
        print(f"ERROR {exc}")
        return 1
    print(f"cases={len(case_ids())} missing={len(missing)} errors={len(errors)} warnings={len(warnings)}")
    for item in missing:
        print(f"MISSING {item}")
    for item in errors:
        print(f"ERROR {item}")
    if args.verbose:
        for item in warnings:
            print(f"WARNING {item}")
    elif warnings:
        print("WARNING details hidden; rerun with --verbose")
    return 1 if args.strict and (missing or errors) else 0


if __name__ == "__main__":
    sys.exit(main())

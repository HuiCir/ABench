"""Validate sandbox/resource metadata for Pareto-enabled cases.

This is a declarative preflight only: it never runs workloads or creates results.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"single_gpu_24gb", "frozen_inference_24gb", "posttrain_24gb", "low_cpu", "pro_4xh100"}
COST_BY_PROFILE = {
    "single_gpu_24gb": "GPU-h",
    "frozen_inference_24gb": "GPU-h",
    "posttrain_24gb": "GPU-h",
    "low_cpu": "CPU-h",
    "pro_4xh100": "H100-h",
}


def is_pareto(cfg: dict) -> bool:
    return cfg.get("objective_mode") == "pareto" or cfg.get("objective") == "pareto_frontier" or isinstance(cfg.get("pareto"), dict)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    errors: list[str] = []
    checked = 0
    for case_dir in sorted((ROOT / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        base = case_dir / "case"
        cfg_path = base / "scoring_config.yaml"
        if not cfg_path.is_file():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict) or not is_pareto(cfg):
            continue
        checked += 1
        profiles = []
        anchor_profile = None
        anchor_limits = None
        for name in ("anchor_card.yaml", "eval_protocol.yaml"):
            p = base / name
            d = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
            rp = d.get("resource_profile") if isinstance(d, dict) else None
            if not isinstance(rp, dict):
                errors.append(f"{case_dir.name}: {name} resource_profile must be a mapping")
                continue
            for field in ("sandbox_profile", "hardware_class", "cost_unit", "limits"):
                if field not in rp:
                    errors.append(f"{case_dir.name}: {name} resource_profile missing {field}")
            profile = rp.get("sandbox_profile")
            if name == "anchor_card.yaml":
                anchor_profile = profile
                anchor_limits = rp.get("limits")
            if profile not in ALLOWED:
                errors.append(f"{case_dir.name}: {name} unknown sandbox_profile {profile!r}")
            elif rp.get("cost_unit") != COST_BY_PROFILE[profile]:
                errors.append(f"{case_dir.name}: {name} cost_unit does not match {profile}")
            profiles.append(profile)
        if len(profiles) == 2 and profiles[0] != profiles[1]:
            errors.append(f"{case_dir.name}: anchor/eval sandbox_profile mismatch: {profiles}")
        # resources.yaml is consumed by some runners independently of the
        # anchor/eval documents, so catch a third, silently divergent budget.
        resources_path = base / "resources.yaml"
        resources = yaml.safe_load(resources_path.read_text(encoding="utf-8")) if resources_path.is_file() else {}
        if not isinstance(resources, dict):
            errors.append(f"{case_dir.name}: resources.yaml must be a mapping")
            resources = {}
        resource_profile = resources.get("sandbox_profile")
        if isinstance(resources.get("resource_profile"), str):
            resource_profile = resources["resource_profile"]
        if resource_profile != anchor_profile:
            errors.append(f"{case_dir.name}: resources.yaml sandbox_profile mismatch: {resource_profile!r} vs {anchor_profile!r}")
        resource_cost = resources.get("cost_unit")
        if resource_cost != COST_BY_PROFILE.get(anchor_profile):
            errors.append(f"{case_dir.name}: resources.yaml cost_unit mismatch: {resource_cost!r}")
        if resources.get("limits_unit") != COST_BY_PROFILE.get(anchor_profile):
            errors.append(f"{case_dir.name}: resources.yaml limits_unit mismatch: {resources.get('limits_unit')!r}")
        declared_limits = resources.get("resource_limits")
        if not isinstance(declared_limits, dict):
            errors.append(f"{case_dir.name}: resources.yaml resource_limits must be a mapping")
        elif isinstance(anchor_limits, dict):
            for key in ("agent_dev", "per_submission_verify", "reference_precompute"):
                expected = anchor_limits.get(key)
                expected_value = expected.get("value") if isinstance(expected, dict) else expected
                if declared_limits.get(key) != expected_value:
                    errors.append(
                        f"{case_dir.name}: resources.yaml {key} mismatch: "
                        f"{declared_limits.get(key)!r} vs {expected_value!r}"
                    )
    print(f"pareto_cases={checked} errors={len(errors)}")
    if args.verbose:
        for e in errors:
            print("ERROR " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

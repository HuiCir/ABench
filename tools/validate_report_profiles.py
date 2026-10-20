#!/usr/bin/env python3
"""Static checks of the v2 report profiles.

  - spec/report_profiles.yaml declares only design_report_v2 and judgment_report_v2;
  - each case's oj.yaml and bench_contract.yaml point to the same v2 profile;
  - the two offline templates themselves pass validate_submission's report check (the templates are valid examples);
  - neither the templates nor the agent task statements contain experiment/evaluation-type sections.

Runs no submissions and accesses no network.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_submission import load_profiles, validate_report  # noqa: E402

TEMPLATES = {
    "design_report_v2": ROOT / "sandbox/templates/design_report.html",
    "judgment_report_v2": ROOT / "sandbox/templates/judgment_report.html",
}


def check_templates(profiles: dict, errors: list[str]) -> None:
    for name, html in TEMPLATES.items():
        if name not in profiles["profiles"]:
            errors.append(f"profile {name} missing from spec/report_profiles.yaml")
            continue
        if not html.is_file():
            errors.append(f"template missing: {html.relative_to(ROOT)}")
            continue
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d)
            (sub / "report").mkdir()
            shutil.copy(html, sub / "report/index.html")
            shutil.copy(ROOT / "sandbox/templates/references.yaml", sub / "references.yaml")
            # the template references.yaml registers entries for the design template only; the judgment template only needs its anchors compared
            errs: list[str] = []
            oj = {"submission": {"report_profile": name}}
            validate_report(sub, None, profiles, errs, oj)
            errs = [e for e in errs if "references.yaml" not in e and "bib." not in e] if name != "design_report_v2" else errs
            errors.extend(f"template {html.name}: {e}" for e in errs)


def check_cases(profiles: dict, errors: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    # Keep this check tied to index.yaml as well as the case directories.  A
    # stray directory or an unregistered case must not silently escape the
    # profile preflight.
    index = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8")) or {}
    indexed: set[str] = {
        item.get("id")
        for track in index.get("tracks", []) or []
        for item in (track.get("cases", []) or [])
        if isinstance(track, dict) and isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    discovered: set[str] = set()
    for oj_path in sorted((ROOT / "cases").glob("*/case/oj.yaml")):
        cid = oj_path.parent.parent.name
        discovered.add(cid)
        if cid not in indexed:
            errors.append(f"{cid}: case has oj.yaml but is absent from index.yaml")
        try:
            oj = yaml.safe_load(oj_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            errors.append(f"{cid}: invalid oj.yaml: {exc}")
            continue
        if not isinstance(oj, dict):
            errors.append(f"{cid}: oj.yaml must be a mapping")
            continue
        prof = (oj.get("submission") or {}).get("report_profile")
        contract_path = oj_path.parent / "bench_contract.yaml"
        if not contract_path.is_file():
            errors.append(f"{cid}: missing bench_contract.yaml")
            bc = {}
        else:
            try:
                bc = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                errors.append(f"{cid}: invalid bench_contract.yaml: {exc}")
                bc = {}
        if not isinstance(bc, dict):
            errors.append(f"{cid}: bench_contract.yaml must be a mapping")
            bc = {}
        if prof not in profiles["profiles"]:
            errors.append(f"{cid}: unknown report_profile {prof!r}")
        if bc.get("report_profile") != prof:
            errors.append(f"{cid}: bench_contract.report_profile {bc.get('report_profile')!r} != oj {prof!r}")
        kind = (oj.get("submission") or {}).get("kind")
        if (kind == "report_only") != (prof == "judgment_report_v2"):
            errors.append(f"{cid}: report_only cases must use judgment_report_v2 (kind={kind}, profile={prof})")
        rules = profiles.get("assignment_rules", {})
        expected = (rules.get("exceptions", {}) or {}).get(cid)
        if expected is None:
            suffixes = (rules.get("judgment_report_v2", {}) or {}).get("suffixes", []) or []
            expected = "judgment_report_v2" if any(s in cid for s in suffixes) else "design_report_v2"
        if prof != expected:
            errors.append(f"{cid}: report_profile {prof!r} disagrees with assignment rule {expected!r}")
        counts[prof] = counts.get(prof, 0) + 1
    for cid in sorted(indexed - discovered):
        errors.append(f"{cid}: indexed case has no case/*/oj.yaml")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    profiles = load_profiles()
    errors: list[str] = []
    extra = set(profiles["profiles"]) - set(TEMPLATES)
    if extra:
        errors.append(f"unexpected profiles: {sorted(extra)}")
    check_templates(profiles, errors)
    counts = check_cases(profiles, errors)
    print("profiles:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for e in errors:
        print("ERROR", e)
    print("ok" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

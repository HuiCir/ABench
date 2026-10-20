#!/usr/bin/env python3
"""Check whether each case's agent task statement conforms to task statement standard v2.1 (spec/task_statement_standard.md).

Reads files only and runs no submissions. Prints each case's conformance and what is missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_agent_views import leak_allow, leaky  # noqa: E402

SECTIONS = ["Task", "Data", "Interface", "Metrics", "Evaluation", "Sandbox Guide", "Deliverables", "Rules"]


def sections(md: str) -> dict[str, str]:
    out, cur, buf = {}, None, []
    for line in md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def check_case(cid: str) -> dict:
    case = ROOT / "cases" / cid / "case"
    oj = yaml.safe_load((case / "oj.yaml").read_text(encoding="utf-8")) or {}
    task_path = case / "agent" / "task.md"
    md = task_path.read_text(encoding="utf-8") if task_path.is_file() else ""
    sec = sections(md)
    issues: list[str] = []
    for s in SECTIONS:
        if s not in sec or len(sec[s].strip()) < 20:
            issues.append(f"missing or too-short section: {s}")
    metrics = sec.get("Metrics", "")
    if metrics:
        if not re.search(r"^\|.*\|.*\|", metrics, re.M):
            issues.append("Metrics: no metric table")
        if not re.search(r"primary metric|\(primary\)", metrics, re.I):
            issues.append("Metrics: primary metric not marked")
        if not re.search(r"threshold|no threshold", metrics, re.I):
            issues.append("Metrics: threshold not stated (or no-threshold not declared)")
        if not re.search(r"seed|mean|median|average|aggregat", metrics, re.I):
            issues.append("Metrics: aggregation not stated")
        if not re.search(r"valid|invalid", metrics, re.I):
            issues.append("Metrics: no validity rules")
    guide = sec.get("Sandbox Guide", "")
    if guide:
        for need, msg in ((r"ABENCH_DEADLINE", "session limit ABENCH_DEADLINE"), (r"ABENCH_SUBMISSION_DIR", "submission directory"),
                          (r"[Nn]etwork", "network restrictions"), (r"GPU|CPU", "hardware"), (r"selfcheck|self-check", "self-check command")):
            if not re.search(need, guide):
                issues.append(f"Sandbox Guide: missing {msg}")
    task = sec.get("Task", "")
    if task:
        # v2.1 working procedure: retrieval with the offline paper tool, related-work synthesis, a stated list of
        # changes relative to the anchor, and development with the self-check
        for need, msg in ((r"### Workflow", "### Workflow subsection"), (r"abench_papers\.py", "paper search tool"),
                          (r"Related Work", "related-work synthesis (report Section 2)"),
                          (r'id="changes"', 'changes list <ol id="changes">'), (r"selfcheck", "self-check in the workflow")):
            if not re.search(need, task):
                issues.append(f"Task: missing {msg}")
    if oj.get("mode") == "checkpoint" or (oj.get("submission") or {}).get("kind") == "code_plus_checkpoint":
        req = (oj.get("submission") or {}).get("required_files") or []
        if oj.get("standard") and "train.py" not in req:
            issues.append("checkpoint submission without required train.py (the program that produced the checkpoint)")
    vis = oj.get("agent_visible") or {}
    iface = vis.get("interface") or []
    if not iface:
        issues.append("oj.yaml: no agent-visible interface files")
    for rel in iface:
        if not (ROOT / "cases" / cid / rel).is_file():
            issues.append(f"interface file does not exist: {rel}")
    if oj.get("task_statement") != "v2.1":
        issues.append("oj.yaml: task_statement: v2.1 not declared")
    judge = oj.get("judge") or {}
    if judge.get("kind") in (None, "pending"):
        issues.append("judge not configured (judge.kind: pending)")
    hits = leaky(md, leak_allow(oj))
    if hits:
        issues.append(f"statement contains disallowed judge-side terms: {hits}")
    state = "conform" if not issues else ("statement_only" if not [i for i in issues if not i.startswith("judge")] else "draft")
    return {"case": cid, "state": state, "judge": judge.get("kind", "pending"),
            "activation": judge.get("activation", "active" if judge.get("kind") not in (None, "pending") else None),
            "issues": issues}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case")
    ap.add_argument("--strict", action="store_true", help="return non-zero if any case does not conform")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ids = [a.case] if a.case else sorted(p.parent.parent.name for p in (ROOT / "cases").glob("*/case/oj.yaml"))
    res = [check_case(c) for c in ids]
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        counts: dict[str, int] = {}
        for r in res:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        print("task statement standard v2.1:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        for r in res:
            if r["state"] == "conform" or a.case:
                print(f"  {r['case']}: {r['state']}  judge={r['judge']} activation={r['activation']}")
                for i in r["issues"]:
                    print("    -", i)
    return 1 if a.strict and any(r["state"] != "conform" for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())

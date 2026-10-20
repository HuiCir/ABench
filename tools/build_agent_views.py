"""Generate the OJ-view task statement case/agent/task.md and the judge configuration case/oj.yaml for each case.

The agent sees only task.md (plus the interface files listed in oj.yaml, report templates and public data). A generated task.md writes only
the task, anchor, deliverables and resource constraints; metrics and the evaluation procedure are written by hand when a case becomes a standard case.
Task descriptions taken from the old task_prompt.md are filtered sentence by sentence, and any sentence with judge-side terms is removed.

Cases whose oj.yaml is handcrafted (handcrafted: true) or a standard case (standard: true, built by
tools/build_standard_cases.py) are never overwritten.

Usage: python3 tools/build_agent_views.py [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Terms that count as a leak if they appear in the agent view (also used by validate_scaffolds)
LEAK_TERMS = [
    "hidden", "evaluation", "evaluate", "scoring", "score", "metric", "threshold", "reference method", "reference recipe", "reference pool", "reference envelope",
    "benchmark operator", "server-side", "pareto", "leaderboard", "ranking", "held-out", "holdout",
    "precomputed", "noise floor", "probe", "hypervolume", "fidelity", "ablation", "eval_protocol", "scoring_config", "metric_registry",
    "reference_pool", "anti_gaming", "verification budget", "final evaluation", "overall score", "speedup", "envelope",
]
KIND_BY_PROXY = {"MICRO": "code_only", "STREAM": "code_only", "SYN": "code_only", "INFER": "code_only",
                 "FROZEN": "code_plus_checkpoint", "SURG": "code_plus_checkpoint", "SUBSET": "code_plus_checkpoint",
                 "EVAL": "code_plus_checkpoint", "AUDIT": "code_plus_checkpoint", "RERUN": "code_only"}
KIND_OVERRIDE = {"DS-PIPE-002": "code_only"}
ALLOWED_TOP_LEVEL = {
    "code_only": ["README.md", "method.py", "src", "report", "references.yaml", "manifest.yaml", "submission_manifest.yaml"],
    "code_plus_checkpoint": ["README.md", "model.py", "src", "checkpoint", "report", "references.yaml", "manifest.yaml",
                             "submission_manifest.yaml"],
    "report_only": ["README.md", "report", "references.yaml", "manifest.yaml", "submission_manifest.yaml"],
}


# Task statement standard v2.1 (spec/task_statement_standard.md): the statement must state metrics, thresholds and the evaluation method,
# so a case may exempt these terms explicitly in agent_visible.leak_allow of oj.yaml.
# The terms below are never exempt: they point to reference methods and their results, scoring coefficients and judge-side files.
NEVER_ALLOW = {"reference method", "reference recipe", "reference pool", "reference envelope", "envelope", "noise floor", "precomputed", "reference_pool",
               "anti_gaming", "scoring_config", "metric_registry", "eval_protocol", "overall score", "probe", "fidelity"}


def leaky(text: str, allow=()) -> list[str]:
    """Judge-side terms found in text (case-insensitive substring match), minus the case's leak_allow."""
    ok = {a.lower() for a in (allow or ())} - NEVER_ALLOW
    # paper titles in the anchor line (e.g. "Pitfalls of Graph Neural Network Evaluation") are not judge-side content
    text = re.sub(r"(?m)^(?:Anchor: .*|# [A-Z]+-[A-Z0-9]+-\d{3}: .*)$", "", text)
    low = text.lower()
    return [t for t in LEAK_TERMS if t in low and t not in ok]


def leak_allow(oj) -> list[str]:
    return list((((oj or {}).get("agent_visible") or {}).get("leak_allow")) or [])


# Rewrite judge-linked phrasing in the task into neutral design goals
REWRITE = [
    (r"(?:that is |is )?Pareto-optimal over", "with a better trade-off among"), (r"form an? (?:auditable |better )?Pareto front", "reach a better trade-off"), (r"Pareto", "trade-off"),
    (r"once numerical equivalence and the quality threshold pass, ", "while keeping outputs equivalent and model quality intact, "), (r"quality threshold", "quality requirement"),
    (r"hidden capability quality", "capability"), (r"\bhidden ", ""), (r"reference methods?", "existing methods"), (r"metric bias", "measurement bias"),
    (r"evaluation metrics?", "measures"), (r"\s*\(calibration\)", ""), (r"calibration, ", ""), (r"true speedup", "true acceleration"), (r"speedup", "acceleration"), (r"server-side ", ""), (r"the benchmark operator", "the organizers"),
]


def rewrite(text: str) -> str:
    for a, b in REWRITE:
        text = re.sub(a, b, text, flags=re.I)
    return text


def sanitize(text: str) -> str:
    """Sentence by sentence: apply neutral rewrites first, then drop any sentence that still contains a judge-side term."""
    text = rewrite(text)
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    keep = [p.strip() for p in parts if p.strip() and not leaky(p)]
    return " ".join(keep).strip()


def old_task_section(prompt: str) -> str:
    m = re.search(r"^##\s*(Task|Goal|Problem)\s*\n(.*?)(?=^##\s|\Z)", prompt, re.S | re.M)
    if not m:
        return ""
    body = m.group(2)
    body = re.sub(r"^Anchor:.*$", "", body, flags=re.M)
    return sanitize(body)


def submission_kind(cid: str, card: dict, group: str) -> str:
    if cid in KIND_OVERRIDE:
        return KIND_OVERRIDE[cid]
    vp = card.get("verify_plan") or {}
    proxies = [p for p in vp.get("proxy", []) if p != "JUDGE"]
    if "-T7-" in cid or (vp.get("proxy") or [None])[0] == "JUDGE":
        return "report_only"
    for p in proxies:
        if p in KIND_BY_PROXY:
            return KIND_BY_PROXY[p]
    return "code_plus_checkpoint" if group == "pro" else "code_only"


SUBMIT_TEXT = {
    "code_only": ("1. The complete implementation of the method, with entry point `method.py`, written against the interface in `interface/`. After submission the code is trained "
                  "and run from scratch in another environment, so trained parameters need not be submitted."),
    "code_plus_checkpoint": ("1. Model code `model.py` and trained parameters `checkpoint/`, written against the interface in `interface/`; "
                             "after submission only these two parts are loaded to run your model."),
    "report_only": "1. The judgment report is the main deliverable; see the next item.",
}


def unit_text(vp: dict, card: dict) -> str:
    if vp.get("cpu"):
        return "CPU, at most 4 CPU-hours in total during development"
    unit = vp.get("unit") or "GPU-h"
    hw = "4 × H100, 24 hours wall clock" if unit == "H100-h" else "one 24GB GPU"
    return f"{hw}, at most {vp.get('dev')} {unit.replace('GPU-h', 'GPU-hours').replace('H100-h', 'H100-hours')} in total during development"


def build(cid: str, entry: dict, group: str) -> tuple[str, dict]:
    case = ROOT / "cases" / cid / "case"
    card = yaml.safe_load((case / "anchor_card.yaml").read_text(encoding="utf-8")) or {}
    prompt = (case / "task_prompt.md").read_text(encoding="utf-8") if (case / "task_prompt.md").is_file() else ""
    kind = submission_kind(cid, card, group)
    t7 = "-T7-" in cid
    statement = sanitize(entry.get("task", ""))
    if not statement:
        statement = (f"Judge whether a published claim related to {entry.get('anchor')} holds." if t7
                     else f"Design an improved method around {entry.get('anchor')}.")
    statement = statement.rstrip(".;,") + "."
    extra = old_task_section(prompt)
    # complete sentences from the old statement are appended if they add new information (such as optional components or a frozen search space)
    if extra and len(set(re.findall(r"[a-z]{4,}", extra.lower())) - set(re.findall(r"[a-z]{4,}", statement.lower()))) > 6:
        statement += "\n\n" + extra
    anchor = rewrite(entry.get("anchor", ""))
    arx = entry.get("arxiv")
    anchor_line = f"Anchor: **{anchor}**" + (f" (arXiv:{arx})" if arx else "") + "."
    is_t7 = kind == "report_only"
    report_line = ("2. Judgment report `report/index.html`: following the structure of `templates/report_template.html`, give your judgment of the claim, "
                   "the reasons and a minimal verification plan. Write only analysis and a plan, no numerical results.") if is_t7 else (
                   "2. Design report `report/index.html`: following the structure of `templates/report_template.html`, explain the design idea, "
                   "definitions, algorithm, complexity and principles of the method. The report explains the design only, with no experimental procedure or numerical results.")
    vp = card.get("verify_plan") or entry.get("verify_plan") or {}
    lines = [
        f"# {cid}: {anchor}" if is_t7 else f"# {cid}",
        "",
        "The literature cutoff is **2026-07-01**; use only material that was public and verifiable before it.",
        "",
        "## Task" if not is_t7 else "## Question to judge",
        "",
        statement,
        "",
        anchor_line,
        "",
        "## What you submit",
        "",
        SUBMIT_TEXT[kind],
        report_line,
        "3. `references.yaml`: every paper cited in the report, in the format of `templates/references_template.yaml`; "
        "when you borrow a component from a paper, state what was borrowed and where it is in the code.",
        "",
        "## Constraints",
        "",
        "- Use only the data and interfaces provided in `data/` and `interface/`; do not download data or models from the network.",
        f"- Development resources: {unit_text(vp, card)}." if not is_t7 else "- This task needs no training.",
        "- Submission is one-shot.",
        "",
        "## Current status",
        "",
        "The interface and data of this task are still being frozen. Once it is enabled, `interface/` will contain the interface notes and `data/` the usable data.",
        "",
    ]
    text = "\n".join(lines)
    oj = {
        "schema": "abench.oj/v2", "case_id": cid,
        "agent_visible": {"task": "case/agent/task.md", "interface": []},
        "data": {"public": [], "hidden": None, "note": "to be frozen"},
        "submission": {"kind": kind,
                       "entry": None if kind == "report_only" else ("model.py" if kind == "code_plus_checkpoint" else "method.py"),
                       "checkpoint_dir": "checkpoint" if kind == "code_plus_checkpoint" else None,
                       "allowed_top_level": list(ALLOWED_TOP_LEVEL[kind]),
                       "max_bytes": 52428800 if group != "pro" else 21474836480,
                       "report_profile": "judgment_report_v2" if is_t7 else "design_report_v2"},
        "judge": {"kind": "pending", "note": "Judge to be frozen: implemented at activation, reading only the submitted code and checkpoint"},
    }
    return text, oj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only check whether existing task.md files leak judge-side content")
    a = ap.parse_args()
    index = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
    n_written = n_skipped = 0
    leaks = []
    for t in index["tracks"]:
        for c in t["cases"]:
            cid = c["id"]
            case = ROOT / "cases" / cid / "case"
            oj_path, task_path = case / "oj.yaml", case / "agent" / "task.md"
            if not a.check:
                existing = yaml.safe_load(oj_path.read_text(encoding="utf-8")) if oj_path.is_file() else None
                if existing and (existing.get("handcrafted") or existing.get("standard")):
                    n_skipped += 1
                else:
                    text, oj = build(cid, c, t.get("group", "method"))
                    task_path.parent.mkdir(parents=True, exist_ok=True)
                    task_path.write_text(text, encoding="utf-8")
                    oj_path.write_text(yaml.safe_dump(oj, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    n_written += 1
            if task_path.is_file():
                cur = yaml.safe_load(oj_path.read_text(encoding="utf-8")) if oj_path.is_file() else {}
                hits = leaky(task_path.read_text(encoding="utf-8"), leak_allow(cur))
                if hits:
                    leaks.append((cid, hits))
    print(f"written={n_written} handcrafted_skipped={n_skipped} leaks={len(leaks)}")
    for cid, hits in leaks:
        print(f"  LEAK {cid}: {hits}")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())

"""Leak check of agent-visible text: judge-side terms that must not appear in a task statement or interface file.

Used by tools/validate_task_statement.py (statements and interfaces) and tools/sandbox_runner.py (every file copied into the agent
view, checked again at run time). A case may allow the evaluation-rule terms it has to state (metrics, thresholds,
held-out data) through agent_visible.leak_allow in case/oj.yaml; the NEVER_ALLOW terms are never exempt.
"""
from __future__ import annotations

import re

# Terms that count as a leak if they appear in the agent view
LEAK_TERMS = [
    "hidden", "evaluation", "evaluate", "scoring", "score", "metric", "threshold", "reference method", "reference recipe", "reference pool", "reference envelope",
    "benchmark operator", "server-side", "pareto", "leaderboard", "ranking", "held-out", "holdout",
    "precomputed", "noise floor", "probe", "hypervolume", "fidelity", "ablation", "eval_protocol", "scoring_config", "metric_registry",
    "reference_pool", "anti_gaming", "verification budget", "final evaluation", "overall score", "speedup", "envelope",
]

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

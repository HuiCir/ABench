"""Paper weight w and reference quality Q.

w: combination of the venue-tier score V and the citation score C, where C is the percentile of (log) annual citations within the snapshot cohort.
Q: saturating aggregation of w × usage over the sources the agent declared, with penalties for post-cutoff/unresolvable sources.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path


def _d(s: str) -> date:
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def load_snapshot(path: Path, *extra: Path):
    """Load the seed snapshot; extra is an external resolution cache (in the formal benchmark: arXiv/OpenAlex lookups for papers the agent cites that are not in the seed snapshot)."""
    papers = []
    for f in (path, *extra):
        if f.exists():
            papers += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {p["id"]: p for p in papers}


def normalize_id(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    return s


def resolve(source_id: str, snapshot: dict, resolver_cache: dict | None = None):
    """Look up by id or arXiv number in the snapshot; if not found, look in the external resolution cache (in the formal benchmark: arXiv/OpenAlex lookup results)."""
    sid = normalize_id(source_id)
    if sid in snapshot:
        return snapshot[sid]
    for p in snapshot.values():
        if p.get("arxiv") == sid:
            return p
    if resolver_cache and sid in resolver_cache:
        return resolver_cache[sid]
    return None


def citation_scores(snapshot: dict, cutoff: date, cfg: dict | None = None, cite_key: str = "citations_demo"):
    """Citation score C: percentile of log1p annual citations within the "same field × same year" cohort.

    If the cohort has fewer than min_cohort papers, fall back to the same-field cohort; if that is still too small (common for old mathematics
    and other-discipline literature cited for local discoveries), use C = cpy / (cpy + median annual citations of that field), to avoid ranking them together with ML papers.
    """
    cfg = cfg or {}
    min_cohort = cfg.get("min_cohort", 10)
    medians = cfg.get("field_median_cpy", {})
    field = lambda p: p.get("field", "ml")

    def raw_cpy(p):
        age = max((cutoff - _d(p["first_public"])).days / 365.25, 0.5)
        c = p.get("citations_gs")
        if c is None:
            c = p.get(cite_key, 0) or 0
        return c / age

    vals = {pid: math.log1p(raw_cpy(p)) for pid, p in snapshot.items()}
    groups: dict[tuple, list[str]] = {}
    for pid, p in snapshot.items():
        groups.setdefault((field(p), p["first_public"][:4]), []).append(pid)
        groups.setdefault((field(p), None), []).append(pid)

    out = {}
    for pid, p in snapshot.items():
        cohort = groups[(field(p), p["first_public"][:4])]
        if len(cohort) < min_cohort:
            cohort = groups[(field(p), None)]
        if len(cohort) < min_cohort and field(p) != "ml":
            cpy = raw_cpy(p)
            out[pid] = cpy / (cpy + medians.get(field(p), 10))
            continue
        if len(cohort) < min_cohort:
            cohort = list(snapshot)
        pool = sorted(vals[c] for c in cohort)
        below = sum(1 for v in pool if v < vals[pid])
        equal = sum(1 for v in pool if v == vals[pid])
        out[pid] = (below + 0.5 * equal) / len(pool)
    return out


def paper_weight(p: dict, C: float, cutoff: date, cfg: dict) -> dict:
    tiers = cfg["venue_tiers"]
    V = tiers[p["tier"]]["score"]
    if p.get("oral") and p["tier"] == "A":
        V = min(1.0, V + tiers["A_oral_bonus"])
    age = (cutoff - _d(p["first_public"])).days / 365.25
    a = min(1.0, max(age, 0.0) / 2.0)
    w = (1 - 0.6 * a) * V + 0.6 * a * C
    return {"V": round(V, 3), "C": round(C, 3), "age_years": round(age, 2), "w": round(w, 4)}


def all_weights(snapshot: dict, cutoff: date, cfg: dict) -> dict:
    C = citation_scores(snapshot, cutoff, cfg)
    return {pid: paper_weight(p, C[pid], cutoff, cfg) for pid, p in snapshot.items()}


def reference_quality(sources: list, usage: dict, snapshot: dict, weights: dict,
                      cutoff: date, cfg: dict, resolver_cache: dict | None = None) -> dict:
    levels = cfg["usage_levels"]
    pen = cfg["penalties"]
    rows, penalty = [], 1.0
    for s in sources:
        sid = s["id"]
        p = resolve(sid, snapshot, resolver_cache)
        u_level = usage.get(sid, "decorative")
        row = {"id": sid, "usage": u_level}
        if p is None:
            row.update(status="unresolvable", w=0.0, contribution=0.0)
            penalty *= pen["unresolvable_source"]
        elif _d(p["first_public"]) >= cutoff:
            row.update(status="post_cutoff", w=0.0, contribution=0.0, title=p["title"])
            penalty *= pen["post_cutoff_source"]
        else:
            w = weights[p["id"]]["w"]
            row.update(status="ok", title=p["title"], w=w, contribution=round(w * levels[u_level], 4))
        rows.append(row)

    counted = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: -r["contribution"])
    counted = counted[: cfg["max_counted_sources"]]
    prod = 1.0
    for r in counted:
        prod *= 1 - r["contribution"]
    Q_raw = 1 - prod
    return {"Q": round(Q_raw * penalty, 4), "Q_raw": round(Q_raw, 4), "penalty": round(penalty, 4),
            "sources": rows}

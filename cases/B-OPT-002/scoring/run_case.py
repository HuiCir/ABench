"""ABench case scoring pipeline (real run results → L0/S5 scoring → report).

`--runs` and `--judge` must come from the benchmark operator's real runs and blind-review artifacts; this entry point generates no
training curves, reference results or judge results.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

# The ill-conditioned quadratic/power-iteration probes of the reference rules deliberately hit extreme values; L0 still decides via explicit finite/descends
# checks, and silencing NumPy's repeated warnings only keeps the demo output readable.
np.seterr(over="ignore", divide="ignore", invalid="ignore", under="ignore")
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "scoring")]

from optim_api import Ops, state_numel                       # noqa: E402
from reference_rules import REFERENCE_RULES                  # noqa: E402
import divergence as dv                                      # noqa: E402
import paper_weight as pw                                    # noqa: E402
import discovery as dsc                                      # noqa: E402
import sota_margin as sm                                     # noqa: E402

_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")

def load_real_payload(path: str, label: str, case_id: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"{label} not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"{label} is not valid JSON: {exc}")
    if not isinstance(obj, dict):
        raise SystemExit(f"{label} must be a JSON object")
    required = ("schema_version", "case_id", "provenance", "runner_version", "config_hash", "resource_hash", "artifact_hash", "metrics")
    missing = [k for k in required if k not in obj]
    if missing:
        raise SystemExit(f"{label} missing required fields: {', '.join(missing)}")
    if obj.get("case_id") != case_id:
        raise SystemExit(f"{label}.case_id does not match {case_id}")
    if str(obj.get("provenance", "")).lower() not in {"benchmark_real", "reference_real", "judge_real", "real"}:
        raise SystemExit(f"{label} must carry accepted real provenance")
    if obj.get("run_mode", "benchmark") != "benchmark":
        raise SystemExit(f"{label}.run_mode={obj.get('run_mode')!r}: smoke runs are not scored")
    for field in ("config_hash", "resource_hash", "artifact_hash"):
        if not isinstance(obj.get(field), str) or not _SHA256.fullmatch(obj[field]):
            raise SystemExit(f"{label}.{field} must be a SHA-256 hex digest")
    if not isinstance(obj.get("metrics"), dict):
        raise SystemExit(f"{label}.metrics must be a JSON object")
    return obj

import primitive_score as ps                                 # noqa: E402


def load_rule(sub_dir: Path):
    """Load the submitted rule and instantiate it with the defaults from hparam_space.yaml (excluding lr).

    In the formal benchmark this uses the best hyperparameters found by the benchmark operator at L1/L2; the probes do not depend on the lr scale (cos and spectral distribution are scale-invariant).
    """
    import functools
    spec = importlib.util.spec_from_file_location("submission_optimizer", sub_dir / "optimizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = yaml.safe_load((sub_dir / "hparam_space.yaml").read_text(encoding="utf-8"))
    kw = {k: v["default"] for k, v in hs.get("tunable", {}).items() if k != "lr"}
    kw.update(hs.get("fixed", {}) or {})
    return functools.partial(mod.RULE, **kw), (sub_dir / "optimizer.py").read_text(encoding="utf-8")


def load_sources(sub_dir: Path) -> list:
    """Load v2 references.yaml, with a legacy sources.yaml fallback.

    The scoring functions consume the normalized ``id/borrowed/code`` shape;
    report submissions use ``references.yaml`` so the report validator can
    check that every citation has a registered provenance entry.
    """
    ref = sub_dir / "references.yaml"
    if ref.is_file():
        doc = yaml.safe_load(ref.read_text(encoding="utf-8")) or {}
        out = []
        for r in doc.get("references", []):
            if r.get("arxiv"):
                sid = f"arXiv:{r['arxiv']}"
            elif r.get("doi"):
                sid = f"doi:{r['doi']}"
            else:
                sid = f"url:{r.get('url')}"
            out.append({"id": sid, "borrowed": r.get("used_for", ""),
                        "code": r.get("code_location") or "unused"})
        return out
    legacy = sub_dir / "sources.yaml"
    if legacy.is_file():
        return (yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}).get("sources", [])
    raise SystemExit("submission must provide references.yaml (or legacy sources.yaml)")


# ---------------------------------------------------------------- L0
def l0_checks(rule_cls) -> dict:
    ops = Ops("numpy")
    rng = np.random.default_rng(0)
    res = {}

    # interface and no rewriting of inputs
    rule = rule_cls()
    p = rng.standard_normal((32, 24)); g = rng.standard_normal((32, 24))
    p0, g0 = p.copy(), g.copy()
    st = rule.init_state(p, ops)
    d = rule.update(p, g, st, 1, 1e-3, ops)
    res["interface"] = d.shape == p.shape and np.array_equal(p, p0) and np.array_equal(g, g0)

    # state memory
    ratios = []
    for shape in [(256, 64), (64, 256), (128,)]:
        q = rng.standard_normal(shape)
        ratios.append(state_numel(rule_cls().init_state(q, ops), ops) / q.size)
    res["state_ratio_max"] = round(max(ratios), 3)
    res["state_memory<=3x"] = max(ratios) <= 3.0

    # numerical stability: ill-conditioned quadratic for 1000 steps
    rule = rule_cls()
    H = np.diag(np.geomspace(1e-3, 1e2, 48))
    W = rng.standard_normal((48, 16))
    st = rule.init_state(W, ops)
    f0 = 0.5 * np.sum(W * (H @ W))
    for t in range(1, 1001):
        grad = H @ W + 0.01 * rng.standard_normal(W.shape)
        W = W + rule.update(W, grad, st, t, 3e-3, ops)
    f1 = 0.5 * np.sum(W * (H @ W))
    res["finite"] = bool(np.all(np.isfinite(W)))
    res["quadratic_loss_ratio"] = round(float(f1 / f0), 4)
    res["descends"] = bool(f1 < f0)

    # determinism
    outs = []
    for _ in range(2):
        r = rule_cls(); q = np.ones((16, 8)) * 0.1; s = r.init_state(q, ops)
        rr = np.random.default_rng(3)
        for t in range(1, 21):
            q = q + r.update(q, rr.standard_normal(q.shape), s, t, 1e-2, ops)
        outs.append(q)
    res["deterministic"] = bool(np.allclose(outs[0], outs[1], atol=1e-6))

    res["pass"] = all(res[k] for k in ["interface", "state_memory<=3x", "finite", "descends", "deterministic"])
    return res


# ---------------------------------------------------------------- main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=str(ROOT / "example_submission"))
    ap.add_argument("--runs", required=True, help="real run-result JSON from the benchmark operator")
    ap.add_argument("--judge", required=True, help="blind-review and source-usage JSON")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example_submission"))
    a = ap.parse_args()

    sub_dir = Path(a.submission)
    if not sub_dir.exists():
        raise SystemExit(f"submission not found: {sub_dir}")
    sys.path.insert(0, str(sub_dir))
    case = ROOT / "case"
    card = yaml.safe_load((case / "anchor_card.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((case / "scoring_config.yaml").read_text(encoding="utf-8"))
    proto = yaml.safe_load((case / "eval_protocol.yaml").read_text(encoding="utf-8"))
    cutoff = date.fromisoformat(card["cutoff"])
    runs = load_real_payload(a.runs, "runs", card["case_id"])
    judge = load_real_payload(a.judge, "judge", card["case_id"])
    sources = load_sources(sub_dir)

    rule_cls, sub_src = load_rule(sub_dir)
    report = {"case_id": card["case_id"], "submission": rule_cls().name,
              "data_provenance": runs.get("_note", "real")}

    # S4: L0
    report["L0"] = l0_checks(rule_cls)

    # S5-1: reference quality Q
    snapshot = pw.load_snapshot(case / "paper_snapshot/seed_neighborhood.jsonl",
                                case / "paper_snapshot/resolver_cache.jsonl")
    weights = pw.all_weights(snapshot, cutoff, cfg["paper_weight"])
    report["paper_weights"] = weights
    report["Q"] = pw.reference_quality(sources, judge["source_usage"], snapshot, weights, cutoff,
                                       cfg["reference_quality"])

    # S5-2: SOTA margin M
    report["M"] = sm.sota_margin(runs, proto["settings_weights"], cfg["sota_margin"])

    # S5-3: divergence D
    refs = {k: v() for k, v in REFERENCE_RULES.items()}
    beh = dv.behavioral_divergence(rule_cls(), refs)
    code = dv.code_divergence(sub_src, REFERENCE_RULES)
    mech = dv.mechanism_divergence(judge)
    floor = dv.hp_noise_floor(REFERENCE_RULES)
    comb = dv.combine(beh, code, mech, cfg["divergence"], floor["floor"])
    report["D"] = {**comb, "behavioral": beh, "code": code, "mechanism": mech, "noise_floor": floor}

    # S5-4: local discovery X
    dfile = sub_dir / "discoveries.yaml"
    decls = (yaml.safe_load(dfile.read_text(encoding="utf-8")) or {}).get("discoveries", []) if dfile.exists() else []
    report["X"] = dsc.discovery_score(decls, runs, report["M"], judge, cfg["discovery"],
                                      cfg["sota_margin"].get("cap", 2.0))

    # S5-5: primitive-search audit (grammar, continuous log, fixed seed, deterministic replay)
    search_cfg = proto.get("search", {})
    report["E"] = ps.validate_search(sub_dir,
                                      candidate_limit=int(search_cfg.get("candidate_limit", 256)),
                                      seed=int(search_cfg.get("seed", 17)))

    # overall score
    e = cfg["composite"]["exponents"]
    beta = cfg["composite"]["discovery_bonus"]
    M, D, Q, X = report["M"]["M"], report["D"]["D"], report["Q"]["Q"], report["X"]["X"]
    gates = {"L0": report["L0"]["pass"], "SOTA": report["M"]["gate_pass"],
             "differentiated": report["D"]["gate_pass"], "search_audit": report["E"]["pass"]}
    base = 100 * (M ** e["M"]) * (D ** e["D"]) * (Q ** e["Q"])
    raw = base * (1 + beta * X)
    report["composite"] = {"gates": gates, "base": round(base, 2), "score_if_gates_pass": round(raw, 2),
                           "score": round(raw if all(gates.values()) else 0.0, 2),
                           "M": M, "D": D, "Q": Q, "X": X, "exponents": e, "discovery_bonus": beta}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "score_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    (out / "score_report.md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps(report["composite"], ensure_ascii=False, indent=1))


def render_md(r: dict) -> str:
    c, M, D, Q = r["composite"], r["M"], r["D"], r["Q"]
    yes = lambda b: "pass" if b else "fail"
    L = [f"# ABench score report: {r['case_id']} / {r['submission']}", ""]
    if "SIMULATED" in r["data_provenance"]:
        L += ["> **Training results and judge outputs are simulated data**; this report only validates the pipeline and does not represent the method's real performance.", ""]
    L += [f"## Overall score: {c['score']}", "",
          "| Threshold | Result |", "|---|---|"] + [f"| {k} | {yes(v)} |" for k, v in c["gates"].items()] + [
          "", f"Score = 100 · M^{c['exponents']['M']} · D^{c['exponents']['D']} · Q^{c['exponents']['Q']} · (1 + {c['discovery_bonus']}·X)"
          f" = 100 · {c['M']}^{c['exponents']['M']} · {c['D']}^{c['exponents']['D']} · {c['Q']}^{c['exponents']['Q']} · (1 + {c['discovery_bonus']}·{c['X']})"
          f" = {c['base']} × {1 + c['discovery_bonus'] * c['X']:.3f} = **{c['score_if_gates_pass']}**", ""]

    L += ["## L0 automatic checks", "", "| Check | Result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in r["L0"].items()]

    L += ["", f"## SOTA margin M = {M['M']}", "",
          f"Aggregate effective speedup (relative to the reference envelope): **{M['aggregate_speedup']}×**, lower confidence bound {M['aggregate_speedup_lcb']}×", "",
          "| Setting | Speedup | Per seed | Submission loss | Best reference at 1× budget | Its loss | Envelope-dominant method |", "|---|---|---|---|---|---|---|"]
    for k, v in M["per_setting"].items():
        L.append(f"| {k} | {v['speedup']} | {v['per_seed']} | {v['submission_loss_mean']} | {v['best_ref_at_1x']} | "
                 f"{v['best_ref_loss_at_1x']} | {v['envelope_leader_at_budget']} |")
    L += ["", "| SOTA threshold | Result |", "|---|---|"] + [f"| {k} | {yes(v)} |" for k, v in M["gate_checks"].items()]
    L += ["", "OOD: " + "; ".join(f"{k} degraded {v['regression']*100:.2f}%" for k, v in M["ood"].items())]

    b, cd, mc = D["behavioral"], D["code"], D["mechanism"]
    L += ["", f"## Method divergence D = {D['D']}", "",
          f"- Behavioral divergence D_beh = {b['D_beh']} (global {b['D_global']}, patch {b['D_patch']}), nearest reference: {b['nearest_reference']}",
          f"- Hyperparameter-perturbation noise floor {D['noise_floor']['floor']} ({D['noise_floor']['by_reference']}), clone threshold {D['behavioral_threshold']}",
          f"- Code divergence D_code = {cd['D_code']}, nearest reference: {cd['nearest_reference']}",
          f"- Mechanism divergence D_mech = {mc['D_mech']} (judge levels {mc['levels']}), closest prior work: {mc['nearest_prior_art']}", "",
          "| Tensor | Nearest reference | Similarity |", "|---|---|---|"]
    L += [f"| {n} | {v['ref']} | {v['sim']} |" for n, v in b["nearest_by_tensor"].items()]
    L += ["", "Mean behavioral similarity to each reference method: " + ", ".join(f"{k} {v}" for k, v in b["similarity_by_reference"].items())]
    L += ["", "| Divergence threshold | Result |", "|---|---|"] + [f"| {k} | {yes(v)} |" for k, v in D["gate_checks"].items()]

    L += ["", f"## Reference quality Q = {Q['Q']} (before penalty {Q['Q_raw']}, penalty factor {Q['penalty']})", "",
          "| Source | Status | Usage | Paper weight w | Contribution w·u |", "|---|---|---|---|---|"]
    L += [f"| {s['id']} | {s['status']} | {s['usage']} | {s['w']} | {s['contribution']} |" for s in Q["sources"]]
    X = r["X"]
    L += ["", f"## Local discovery X = {X['X']}", ""]
    if not X["declarations"]:
        L += ["No local discovery claimed."]
    else:
        L += ["| Module | Source | Status | Full S | Ablated S | Effect ln | a | n | c | r | x |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in X["declarations"]:
            L.append(f"| {d['module']} | {d['source']} ({d.get('source_year')}, {d.get('source_field')}) | {d['status']} | "
                     f"{d.get('S_full','')} | {d.get('S_ablated','')} | {d.get('effect_ln','')} | {d.get('a','')} | "
                     f"{d.get('n','')} | {d.get('c','')} | {d.get('r','')} | {d['x']} |")
        L += [""] + [f"- {d['module']}: {d.get('ml_precedent') or ''}" for d in X["declarations"] if d.get("ml_precedent")]
    E = r.get("E")
    if E is not None:
        L += ["", f"## Primitive-search audit E", "",
              f"- Result: {yes(E['pass'])}; grammar={E.get('grammar')}; evaluations={E.get('evaluations')}; "
              f"valid candidates={E.get('valid_programs')}; first best candidate={E.get('best_eval')}",
              f"- Target reached: {yes(E.get('target_reached'))}; replayable: {yes(E.get('replay'))}; "
              f"errors: {'; '.join(E.get('errors', [])) or 'none'}"]
    L += ["", "## Appendix: paper weights of the seed neighborhood", "", "| Paper | V | C | Age | w |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['V']} | {v['C']} | {v['age_years']} | {v['w']} |"
          for k, v in sorted(r["paper_weights"].items(), key=lambda x: -x[1]["w"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

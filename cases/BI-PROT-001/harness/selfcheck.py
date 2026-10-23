#!/usr/bin/env python3
"""Self-check for an ABench standard case (identical copy in each case's harness/ directory).

  python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick       # interface check on a small dev slice, 1 seed
  python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --seeds 2     # full dev split, 2 seeds

It runs your submission through exactly the same stage protocol as the evaluation (abench_stage.py, the case's
stages.py and metrics.py), on a dev split carved out of the public data. Differences from the evaluation: the
data (dev split instead of the held-out set), the seeds, and, with --quick, smaller data and shorter time limits.
Dev metrics are only an estimate of held-out metrics. Self-check time counts against your development budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from abench_stage import load_module, run_pipeline  # noqa: E402


DOCS = ("report/index.html", "references.yaml")     # needed for submission, not for running the pipeline


def precheck(sub: Path, rules: dict) -> list[str]:
    errs = []
    for f in rules.get("required", []):
        if f not in DOCS and not (sub / f).exists():
            errs.append(f"{f} is missing")
    allowed_top = set(rules.get("allowed_top_level", []))
    for p in sub.iterdir():
        if allowed_top and p.name not in allowed_top:
            errs.append(f"not allowed at the top level of the submission: {p.name}")
    code_suffixes = set(rules.get("code_suffixes", [".py", ".yaml", ".yml", ".md", ".txt"]))
    data_dirs = set(rules.get("data_dirs", []))
    code_bytes = 0
    for p in sub.rglob("*"):
        if not p.is_file():
            continue
        top = p.relative_to(sub).parts[0]
        if top == "report" or top in data_dirs:
            continue
        if p.suffix.lower() not in code_suffixes:
            errs.append(f"file type not allowed in the code area: {p.relative_to(sub)}")
        code_bytes += p.stat().st_size
    if code_bytes > rules.get("code_max_bytes", 200_000):
        errs.append(f"code and configuration total {code_bytes} bytes, over the {rules.get('code_max_bytes', 200_000)}-byte limit")
    for d in data_dirs:
        size = sum(p.stat().st_size for p in (sub / d).rglob("*") if p.is_file()) if (sub / d).is_dir() else 0
        cap = rules.get("data_max_bytes", {}).get(d)
        if cap and size > cap:
            errs.append(f"{d}/ is {size} bytes, over the {cap}-byte limit")
    return errs


def aggregate(per_seed: list[dict]) -> dict:
    import numpy as np
    keys = sorted({k for m in per_seed for k in m})
    return {k: round(float(np.mean([m[k] for m in per_seed if k in m])), 6) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", default=os.environ.get("ABENCH_SUBMISSION_DIR", "."))
    ap.add_argument("--data", default=os.environ.get("ABENCH_DATA_DIR"))
    ap.add_argument("--models", default=os.environ.get("ABENCH_MODELS_DIR"))
    ap.add_argument("--device", default=os.environ.get("ABENCH_DEVICE", "cuda"))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    stages_path = HERE / "stages.py"
    stages = load_module(stages_path, "abench_case_stages")
    metrics = load_module(HERE / "metrics.py", "abench_case_metrics")
    sub = Path(a.submission).resolve()
    rules_path = HERE / "submission_rules.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.is_file() else {}
    errs = precheck(sub, rules)
    if errs:
        print("\n".join("✗ " + e for e in errs))
        return 1
    pending = [f for f in DOCS if f in rules.get("required", []) and not (sub / f).exists()]
    if pending:
        print("note: not yet written (required for submission, not for this check): " + ", ".join(pending))
    seeds = 1 if a.quick else a.seeds
    per_seed = []
    with tempfile.TemporaryDirectory(prefix="abench-selfcheck-") as d:
        d = Path(d)
        dev = stages.make_dev(Path(a.data) if a.data else None, d / "dev", quick=a.quick)
        for i in range(seeds):
            seed = 1000 + i
            res = run_pipeline(stages, stages_path, sub, d / f"run{i}", public=dev["public"],
                               models=Path(a.models) if a.models else None, held_out=dev["held_out"], seed=seed,
                               device=a.device, scale=dev.get("time_scale", 1.0), held_pre=dev.get("held_pre"))
            for rec in res["stages"]:
                line = f"seed {seed} stage {rec['stage']}: {rec['status']}"
                if "timed_seconds" in rec:
                    line += f"  timed {rec['timed_seconds']:.3f}s"
                if "finish_seconds" in rec:
                    line += f"  finish {rec['finish_seconds']:.3f}s"
                print(line, flush=True)
                if rec["status"] != "ok":
                    print(rec.get("stderr_tail", "")[-2000:])
                    return 1
            try:
                m = metrics.score(res["outputs"], dev["reference"], getattr(stages, "PARAMS", {}), res["seconds"])
            except metrics.OutputError as e:
                print(f"✗ wrong output format: {e}")
                return 1
            print(f"seed {seed} dev metrics: {json.dumps(m)}", flush=True)
            groups, primary = getattr(metrics, "RESAMPLE", None), (getattr(metrics, "PRIMARY", None) or {}).get("metric")
            if groups and primary:
                from abench_metrics import bootstrap_ci
                try:
                    ci = bootstrap_ci(metrics.score, res["outputs"], dev["reference"], getattr(stages, "PARAMS", {}),
                                      res["seconds"], groups, primary)
                except (KeyError, IndexError, TypeError, ValueError):
                    ci = None
                if ci:
                    print(f"seed {seed} {primary} 95% interval over dev examples: [{ci['low']:.4f}, {ci['high']:.4f}]"
                          " (differences inside it are not meaningful)", flush=True)
            per_seed.append(m)
    print(json.dumps({"seeds": len(per_seed), "quick": a.quick, "dev_mean": aggregate(per_seed),
                      "submission_complete": not pending}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

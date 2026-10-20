#!/usr/bin/env python3
"""IC-SPD-001 self-check: check your submission on the public dev split with exactly the same timing protocol as the evaluation.

  python3 $ABENCH_VIEW_DIR/interface/selfcheck.py                  # 2 seeds, train on 45k, test on dev 5k
  python3 $ABENCH_VIEW_DIR/interface/selfcheck.py --seeds 5        # same number of seeds as the evaluation
  python3 $ABENCH_VIEW_DIR/interface/selfcheck.py --quick          # interface check only: 2k images, 1 seed

The only difference from the evaluation is the data: here training uses the 45,000 images of train.npz outside dev_idx and testing uses the 5,000 images of dev_idx;
the evaluation trains on all 50,000 images and tests on test images you cannot see, with different seeds. Timing, cache isolation and the inference cap are exactly the same.
Self-check run time counts against your development budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speedrun_runner import run_once  # noqa: E402

CODE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt"}


def precheck(sub: Path) -> list[str]:
    errs = []
    if not (sub / "method.py").is_file():
        errs.append("method.py is missing")
    code = [p for p in sub.rglob("*") if p.is_file() and p.relative_to(sub).parts[0] not in ("report",)]
    for p in code:
        if p.suffix.lower() not in CODE_SUFFIXES and p.name != "references.yaml":
            errs.append(f"file type not allowed (the code area may contain only source files): {p.relative_to(sub)}")
    size = sum(p.stat().st_size for p in code)
    if size > 200_000:
        errs.append(f"code and configuration total {size} bytes, over the 200,000-byte limit")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", default=os.environ.get("ABENCH_SUBMISSION_DIR", "."))
    ap.add_argument("--data", default=os.path.join(os.environ.get("ABENCH_DATA_DIR", "."), "train.npz"))
    ap.add_argument("--device", default=os.environ.get("ABENCH_DEVICE", "cuda"))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    sub = Path(a.submission).resolve()
    errs = precheck(sub)
    if errs:
        print("\n".join("✗ " + e for e in errs))
        return 1
    with np.load(a.data, allow_pickle=False) as z:
        x, y, dev = z["x"], z["y"], z["dev_idx"]
    mask = np.ones(len(y), bool)
    mask[dev] = False
    xtr, ytr, xdv, ydv = x[mask], y[mask], x[dev], y[dev]
    if a.quick:
        xtr, ytr, xdv, ydv, a.seeds = xtr[:2000], ytr[:2000], xdv[:500], ydv[:500], 1
    rows = []
    with tempfile.TemporaryDirectory(prefix="ic-spd-selfcheck-") as d:
        d = Path(d)
        np.savez(d / "train.npz", x=xtr, y=ytr)
        np.save(d / "x_dev.npy", xdv, allow_pickle=False)
        for i in range(a.seeds):
            seed = 1000 + i
            r = run_once(sub, d / "train.npz", d / "x_dev.npy", seed, d / f"run{i}", device=a.device)
            if r["status"] == "ok":
                P = np.load(r["pred_path"])
                if P.shape != (len(ydv), 10):
                    r["status"] = f"wrong_output_format {list(P.shape)}"
                else:
                    r["dev_accuracy"] = round(float((P.argmax(1) == ydv).mean()), 4)
            rows.append(r)
            line = (f"seed {seed}: train {r['train_seconds']:.3f}s  predict {r['predict_seconds']:.3f}s  "
                    f"dev acc {r['dev_accuracy']:.4f}") if r["status"] == "ok" and "dev_accuracy" in r else \
                   f"seed {seed}: {r['status']}\n{r.get('stderr_tail', '')[-1500:]}"
            print(line, flush=True)
            if r["status"] != "ok":
                return 1
    ok = [r for r in rows if "dev_accuracy" in r]
    summary = {"seeds": len(ok), "train_seconds_mean": round(float(np.mean([r["train_seconds"] for r in ok])), 4),
               "dev_accuracy_mean": round(float(np.mean([r["dev_accuracy"] for r in ok])), 4), "quick": a.quick}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

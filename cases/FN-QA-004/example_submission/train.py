"""FN-QA-004 reference submission, preparation program (no gradient training; CPU, about a minute).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--quick]

Writes checkpoint/idf.json (inverse document frequencies over the training questions and facts) and
checkpoint/exemplars.json (training questions with their gold evidence and program, the few-shot pool).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import retrieve as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    train = json.loads((Path(a.data) / "train.json").read_text(encoding="utf-8"))
    if a.quick:
        train = train[:300]
    texts = [x["qa"]["question"] for x in train] + [s for x in train[:2000] for _, s in R.facts(x)]
    idf = R.fit_idf(texts)
    bank = []
    for x in train:
        f = dict(R.facts(x))
        ev = [f[k] for k in x["qa"].get("gold_inds", {}) if k in f]
        if ev and len(x["qa"]["program"]) < 200:
            bank.append({"question": x["qa"]["question"], "facts": ev, "program": x["qa"]["program"]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "idf.json").write_text(json.dumps(idf))
    (out / "exemplars.json").write_text(json.dumps(bank))
    print(f"{len(bank)} exemplars, {len(idf)} terms")


if __name__ == "__main__":
    main()

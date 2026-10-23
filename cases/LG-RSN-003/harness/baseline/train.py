"""LG-RSN-003 reference submission, preparation program: the reference trains nothing; it records the public task
names and label sets in checkpoint/tasks.json for inspection.

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--quick]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    pub = json.loads((Path(a.data) / "legalbench_public.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps({t["name"]: t["labels"] for t in pub["tasks"]}, indent=1))
    print(len(pub["tasks"]), "public tasks")


if __name__ == "__main__":
    main()

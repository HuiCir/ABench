"""Reference submission (\"training\"): nothing is fitted before evaluation; this writes the settings to checkpoint/.
The ridge strength 1.0 was the default of the design, not tuned on held-out assays.

  python3 train.py --data $ABENCH_DATA_DIR --out checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"ridge": 1.0, "models": ["esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

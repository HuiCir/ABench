"""Reference submission (\"training\"): nothing is trained; this writes the settings to checkpoint/.

  python3 train.py --out checkpoint
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
    (out / "config.json").write_text(json.dumps({"k0": 4}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

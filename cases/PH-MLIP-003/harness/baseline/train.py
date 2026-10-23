"""Reference submission (training): fit the random-feature force field of ff.py to each molecule's training
configurations (energies and forces) and save it to checkpoint/<molecule>.npz.

  python3 train.py --data $ABENCH_DATA_DIR --out checkpoint          # all 1,000 configurations per molecule
  python3 train.py --data $ABENCH_DATA_DIR --out ckpt_dev --dev      # the self-check's 800-configuration split
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff import fit  # noqa: E402

MOLECULES = ("aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene")
N_FEATURES, ENERGY_WEIGHT, REG = 2000, 1.0, 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    if a.dev:
        sys.path.insert(0, str(Path(os.environ.get("ABENCH_VIEW_DIR", ".")) / "interface"))
        from stages import make_dev
        data = make_dev(data, Path(tempfile.mkdtemp()))["public"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in MOLECULES:
        with np.load(data / f"{m}.npz", allow_pickle=False) as z:
            pos, e, f = z["pos"], z["energy"], z["forces"]
        model = fit(pos, e, f, n_features=200 if a.quick else N_FEATURES, reg=REG, energy_weight=ENERGY_WEIGHT)
        np.savez(out / f"{m}.npz", W=model.W, b=model.b, w=model.w, c=np.asarray(model.c), mu=model.mu, sd=model.sd)
        print(m, "fitted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

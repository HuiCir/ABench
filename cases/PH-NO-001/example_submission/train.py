"""PH-NO-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--include-dev] [--quick]

Fits, for each of the lowest N_MODES wavenumbers, the complex coefficients of u_{t+1}(k) = a_k u_t(k) + b_k u_{t-1}(k)
by least squares over all training transitions. By default the dev trajectories (dev_idx) are left out so that
selfcheck.py gives an unbiased dev estimate. Saves checkpoint/spectral.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import N_MODES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)          # deterministic; kept for a uniform command line
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true")         # least squares: nothing to shorten
    a = ap.parse_args()
    with np.load(Path(a.data) / "train.npz", allow_pickle=False) as z:
        u, dev = z["u"], z["dev_idx"]
    keep = np.ones(len(u), bool)
    if not a.include_dev:
        keep[dev] = False
    u = u[keep].astype(np.float64)                          # (N, T, X)
    U = np.fft.rfft(u, axis=-1)[..., :N_MODES]
    cur, prev, nxt = U[:, 1:-1], U[:, :-2], U[:, 2:]
    ca, cb = np.zeros(N_MODES, complex), np.zeros(N_MODES, complex)
    for k in range(N_MODES):
        A = np.stack([cur[..., k].ravel(), prev[..., k].ravel()], 1)
        sol, *_ = np.linalg.lstsq(A, nxt[..., k].ravel(), rcond=None)
        ca[k], cb[k] = sol
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "spectral.npz", a=ca, b=cb, grid=np.int64(u.shape[-1]))


if __name__ == "__main__":
    main()

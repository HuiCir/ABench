"""Reference submission (training): implicit alternating least squares (Hu, Koren & Volinsky 2008) with the
frequency-scaled regularisation of Rendle et al. (2022, "Revisiting the performance of iALS on item recommendation
benchmarks"): for user u, minimise Σ_{i ∈ I_u} (1 - u·v_i)² + α0 Σ_{all i} (u·v_i)² + λ (|I_u| + α0 |I|)^ν ||u||²,
and symmetrically for items. Solved exactly per row with the Gram trick (V^T V is shared).

  python3 train.py --data $ABENCH_DATA_DIR --out checkpoint            # final model on all public interactions
  python3 train.py --data $ABENCH_DATA_DIR --out ckpt_dev --dev        # model for the self-check's dev split
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DIM, ALPHA0, LAM, NU, ITERS = 128, 0.003, 0.01, 1.0, 12


def solve_side(ip: np.ndarray, it: np.ndarray, H: np.ndarray, n_other: int, chunk: int = 2048) -> np.ndarray:
    """Rows W minimising Σ_{j in row} (1 - w·h_j)² + α0 Σ_all (w·h)² + λ (n_row + α0 n_other)^ν ||w||²."""
    k = H.shape[1]
    G = ALPHA0 * (H.T @ H)
    n = len(ip) - 1
    W = np.zeros((n, k))
    lens = np.diff(ip)
    for s in range(0, n, chunk):
        rows = range(s, min(n, s + chunk))
        A = np.repeat(G[None], len(rows), 0)
        b = np.zeros((len(rows), k))
        for j, r in enumerate(rows):
            Hr = H[it[ip[r]:ip[r + 1]]]
            A[j] += Hr.T @ Hr
            b[j] = Hr.sum(0)
            A[j][np.diag_indices(k)] += LAM * (lens[r] + ALPHA0 * n_other) ** NU
        W[s:s + len(rows)] = np.linalg.solve(A, b[..., None])[..., 0]
    return W


def transpose(ip: np.ndarray, it: np.ndarray, n_cols: int):
    rows = np.repeat(np.arange(len(ip) - 1), np.diff(ip))
    order = np.lexsort((rows, it))
    cip = np.concatenate([[0], np.cumsum(np.bincount(it, minlength=n_cols))]).astype(np.int64)
    return cip, rows[order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", action="store_true", help="train on the self-check's 80% dev split")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    if a.dev:
        import os
        import tempfile
        sys.path.insert(0, str(Path(os.environ.get("ABENCH_VIEW_DIR", ".")) / "interface"))
        from stages import make_dev
        data = make_dev(data, Path(tempfile.mkdtemp()))["public"]
    with np.load(data / "train.npz", allow_pickle=False) as z:
        ip, it, n_items = z["indptr"].astype(np.int64), z["items"].astype(np.int64), int(z["n_items"])
    n_users = len(ip) - 1
    cip, cit = transpose(ip, it, n_items)
    rng = np.random.default_rng(0)
    dim, iters = (16, 2) if a.quick else (DIM, ITERS)
    U = rng.normal(0, 0.1 / np.sqrt(dim), (n_users, dim))
    V = rng.normal(0, 0.1 / np.sqrt(dim), (n_items, dim))
    for t in range(iters):
        U = solve_side(ip, it, V, n_items)
        V = solve_side(cip, cit, U, n_users)
        print(f"iteration {t + 1}/{iters}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "factors.npz", U=U.astype(np.float32), V=V.astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())

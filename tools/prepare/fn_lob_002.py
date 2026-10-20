"""FN-LOB-002: FI-2010 (Ntakaris et al. 2018, CC BY 4.0), the setup used by DeepLOB.

Training: Train_Dst_NoAuction_ZScore_CF_7.txt (days 1-7). Held-out: Test_Dst_NoAuction_ZScore_CF_7/8/9.txt (days 8-10).
Each file is a 149 × E text matrix: rows 0-39 are the 10-level order book (ask price, ask volume, bid price, bid volume
per level, z-score normalised), rows 144-148 the labels for horizons k = 1, 2, 3, 5, 10 (1 up, 2 stationary, 3 down).
The archive is obtained from the Fairdata/Etsin record (manual download, accept the CC BY 4.0 terms); the files may sit
anywhere under --raw-dir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

SOURCES = [{"name": "FI-2010 BenchmarkDatasets (NoAuction, ZScore)",
            "url": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
            "urn": "urn:nbn:fi:csc-kata20170601153214969115", "license": "CC BY 4.0",
            "files": ["Train_Dst_NoAuction_ZScore_CF_7.txt", "Test_Dst_NoAuction_ZScore_CF_7.txt",
                      "Test_Dst_NoAuction_ZScore_CF_8.txt", "Test_Dst_NoAuction_ZScore_CF_9.txt"]}]
WINDOW, N_QUERY, QUERY_SEED = 100, 15000, 2010


def _find(raw: Path, name: str) -> Path:
    hits = sorted(raw.rglob(name))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {name} under {raw}, found {len(hits)}")
    return hits[0]


def _load(p: Path):
    m = np.loadtxt(p, dtype=np.float64)
    assert m.shape[0] == 149, m.shape
    return m[:40].T.astype(np.float32), (m[-5:].T - 1).astype(np.int8)    # labels 0 up, 1 stationary, 2 down


def _freeze(tr_ev, tr_lab, te_parts, sums) -> Frozen:
    ev = np.concatenate([e for e, _ in te_parts])
    lab = np.concatenate([l for _, l in te_parts])
    seg = np.concatenate([np.full(len(e), i, np.int64) for i, (e, _) in enumerate(te_parts)])
    starts = np.concatenate([[0], np.cumsum([len(e) for e, _ in te_parts])[:-1]])
    valid = np.concatenate([np.arange(s + WINDOW - 1, s + len(e)) for s, (e, _) in zip(starts, te_parts)])
    rng = np.random.default_rng(QUERY_SEED)
    q = np.sort(rng.choice(valid, size=min(N_QUERY, len(valid)), replace=False)).astype(np.int64)
    return Frozen(public={"train.npz": {"events": tr_ev, "labels": tr_lab}},
                  hidden={"events": ev, "query_idx": q, "y": lab[q, 4].astype(np.int64), "segment": seg[q]},
                  sources_sha256=sums, notes={"label_map": "0 up, 1 stationary, 2 down", "horizon_column": 4})


def build(raw: Path) -> Frozen:
    names = SOURCES[0]["files"]
    paths = {n: _find(raw, n) for n in names}
    sums = {n: check(p) for n, p in paths.items()}
    tr_ev, tr_lab = _load(paths[names[0]])
    return _freeze(tr_ev, tr_lab, [_load(paths[n]) for n in names[1:]], sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)

    def day(m):
        mid = np.cumsum(rng.normal(0, 1, m))
        ev = rng.normal(0, 1, (m, 40)).astype(np.float32)
        ev[:, 0::4] = mid[:, None] + 0.5
        ev[:, 2::4] = mid[:, None] - 0.5
        ev[:, 1] += 2 * np.tanh(np.diff(np.concatenate([mid, mid[-1:]]))).astype(np.float32)
        fut = np.array([mid[i + 1:i + 11].mean() - mid[i] if i + 11 <= m else 0 for i in range(m)])
        y = np.where(fut > 0.8, 0, np.where(fut < -0.8, 2, 1)).astype(np.int8)
        return ev, np.repeat(y[:, None], 5, 1)
    tr = day(n)
    return _freeze(tr[0], tr[1], [day(n // 6) for _ in range(3)], {"synthetic": f"seed0-n{n}"})

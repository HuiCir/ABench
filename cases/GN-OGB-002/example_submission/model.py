"""GN-OGB-002 reference submission, inference side: simplified graph convolution features with a ridge classifier.

Features are [X, ÂX, Â²X, 1] with Â the symmetric-normalised adjacency of the undirected graph with self-loops
(SGC-style propagation), computed from the graph passed to predict(); the class weights W come from the checkpoint
(train.py solves a one-vs-rest ridge regression on the train nodes in closed form).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

HOPS = 2


def adjacency(edge_index: np.ndarray, n: int) -> sparse.csr_matrix:
    r, c = edge_index
    a = sparse.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32) + sparse.identity(n, dtype=np.float32, format="csr")
    d = np.asarray(a.sum(1)).ravel()
    dinv = sparse.diags(1.0 / np.sqrt(d))
    return (dinv @ a @ dinv).tocsr()


def features(graph: dict) -> np.ndarray:
    x = graph["x"].astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    a = adjacency(graph["edge_index"], len(x))
    feats, h = [x], x
    for _ in range(HOPS):
        h = a @ h
        feats.append(h)
    f = np.concatenate(feats, 1)
    return np.concatenate([f, np.ones((len(f), 1), np.float32)], 1)


def load(checkpoint_dir, device="cpu"):
    with np.load(Path(checkpoint_dir) / "ridge.npz", allow_pickle=False) as z:
        return {"W": z["W"]}


def predict(model, graph, nodes):
    return features(graph)[nodes] @ model["W"]

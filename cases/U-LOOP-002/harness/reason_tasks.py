"""U-LOOP-002 tasks and training sampler (shared by the evaluator, the self-check and your code).

Three synthetic reasoning tasks whose difficulty d sets how many sequential steps a solution needs. Every problem is
a sequence of int tokens; every answer is one class.

  pointer   32 pointers p[0..31] (values 0..31) and a start s; answer p^d(s), the index reached after d hops
            input: [p_0, ..., p_31, s] (33 tokens, vocabulary 32)                     d in 1..16,  32 classes
  reach     a directed graph on 16 nodes as 24 edges (u, v) and a query (a, b); answer 1 if b is reachable from a.
            Positive problems have shortest distance d; a negative one is the same kind of graph with one edge of
            the a-to-b chain reversed, so that b is unreachable although a has out-edges and b has in-edges
            input: [u_1, v_1, ..., u_24, v_24, a, b] (50 tokens, vocabulary 16)       d in 1..8,   2 classes
  arith     a fully bracketed expression over digits with + - * mod 10, nesting depth d; answer its value
            input: tokens 0-9 digits, 10 '+', 11 '-', 12 '*', 13 '(', 14 ')', padded with 15 to 128 tokens
                                                                                       d in 1..6,   10 classes
The held-out problems have the same difficulty ranges, uniformly mixed.
"""
from __future__ import annotations

import numpy as np

TASKS = ("pointer", "reach", "arith")
MAX_D = {"pointer": 16, "reach": 8, "arith": 6}
N_CLASSES = {"pointer": 32, "reach": 2, "arith": 10}
SEQ_LEN = {"pointer": 33, "reach": 50, "arith": 128}
VOCAB = {"pointer": 32, "reach": 16, "arith": 16}
N_NODES, N_EDGES, N_PTR, PAD = 16, 24, 32, 15


def _pointer(d: int, rng: np.random.Generator):
    while True:
        p = rng.integers(0, N_PTR, N_PTR)
        s = int(rng.integers(N_PTR))
        seen, cur = [s], s
        for _ in range(d):
            cur = int(p[cur])
            seen.append(cur)
        if len(set(seen)) == d + 1:                        # no cycle within d hops: every hop matters
            return np.concatenate([p, [s]]), cur


def _bfs(adj: np.ndarray, a: int) -> np.ndarray:
    dist = np.full(N_NODES, -1)
    dist[a] = 0
    frontier = [a]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(adj[u]):
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(int(v))
        frontier = nxt
    return dist


def _reach(d: int, rng: np.random.Generator, positive: bool):
    while True:
        edges = set()
        perm = rng.permutation(N_NODES)
        for i in range(d):                                 # a chain of length d from perm[0] to perm[d]
            edges.add((int(perm[i]), int(perm[i + 1])))
        while len(edges) < N_EDGES:
            u, v = (int(x) for x in rng.integers(0, N_NODES, 2))
            if u != v:
                edges.add((u, v))
        E = np.array(sorted(edges, key=lambda e: rng.random()), np.int64)
        adj = np.zeros((N_NODES, N_NODES), bool)
        adj[E[:, 0], E[:, 1]] = True
        a = int(perm[0])
        dist = _bfs(adj, a)
        if positive:
            b = int(perm[d])
            if dist[b] != d:
                continue
            return np.concatenate([E.reshape(-1), [a, b]]), 1
        # negative: b unreachable from a, but reachable within d hops once one edge on the chain is reversed
        k = int(rng.integers(d))
        adj2 = adj.copy()
        u, v = int(perm[k]), int(perm[k + 1])
        adj2[u, v], adj2[v, u] = False, True
        E2 = np.argwhere(adj2)
        if len(E2) != N_EDGES:
            continue
        d2 = _bfs(adj2, a)
        b = int(perm[d])
        if d2[b] >= 0 or not adj2[:, b].any() or not adj2[a].any():
            continue                                       # b must be unreachable yet have in-edges; a out-edges
        E2 = E2[rng.permutation(len(E2))]
        return np.concatenate([E2.reshape(-1), [a, b]]), 0


def _expr(d: int, rng: np.random.Generator):
    if d == 0:
        v = int(rng.integers(10))
        return [v], v
    deeper = int(rng.integers(2))
    left, lv = _expr(d - 1 if deeper == 0 else int(rng.integers(d)), rng)
    right, rv = _expr(d - 1 if deeper == 1 else int(rng.integers(d)), rng)
    op = int(rng.integers(3))
    val = {0: lv + rv, 1: lv - rv, 2: lv * rv}[op] % 10
    return [13] + left + [10 + op] + right + [14], val


def _arith(d: int, rng: np.random.Generator):
    while True:
        toks, v = _expr(d, rng)
        if len(toks) <= SEQ_LEN["arith"]:
            return np.array(toks + [PAD] * (SEQ_LEN["arith"] - len(toks)), np.int64), v


def make(task: str, d: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    first = bool(rng.integers(2))                        # reach: positives and negatives alternate from a random start
    for i in range(count):
        if task == "pointer":
            x, y = _pointer(d, rng)
        elif task == "reach":
            x, y = _reach(d, rng, positive=(i % 2 == 0) == first)
        elif task == "arith":
            x, y = _arith(d, rng)
        else:
            raise ValueError(task)
        xs.append(np.asarray(x, np.int64))
        ys.append(int(y))
    return np.stack(xs), np.asarray(ys, np.int64)


def make_mixed(task: str, count: int, rng: np.random.Generator):
    """count problems with difficulties drawn uniformly from 1..MAX_D[task]; returns (x, y, d)."""
    ds = rng.integers(1, MAX_D[task] + 1, count)
    xs, ys = [], []
    for d in ds:
        x, y = make(task, int(d), 1, rng)
        xs.append(x[0]); ys.append(int(y[0]))
    return np.stack(xs), np.asarray(ys, np.int64), ds.astype(np.int64)


class Sampler:
    """Training data: sample(task, d, count) -> (x (count, SEQ_LEN[task]), y (count,)) for 1 <= d <= MAX_D[task]."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def sample(self, task: str, d: int, count: int):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        if not 1 <= int(d) <= MAX_D[task]:
            raise ValueError(f"difficulty of {task} is limited to 1..{MAX_D[task]}")
        return make(task, int(d), int(count), self._rng)

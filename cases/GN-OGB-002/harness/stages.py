"""GN-OGB-002 pipeline: node classification on ogbn-arxiv (evaluator and self-check use this same file).

Transductive setting: the whole graph, all features and the labels of the train and valid nodes are public; the labels
of the test nodes (papers from 2019 on) are held out by the evaluator.
The evaluator does not train. One stage, "predict":
  setup   load graph.npz (untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                          (load time, killed at 120 s)
  finish  predict(model, graph, nodes) -> (len(nodes), 40) scores for the test nodes        (≤ 120 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

N_CLASSES = 40
PARAMS = {"n_classes": N_CLASSES}
HELD_OUT_KEYS: list[str] = []                 # test node ids are public (transductive); only their labels are held out
PIPELINE = [{"stage": "predict", "timeout": 120 + 120, "finish_cap": 120, "tail_slack": 5, "threads": 8,
             "held_out": False, "models": False}]
KEYS = ("x", "edge_index", "year", "y", "train_idx", "valid_idx", "test_idx")


def load_graph(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in KEYS}


def _setup(ctx: Context):
    return load_graph(ctx.public / "graph.npz")


def _timed(ctx: Context, g):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device), g


def _finish(ctx: Context, state):
    mod, model, g = state
    graph = {k: v.copy() for k, v in g.items()}
    return {"scores": np.asarray(mod.predict(model, graph, g["test_idx"].copy()), dtype=np.float32)}


STAGES = {"predict": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the public valid nodes (2018) play the role of test nodes; their labels are hidden.

    For an unbiased dev estimate, train your checkpoint without the valid labels (the reference train.py does this
    unless --include-dev is given)."""
    g = load_graph(public / "graph.npz")
    y = g["y"].copy()
    dev = g["valid_idx"]
    target = y[dev].copy()
    y[dev] = -1
    rng = np.random.default_rng(0)
    tr = g["train_idx"]
    new_valid = np.sort(rng.choice(tr, size=max(1, len(tr) // 10), replace=False))
    new_train = np.setdiff1d(tr, new_valid)
    (out / "public").mkdir(parents=True, exist_ok=True)
    np.savez(out / "public" / "graph.npz", x=g["x"], edge_index=g["edge_index"], year=g["year"], y=y,
             train_idx=new_train, valid_idx=new_valid, test_idx=dev)
    return {"public": out / "public", "held_out": None, "reference": {"y_test": target}, "time_scale": 1.0}

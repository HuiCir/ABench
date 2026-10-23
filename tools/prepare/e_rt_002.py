"""E-RT-002: LIBERO-Spatial demonstrations and held-out initial states.

Public: the 10 LIBERO-Spatial demonstration files (50 human demonstrations per task; Liu et al. 2023, released on
Hugging Face as yifengzhu-hf/LIBERO-datasets @f13aa24a, Apache-2.0), verbatim. The simulator, task definitions and
published initial states come with the LIBERO package (github.com/Lifelong-Robot-Learning/LIBERO @8f1084e3, MIT),
installed in the sandbox and the judge environment.

Hidden: 10 new initial states per task, drawn from each task's own placement distribution with the secret seed
ABENCH_E_RT_SEED and kept only if no object moves by more than 1 cm while settling (libero_eval.sample_init_states).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check

SECRET_SEED_ENV = "ABENCH_E_RT_SEED"
DATA_REV, LIBERO_COMMIT = "f13aa24a3da8c43c7225569f28c562979fa0e35a", "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SPATIAL = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5": "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5": "75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5": "7ff1ee60447875c152d8c7819f499d23dbc0eb4d4bf045225d5d87a2cdad99bd",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5": "9d0b5435b313a8c0d7336b429b5f0f578be6f2a9f5f8a6dd67459ef388f3c74c",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5": "3961cb7126eaca47d0f5be29eb297bb62e7591e3c86172c9587c0bf670112f84",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5": "bf1090dd4826eacb49eb052f17d04798e2a1eb2f74f8d024bc3b9c5d234c25f9",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5": "0c053db7317e584632549bddb963bee8a237661b4bd29f0581a245f715e8eb72",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5": "d9b2da7399a88bea39024c8ef97713a20734e9b765e14b3399e2755890c828b7",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5": "1fcb3990cf9fd6185f56531cc9f6dec0a77a59bfa0c7f39dff95b7e43cf6aeaa",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5": "4d503d447028f437211880bea4ee1a7f6732113ea8560181fcca163251864794"}
SOURCES = [{"name": f"libero_spatial/{f}", "sha256": s, "license": "Apache-2.0",
            "url": f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{DATA_REV}/libero_spatial/{f}",
            "note": f"RAW/libero_spatial/{f}"} for f, s in SPATIAL.items()]
N_PER_TASK = 10


def libero_eval():
    p = ROOT / "tools" / "harness_kit" / "libero" / "libero_eval.py"
    spec = importlib.util.spec_from_file_location("libero_eval", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["libero_eval"] = m
    spec.loader.exec_module(m)
    return m


def held_out(suite: str, n_tasks: int, n: int, seed: int) -> dict:
    LE = libero_eval()
    states, tasks = [], []
    for tid in range(n_tasks):
        s = LE.sample_init_states(suite, tid, n, seed * 1000 + tid)
        states += list(s)
        tasks += [tid] * len(s)
    a, lens = LE.pack_states(states)
    return {"init_states": a, "init_len": lens, "task_id": np.asarray(tasks, np.int64)}


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (held-out initial states)")
    sums = {f"libero_spatial/{f}": check(raw / "libero_spatial" / f, sha256=s) for f, s in SPATIAL.items()}
    return Frozen(public={}, hidden=held_out("libero_spatial", 10, N_PER_TASK, seed), sources_sha256=sums,
                  raw_files={f"libero_spatial/{f}": raw / "libero_spatial" / f for f in SPATIAL},
                  notes={"libero_commit": LIBERO_COMMIT})


def synthetic(n: int) -> Frozen:
    """Random flattened states (plumbing only; they are not valid LIBERO states)."""
    rng = np.random.default_rng(0)
    k = max(1, n // 10)
    return Frozen(public={}, hidden={"init_states": rng.normal(size=(10 * k, 79)), "init_len": np.full(10 * k, 79),
                                     "task_id": np.repeat(np.arange(10), k)}, sources_sha256={"synthetic": f"n{n}"})

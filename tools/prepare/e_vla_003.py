"""E-VLA-003: LIBERO-Spatial and LIBERO-Object demonstrations, the frozen VLM and held-out initial states.

Public: the 20 demonstration files (50 per task; yifengzhu-hf/LIBERO-datasets @f13aa24a, Apache-2.0), verbatim.
Models: SmolVLM2-500M-Video-Instruct (HuggingFaceTB @7b375e1b, Apache-2.0), the frozen backbone the evaluator loads,
and smolvla_base (lerobot @d9f33c94, Apache-2.0), SmolVLA's pretrained weights as an optional starting point.
Hidden: 10 new initial states per task of both suites (secret seed ABENCH_E_VLA_SEED; e_rt_002.held_out).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prepare import e_rt_002 as RT
from prepare.common import Frozen, check

SECRET_SEED_ENV = "ABENCH_E_VLA_SEED"
VLM, VLM_REV = "SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SMOLVLA_REV = "d9f33c94a60fb382c90dea2164c96845bd955e28"
OBJECT = {
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5": "42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5": "90a722f2243a72878baff494d07ce7f4206cfc10307a4c82dcda5f2a5fabc060",
    "pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5": "07f48f1c0c06b5e10bc219e63701f0240d8205f85ccd0daab3ff9bd27bc35fb9",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5": "b5b57f36cd78e598e258e0e2a1a9d4e8d37d387901487eecd22d503e0e222302",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5": "7ae50ed3a64bab8418fd6c8e346ba1c39da0fdaa62916a87b9d20f3745b45406",
    "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5": "e0814effbe3a46777952950be0565577c348dad211327bc5d2a16f4eacc55ba1",
    "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5": "e56d954928e613f9db58085b7677ccf9c4824f1ab982265f71405d5bb14cfb4a",
    "pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5": "53e40490c62e94c678233eefbfd874a61c93d92c8088f59d426ae6c3fd90d288",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5": "45960606a85b9c68980eb8130e0fe16bf3621539cf5d4147f35aa8139573c5ac",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5": "43c52bdaa78b4c6aff3afb54d932d6a0e065f5c30c8e30a6c17f78749d487ea9"}
FILES = {**{f"libero_spatial/{f}": s for f, s in RT.SPATIAL.items()}, **{f"libero_object/{f}": s for f, s in OBJECT.items()}}
VLM_FILES = ["added_tokens.json", "chat_template.json", "config.json", "generation_config.json", "merges.txt",
             "model.safetensors", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.json"]
SOURCES = [{"name": n, "sha256": s, "license": "Apache-2.0", "note": f"RAW/{n}",
            "url": f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{RT.DATA_REV}/{n}"}
           for n, s in FILES.items()] + [
          {"name": f"{VLM}/", "url": f"https://huggingface.co/HuggingFaceTB/{VLM}/tree/{VLM_REV}", "files": VLM_FILES,
           "license": "Apache-2.0", "note": f"RAW/{VLM}/"},
          {"name": "smolvla_base/", "url": f"https://huggingface.co/lerobot/smolvla_base/tree/{SMOLVLA_REV}",
           "files": ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                     "policy_preprocessor_step_5_normalizer_processor.safetensors",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors"],
           "license": "Apache-2.0", "note": "RAW/smolvla_base/"}]
N_PER_TASK = 10


def held_out(seed: int) -> dict:
    parts = [RT.held_out(s, 10, N_PER_TASK, seed + 7919 * i) for i, s in enumerate(RT.libero_eval().SUITES)]
    LE = RT.libero_eval()
    states = [p["init_states"][i, :p["init_len"][i]] for p in parts for i in range(len(p["init_len"]))]
    a, lens = LE.pack_states(states)
    return {"init_states": a, "init_len": lens, "task_id": np.concatenate([p["task_id"] for p in parts]),
            "suite_id": np.concatenate([np.full(len(p["task_id"]), i, np.int64) for i, p in enumerate(parts)])}


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (held-out initial states)")
    sums = {n: check(raw / n, sha256=s) for n, s in FILES.items()}
    return Frozen(public={}, hidden=held_out(seed), sources_sha256=sums, raw_files={n: raw / n for n in FILES},
                  models={VLM: raw / VLM, "smolvla_base": raw / "smolvla_base"}, notes={"libero_commit": RT.LIBERO_COMMIT})


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    k = max(1, n // 20)
    return Frozen(public={}, hidden={"init_states": rng.normal(size=(20 * k, 79)), "init_len": np.full(20 * k, 79),
                                     "task_id": np.tile(np.repeat(np.arange(10), k), 2),
                                     "suite_id": np.repeat([0, 1], 10 * k)}, sources_sha256={"synthetic": f"n{n}"})

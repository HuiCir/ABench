"""Reference submission (inference): a recurrent-depth transformer with a confidence-and-ledger halting rule.

Architecture (src/looped.py): per-task token and position embeddings plus a learned answer token; one shared pre-norm
transformer layer (width 256, 4 heads, MLP 1,024) applied repeatedly, with the embedded input re-injected at every
iteration through a linear adapter on [h, e] (as in Geiping et al. 2025); per-task linear heads read the answer token.

Halting (readout): a problem stops when its softmax confidence reaches TAU. A ledger keeps the batch within the
budget: before each further iteration the remaining budget is compared with the cost of iterating every active
problem once more (step and readout, measured once at load time), and if it does not cover them, the most confident
active problems are stopped first. Only active problems are processed by step() and readout().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import reason_tasks as T
from src.looped import LoopedModel, measure_costs

TAU = 0.95


def load(checkpoint_dir: str, device: str):
    cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    net = LoopedModel(**cfg["model"]).to(device)
    net.load_state_dict(torch.load(Path(checkpoint_dir) / "model.pt", map_location=device, weights_only=True))
    net.eval()
    cost = {t: measure_costs(net, t, T.SEQ_LEN[t], device) for t in T.TASKS}   # outside the counted calls
    return {"net": net, "device": device, "cost": cost, "tau": float(cfg.get("tau", TAU))}


def n_params(model) -> float:
    return float(sum(p.numel() for p in model["net"].parameters()))


@torch.no_grad()
def init(model, task: str, x: np.ndarray, budget_flops: float):
    net, dev = model["net"], model["device"]
    e = net.embed(task, torch.as_tensor(x, device=dev))
    n = len(x)
    step_c, read_c = model["cost"][task]
    return {"task": task, "e": e, "h": e.clone(), "n": n, "spent": 0.0, "budget": budget_flops * n * 0.98,
            "unit": step_c + read_c, "active": np.ones(n, bool),
            "logits": torch.zeros(n, T.N_CLASSES[task], device=dev)}


@torch.no_grad()
def step(model, state, active: np.ndarray):
    idx = torch.as_tensor(np.flatnonzero(active), device=state["h"].device)
    if len(idx):
        state["h"][idx] = model["net"].core(state["h"][idx], state["e"][idx])
        state["spent"] += float(len(idx)) * state["unit"]
    state["active"] = active.copy()
    return state


@torch.no_grad()
def readout(model, state):
    act = state["active"]
    ids = np.flatnonzero(act)
    if len(ids):
        it = torch.as_tensor(ids, device=state["h"].device)
        state["logits"][it] = model["net"].head(state["task"], state["h"][it]).float()
    p = torch.softmax(state["logits"], -1).max(-1).values.cpu().numpy()
    act = act & (p < model["tau"])
    afford = int(max(0.0, state["budget"] - state["spent"]) // state["unit"])
    if act.sum() > afford:                                  # ledger: the least confident keep iterating
        keep = np.flatnonzero(act)
        keep = keep[np.argsort(p[keep])[:afford]]
        act = np.zeros_like(act)
        act[keep] = True
    return state["logits"], ~act

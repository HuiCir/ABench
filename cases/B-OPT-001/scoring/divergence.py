"""Method divergence D = behavioral divergence × 0.5 + code divergence × 0.15 + mechanism divergence × 0.35.

Behavioral divergence: run the submitted rule and each reference rule on a fixed (open-loop) gradient stream and compare the direction of each step's update (|cos|)
and the spectral distribution (total-variation similarity of normalized singular values). Every rule sees exactly the same gradient sequence.
Two views:
  - global: 1 - the maximum mean similarity to any "single" reference method. Used for the threshold (renamed/retuned clone detection)
  - patch : 1 - mean similarity after taking the nearest reference per tensor (splice detection: different layers copying different methods)
  D_beh = 0.5·global + 0.5·patch enters the score; the threshold uses only global, calibrated by the "hyperparameter-perturbation noise floor":
  double each reference method's momentum coefficient (1-β) and halve its weight decay, compare with the original, and take the largest behavioral difference as the noise floor;
  threshold = max(configured lower bound, 1.5 × noise floor). Below it, the difference from some reference method is no larger than "changing a hyperparameter".
Only 2D hidden-layer matrices are compared; for 1D parameters and embeddings every method falls back to AdamW, which carries no distinguishing information.
"""
from __future__ import annotations

import ast
import inspect
import statistics

import numpy as np

from optim_api import Ops


# ---------------------------------------------------------------- gradient stream
def synthetic_gradient_stream(seed: int = 0, steps: int = 64):
    """Stand-in during construction: in the formal benchmark the gradient stream is recorded from the 10%/50%/90% checkpoints of the AdamW reference run.

    Here "a slowly rotating low-rank signal + noise" simulates three training stages: later stages have a higher noise share and a lower signal rank.
    """
    rng = np.random.default_rng(seed)
    specs = {                      # name: (shape, signal rank)
        "layers.4.self_attn.q_proj": ((96, 96), 8),
        "layers.4.mlp.up_proj": ((160, 96), 24),
        "layers.20.self_attn.o_proj": ((96, 96), 2),
        "layers.20.mlp.down_proj": ((96, 160), 1),
    }
    phases = {"p10": 0.3, "p50": 0.8, "p90": 1.6}          # noise/signal ratio
    streams = {}
    for ph, nsr in phases.items():
        streams[ph] = {}
        for name, ((m, n), k) in specs.items():
            U = np.linalg.qr(rng.standard_normal((m, k)))[0]
            V = np.linalg.qr(rng.standard_normal((n, k)))[0]
            s = np.geomspace(1.0, 0.05, k)
            drift = rng.standard_normal((n, k)) * 0.02
            seq = []
            for t in range(steps):
                Vt = np.linalg.qr(V + t * drift)[0]
                sig = (U * s) @ Vt.T
                noise = rng.standard_normal((m, n)) * nsr * np.linalg.norm(sig) / np.sqrt(m * n)
                seq.append((sig + noise).astype(np.float64))
            streams[ph][name] = seq
    return streams


def run_rule_on_stream(rule, streams, warmup: int = 48, lr: float = 1e-3, seed: int = 0):
    ops = Ops("numpy")
    rng = np.random.default_rng(seed + 1)
    out = {}
    for ph, tensors in streams.items():
        out[ph] = {}
        for name, seq in tensors.items():
            p = rng.standard_normal(seq[0].shape) * 0.02
            st = rule.init_state(p, ops)
            deltas = []
            for t, g in enumerate(seq, start=1):
                d = rule.update(p, g, st, t, lr, ops)
                p = p + d
                if t > warmup:
                    deltas.append(np.asarray(d))
            out[ph][name] = deltas
    return out


def _cos(a, b):
    return float(abs((a * b).sum()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _spec_sim(a, b):
    sa = np.linalg.svd(a, compute_uv=False)
    sb = np.linalg.svd(b, compute_uv=False)
    sa, sb = sa / sa.sum(), sb / sb.sum()
    return 1.0 - 0.5 * float(np.abs(sa - sb).sum())


def pair_similarity(sub_d, ref_d):
    per_tensor = {}
    for ph in sub_d:
        for name in sub_d[ph]:
            vals = [0.6 * _cos(a, b) + 0.4 * _spec_sim(a, b) for a, b in zip(sub_d[ph][name], ref_d[ph][name])]
            per_tensor.setdefault(name, []).append(float(np.mean(vals)))
    return {k: float(np.mean(v)) for k, v in per_tensor.items()}


_MOMENTUM_KEYS = {"b1", "b2", "b3", "mu", "beta"}


def perturb(rule_cls):
    """Return a hyperparameter-perturbed version of rule_cls: momentum (1-β) ×2, weight decay ×0.5."""
    base = rule_cls()
    kw = {}
    for k, v in base.hp.items():
        if k in _MOMENTUM_KEYS:
            kw[k] = max(0.5, 1 - 2 * (1 - v))
        elif k == "wd":
            kw[k] = v * 0.5
    return rule_cls(**kw)


def hp_noise_floor(ref_classes: dict, seed: int = 0):
    streams = synthetic_gradient_stream(seed)
    floor = {}
    for rid, cls in ref_classes.items():
        a = run_rule_on_stream(cls(), streams, seed=seed)
        b = run_rule_on_stream(perturb(cls), streams, seed=seed)
        floor[rid] = round(1 - float(np.mean(list(pair_similarity(a, b).values()))), 4)
    return {"floor": max(floor.values()), "by_reference": floor}


def behavioral_divergence(sub_rule, ref_rules: dict, seed: int = 0):
    streams = synthetic_gradient_stream(seed)
    sub_d = run_rule_on_stream(sub_rule, streams, seed=seed)
    table = {}
    for rid, rule in ref_rules.items():
        table[rid] = pair_similarity(sub_d, run_rule_on_stream(rule, streams, seed=seed))
    mean_sim = {rid: float(np.mean(list(t.values()))) for rid, t in table.items()}
    nearest = max(mean_sim, key=mean_sim.get)
    tensors = list(next(iter(table.values())).keys())
    patch = {name: max(table, key=lambda r: table[r][name]) for name in tensors}
    patch_sim = float(np.mean([table[patch[n]][n] for n in tensors]))
    D_global, D_patch = 1 - mean_sim[nearest], 1 - patch_sim
    return {
        "D_beh": round(0.5 * D_global + 0.5 * D_patch, 4),
        "D_global": round(D_global, 4),
        "D_patch": round(D_patch, 4),
        "nearest_reference": nearest,
        "similarity_by_reference": {k: round(v, 4) for k, v in sorted(mean_sim.items(), key=lambda x: -x[1])},
        "nearest_by_tensor": {n: {"ref": patch[n], "sim": round(table[patch[n]][n], 4)} for n in tensors},
    }


# ---------------------------------------------------------------- code divergence
class _Seq(ast.NodeVisitor):
    def __init__(self):
        self.toks = []

    def generic_visit(self, node):
        t = type(node).__name__
        if isinstance(node, ast.Attribute):
            t += ":" + node.attr
        elif isinstance(node, ast.BinOp):
            t += ":" + type(node.op).__name__
        elif isinstance(node, ast.Constant):
            t = "Const"
        elif isinstance(node, (ast.Name, ast.arg, ast.Load, ast.Store, ast.Expr)):
            return super().generic_visit(node)  # variable names do not count, so renaming cannot evade
        self.toks.append(t)
        super().generic_visit(node)


def _ngrams(src: str, n: int = 5):
    v = _Seq()
    v.visit(ast.parse(src))
    toks = v.toks
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def code_divergence(sub_src: str, ref_classes: dict, n: int = 5):
    sg = _ngrams(sub_src, n)
    sims = {}
    for rid, cls in ref_classes.items():
        rg = _ngrams(inspect.getsource(cls), n)
        sims[rid] = len(sg & rg) / max(1, len(sg | rg))
    nearest = max(sims, key=sims.get)
    return {"D_code": round(1 - sims[nearest], 4), "nearest_reference": nearest,
            "jaccard_by_reference": {k: round(v, 4) for k, v in sorted(sims.items(), key=lambda x: -x[1])}}


# ---------------------------------------------------------------- mechanism divergence (judges)
def mechanism_divergence(judge: dict):
    levels = [j["novelty_level"] for j in judge["judges"]]
    med = statistics.median(levels)
    return {"D_mech": round(med / 3, 4), "levels": levels,
            "nearest_prior_art": judge.get("nearest_prior_art"), "notes": [j.get("note") for j in judge["judges"]]}


def combine(beh, code, mech, cfg, noise_floor: float):
    w = cfg["weights"]
    D = w["behavioral"] * beh["D_beh"] + w["code"] * code["D_code"] + w["mechanism"] * mech["D_mech"]
    g = cfg["gate"]
    thr = max(g["min_behavioral"], g["noise_floor_multiplier"] * noise_floor)
    checks = {
        "D>=%.2f" % g["min_total"]: D >= g["min_total"],
        "D_global>=%.3f (noise floor %.3f × %.1f)" % (thr, noise_floor, g["noise_floor_multiplier"]): beh["D_global"] >= thr,
    }
    return {"D": round(D, 4), "gate_pass": all(checks.values()), "gate_checks": checks,
            "behavioral_threshold": round(thr, 4)}

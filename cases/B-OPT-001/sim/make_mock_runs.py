"""Generate [SIMULATED] training results, to exercise the scoring pipeline without training.

All numbers are simulated and do not represent the real performance of any optimizer.
Model: final loss L(D) = E + A · (k · D)^(-alpha) + seed noise, where D is the number of tokens and k is the optimizer's
"effective data multiplier" in that setting (AdamW = 1). The relative sizes of the multipliers follow the qualitative conclusions of 2509.02046:
matrix optimizers lead more at small scale, and the advantage shrinks as scale grows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs" / "mock"

SETTINGS = {
    #  name (micro-scale tier)        token budget   E     A      alpha
    "L2_main":      (0.35e9, 3.10, 420.0, 0.30),   # μ2
    "L3a_scale":    (0.84e9, 2.85, 420.0, 0.30),   # μ3
    "L3b_long":     (1.12e9, 3.40, 420.0, 0.30),   # μ1 × 8× data
    "L3c_bigbatch": (0.35e9, 3.10, 440.0, 0.30),   # μ2 × 4× batch size
}

# effective data multiplier k (simulated values)
K = {
    "adamw":              {"L2_main": 1.00, "L3a_scale": 1.00, "L3b_long": 1.00, "L3c_bigbatch": 1.00},
    "adafactor":          {"L2_main": 0.96, "L3a_scale": 0.97, "L3b_long": 0.98, "L3c_bigbatch": 0.95},
    "lion":               {"L2_main": 0.97, "L3a_scale": 0.98, "L3b_long": 0.97, "L3c_bigbatch": 1.06},
    "sophia":             {"L2_main": 1.05, "L3a_scale": 1.03, "L3b_long": 1.02, "L3c_bigbatch": 1.04},
    "schedulefree_adamw": {"L2_main": 1.06, "L3a_scale": 1.04, "L3b_long": 1.03, "L3c_bigbatch": 1.05},
    "ademamix":           {"L2_main": 1.12, "L3a_scale": 1.09, "L3b_long": 1.13, "L3c_bigbatch": 1.08},
    "soap":               {"L2_main": 1.30, "L3a_scale": 1.20, "L3b_long": 1.19, "L3c_bigbatch": 1.26},
    "mars":               {"L2_main": 1.14, "L3a_scale": 1.10, "L3b_long": 1.08, "L3c_bigbatch": 1.12},
    "cautious_adamw":     {"L2_main": 1.07, "L3a_scale": 1.05, "L3b_long": 1.04, "L3c_bigbatch": 1.06},
    "muon":               {"L2_main": 1.33, "L3a_scale": 1.22, "L3b_long": 1.17, "L3c_bigbatch": 1.30},
}

# submissions under evaluation (simulated values): example submission + two control groups
CANDIDATES = {
    "example_submission":   {"k": {"L2_main": 1.43, "L3a_scale": 1.32, "L3b_long": 1.24, "L3c_bigbatch": 1.33}, "time": 1.07},
    "clone_muon":           {"k": {"L2_main": 1.34, "L3a_scale": 1.22, "L3b_long": 1.18, "L3c_bigbatch": 1.30}, "time": 1.04},
    "rankgate_linear_gate": {"k": {"L2_main": 0.99, "L3a_scale": 0.99, "L3b_long": 1.00, "L3c_bigbatch": 0.96}, "time": 1.06},
}

# ablation (simulated values): effective data multiplier and step-time ratio on L2_main after the module is replaced
ABLATIONS = {
    "example_submission": {
        "stable_rank_gate":      {"k": 1.345, "time": 1.06},   # gate fixed at 0.5: most of the gain disappears
        "power_iteration_sigma": {"k": 1.43,  "time": 1.30},   # exact SVD: token efficiency unchanged, steps slower
    },
    "rankgate_linear_gate": {
        "stable_rank_gate":      {"k": 1.20,  "time": 1.06},   # the linear gate is almost always 0, so replacing it by 0.5 is actually better
        "power_iteration_sigma": {"k": 0.99,  "time": 1.29},
    },
}

# step-time ratio (relative to AdamW, simulated values)
TIME = {"adamw": 1.00, "adafactor": 0.98, "lion": 0.95, "sophia": 1.06, "schedulefree_adamw": 1.01,
        "ademamix": 1.03, "soap": 1.15, "mars": 1.05, "cautious_adamw": 1.01, "muon": 1.04}

BUDGETS = [0.5, 0.7, 1.0, 1.4, 2.0]
SEED_NOISE = 0.004


def loss(E, A, alpha, k, D, rng):
    return float(E + A * (k * D) ** (-alpha) + rng.normal(0, SEED_NOISE))


def main(seed: int = 7):
    rng = np.random.default_rng(seed)
    ref_runs = {}
    for s, (T, E, A, alpha) in SETTINGS.items():
        n_seeds = 3 if s == "L2_main" else 2
        ref_runs[s] = {opt: {"budgets": BUDGETS,
                             "loss": [[loss(E, A, alpha, K[opt][s], b * T, rng) for _ in range(n_seeds)]
                                      for b in BUDGETS]} for opt in K}
    ood_ref = {"wikitext-103": 3.112, "the-stack-smol/python": 2.641}
    ood_reg = {"example_submission": [0.0019, 0.0034], "clone_muon": [0.0004, 0.0008],
               "rankgate_linear_gate": [0.0021, 0.0102]}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cand in CANDIDATES.items():
        runs = {"_note": "SIMULATED: simulated data, used only to exercise the scoring pipeline", "settings": {}, "ood": {}}
        for s, (T, E, A, alpha) in SETTINGS.items():
            n_seeds = 3 if s == "L2_main" else 2
            sub = {"loss": [loss(E, A, alpha, cand["k"][s], T, rng) for _ in range(n_seeds)],
                   "time_ratio": cand["time"]}
            runs["settings"][s] = {"token_budget": T, "references": ref_runs[s], "submission": sub,
                                   "time_ratios": dict(TIME)}
        T, E, A, alpha = SETTINGS["L2_main"]
        runs["ablations"] = {mod: {"loss": [loss(E, A, alpha, v["k"], T, rng) for _ in range(2)], "time_ratio": v["time"]}
                             for mod, v in ABLATIONS.get(name, {}).items()}
        runs["ood"] = {k: {"best_ref": v, "submission": round(v * (1 + r), 4)}
                       for (k, v), r in zip(ood_ref.items(), ood_reg[name])}
        f = OUT / f"runs_{name}.json"
        f.write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", f)


if __name__ == "__main__":
    main()

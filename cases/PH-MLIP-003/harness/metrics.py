"""PH-MLIP-003 metrics: energy and force mean absolute errors on the held-out configurations (meV and meV/Å; 1 kcal/mol
= 43.364 meV), per molecule and averaged over the five molecules; the primary metric is the mean force MAE. Stability:
the fraction of the 10-ps Langevin trajectories (4 per molecule) that stay stable for the whole run, and the mean
stable time. Inference time per configuration is reported."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

MOLECULES = ("aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene")
MEV = 43.3641
PRIMARY = {"metric": "mean_force_mae_mev", "goal": "min"}
GATES = [{"metric": "stable_fraction", "min": 0.75}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802  (bootstrap groups: each molecule's configurations)
    return [{"reference": [f"{m}_energy", f"{m}_forces"], "outputs": [["predict", f"{m}_energy"], ["predict", f"{m}_forces"]]}
            for m in MOLECULES]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, fm, em, stab, st_time = {}, [], [], [], []
    steps = float(need(outputs, "predict", "md_steps", shape=(1,))[0])
    for m in MOLECULES:
        e_ref, f_ref = np.asarray(reference[f"{m}_energy"]), np.asarray(reference[f"{m}_forces"])
        e = need(outputs, "predict", f"{m}_energy", shape=e_ref.shape)
        f = need(outputs, "predict", f"{m}_forces", shape=f_ref.shape)
        out[f"{m}_energy_mae_mev"] = float(np.abs(e - e_ref).mean() * MEV)
        out[f"{m}_force_mae_mev"] = float(np.abs(f - f_ref).mean() * MEV)
        s = need(outputs, "predict", f"{m}_stable_fs", shape=(4,))
        out[f"{m}_stable_fraction"] = float((s >= steps * 0.5 - 1e-9).mean())
        out[f"{m}_ms_per_config"] = float(need(outputs, "predict", f"{m}_ms_per_config", shape=(1,))[0])
        fm.append(out[f"{m}_force_mae_mev"]); em.append(out[f"{m}_energy_mae_mev"])
        stab.append(out[f"{m}_stable_fraction"]); st_time.append(float(s.mean()))
    out["mean_force_mae_mev"] = float(np.mean(fm))
    out["mean_energy_mae_mev"] = float(np.mean(em))
    out["stable_fraction"] = float(np.mean(stab))
    out["mean_stable_ps"] = float(np.mean(st_time) / 1000.0)
    out["load_seconds"] = float(seconds["predict"])
    return out

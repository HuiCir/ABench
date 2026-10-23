"""PH-SYM-005 metrics.

Per system: the returned coefficient matrix C_hat is compared with the true C over systems.library(d). A term is
present when its coefficient is non-zero (return exact zeros for absent terms).
  exact        1 if the set of present terms equals the true set in every equation
  term_f1      F1 of the present terms against the true terms (all equations together)
  coef_error   ||C_hat - C||_F / ||C||_F
  extrap_nrmse both models are integrated from 4 new initial conditions over the training duration; RMS difference
               divided by the standard deviation of the true trajectory, per variable, averaged, capped at 1
               (1 if the discovered model leaves 10x the range of the true trajectory or cannot be integrated)
Primary: recovery_rate = mean of `exact` over the systems. Ties on it are broken by mean term_f1, then by mean
extrap_nrmse."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401
import systems as S

PRIMARY = {"metric": "recovery_rate", "goal": "max"}
GATES = [{"metric": "discover_seconds", "max": 1200.0}]


def ids(reference: dict) -> list[str]:
    return sorted(k[: -len("_coef")] for k in reference if k.endswith("_coef"))


def extrapolation(C_hat: np.ndarray, C: np.ndarray, x0s: np.ndarray, duration: float) -> float:
    errs = []
    for x0 in x0s:
        tr = S.simulate(C, x0, duration, n=500)
        if tr is None:
            continue
        xt = tr[1]
        span = float(np.abs(xt).max())
        pr = S.simulate(C_hat, x0, duration, n=500, bound=10 * span + 1.0)
        if pr is None:
            errs.append(1.0)
            continue
        e = np.sqrt(((pr[1] - xt) ** 2).mean(axis=0)) / (xt.std(axis=0) + 1e-12)
        errs.append(float(min(1.0, e.mean())))
    return float(np.mean(errs)) if errs else 1.0


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, exact, f1s, ext, cerr = {}, [], [], [], []
    for s in ids(reference):
        C = np.asarray(reference[f"{s}_coef"], np.float64)
        Ch = need(outputs, "discover", f"coef_{s}", shape=C.shape)
        t, p = S.support(C), S.support(Ch)
        tp = float((t & p).sum())
        prec = tp / max(1.0, float(p.sum()))
        rec = tp / max(1.0, float(t.sum()))
        f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
        ex = float((t == p).all())
        ce = float(np.linalg.norm(Ch - C) / np.linalg.norm(C))
        xe = extrapolation(Ch, C, np.asarray(reference[f"{s}_test_x0"]), float(np.asarray(reference[f"{s}_duration"])[0]))
        out.update({f"exact_{s}": ex, f"term_f1_{s}": f1, f"coef_error_{s}": ce, f"extrap_nrmse_{s}": xe})
        exact.append(ex); f1s.append(f1); ext.append(xe); cerr.append(ce)
    out["recovery_rate"] = float(np.mean(exact))
    out["mean_term_f1"] = float(np.mean(f1s))
    out["mean_extrap_nrmse"] = float(np.mean(ext))
    out["median_coef_error"] = float(np.median(cerr))
    out["n_systems"] = float(len(exact))
    out["discover_seconds"] = float(seconds["discover"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"recovery_rate_delta_vs_stlsq": sub["recovery_rate"] - base["recovery_rate"],
            "mean_term_f1_delta_vs_stlsq": sub["mean_term_f1"] - base["mean_term_f1"],
            "mean_extrap_nrmse_delta_vs_stlsq": sub["mean_extrap_nrmse"] - base["mean_extrap_nrmse"],
            "stlsq_recovery_rate": base["recovery_rate"]}

"""Long-horizon multivariate forecasting (evaluator code, agent-visible; FN-LTSF-005): datasets, splits, forecast
windows and the closed-form linear reference that every submission is compared with.

Splits follow the common long-horizon benchmark protocol (Informer, Autoformer, DLinear, PatchTST):
  ETTh1, ETTh2   hourly, 7 channels: train 12 months, validation 4, test 4  (rows [0, 8640), [8640, 11520), [11520, 14400))
  ETTm1, ETTm2   15-minute, 7 channels: the same months                   ([0, 34560), [34560, 46080), [46080, 57600))
  electricity    hourly, 321 clients: train 70%, validation 10%, test 20% of the 26,304 rows
Values are standardised per channel with the mean and standard deviation of the training rows; errors are measured
on that scale.

Forecast windows. An origin o is a time step; the model sees the LOOKBACK_MAX = 720 steps before it, rows [o - 720, o),
and forecasts rows [o, o + H) for H in HORIZONS. Test origins run from the start of the test rows to the last origin
whose 720-step forecast ends inside them, every STRIDE[d] steps. The strides are chosen not to divide a day or a week,
so the origins move through the hours of the day and the days of the week. Every horizon uses the same origins.

Reference (ridge_fit / ridge_predict): one linear map per dataset, shared by all channels, from the last 336 steps
minus the last observed value to the next 720 steps minus that value, plus a bias; ridge penalty (per training row)
chosen on the validation rows from RIDGE_GRID after a fit on the training rows, then refitted on training and
validation rows together (all the data a submission has). Shorter horizons use the first H outputs (a multi-output
ridge solves each output column separately, so this is the same as fitting each horizon on its own). It is the
"linear model with last-value normalisation" family of DLinear and RLinear: a strong baseline.
"""
from __future__ import annotations

import numpy as np

DATASETS = ("etth1", "etth2", "ettm1", "ettm2", "electricity")
CHANNELS = {"etth1": 7, "etth2": 7, "ettm1": 7, "ettm2": 7, "electricity": 321}
HORIZONS = (96, 192, 336, 720)
LOOKBACK_MAX = 720
STRIDE = {"etth1": 23, "etth2": 23, "ettm1": 95, "ettm2": 95, "electricity": 167}
REF_LOOKBACK = 336
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # penalty per training row
MAX_ROWS = 300_000                       # training rows (window x channel) used by the reference fit


def borders(name: str, n: int) -> tuple[int, int, int]:
    """(end of train, end of validation, end of test) row indices."""
    if name.startswith("etth"):
        return 8640, 11520, 14400
    if name.startswith("ettm"):
        return 34560, 46080, 57600
    n_train, n_test = int(0.7 * n), int(0.2 * n)
    return n_train, n - n_test, n


def origins(lo: int, hi: int, stride: int, hmax: int = HORIZONS[-1]) -> np.ndarray:
    """Origins o in [lo, hi - hmax] every `stride` steps (the longest forecast must end by row hi)."""
    return np.arange(lo, hi - hmax + 1, stride, dtype=np.int64)


def windows(series: np.ndarray, orig: np.ndarray, lookback: int = LOOKBACK_MAX, horizon: int = HORIZONS[-1]):
    """History (B, lookback, C) and target (B, horizon, C) for each origin."""
    x = np.stack([series[o - lookback:o] for o in orig]).astype(np.float32)
    y = np.stack([series[o:o + horizon] for o in orig]).astype(np.float32)
    return x, y


def standardise(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = raw[:train_end].mean(axis=0)
    sd = raw[:train_end].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return ((raw - mu) / sd).astype(np.float32), mu.astype(np.float64), sd.astype(np.float64)


# --------------------------------------------------------------------------- the linear reference
def _design(series: np.ndarray, orig: np.ndarray, lookback: int, horizon: int):
    """Rows (window, channel): inputs = the last `lookback` values minus the last one, and a 1 for the bias;
    targets = the next `horizon` values minus the last one."""
    x, y = windows(series, orig, lookback, horizon)
    last = x[:, -1:, :]
    xi = np.transpose(x - last, (0, 2, 1)).reshape(-1, lookback)
    yi = np.transpose(y - last, (0, 2, 1)).reshape(-1, horizon)
    return np.hstack([xi, np.ones((len(xi), 1), np.float32)]).astype(np.float64), yi.astype(np.float64)


def _gram(series, orig, lookback, horizon, chunk=64):
    """X'X / n and X'Y / n over the design rows of the given origins."""
    a = np.zeros((lookback + 1, lookback + 1))
    b = np.zeros((lookback + 1, horizon))
    n = 0
    for s in range(0, len(orig), chunk):
        xi, yi = _design(series, orig[s:s + chunk], lookback, horizon)
        a += xi.T @ xi
        b += xi.T @ yi
        n += len(xi)
    return a / n, b / n


def _solve(a, b, lam):
    reg = lam * np.eye(len(a))
    reg[-1, -1] = 0.0                                   # the bias is not penalised
    return np.linalg.solve(a + reg, b)


def fit_origins(lo: int, hi: int, n_channels: int, lookback: int, horizon: int = HORIZONS[-1]) -> np.ndarray:
    """Training origins whose history starts at or after `lo` and whose target ends by `hi`, thinned so that windows x
    channels stays within MAX_ROWS."""
    o = np.arange(lo + lookback, hi - horizon + 1, dtype=np.int64)
    step = max(1, int(np.ceil(len(o) * n_channels / MAX_ROWS)))
    return o[::step]


def ridge_fit(series: np.ndarray, train_end: int, val_end: int, lookback: int = REF_LOOKBACK,
              grid=RIDGE_GRID, horizon: int = HORIZONS[-1], refit: bool = True) -> tuple[np.ndarray, float]:
    """Weights (lookback + 1, horizon): the penalty chosen by mean squared error on validation origins in
    [train_end, val_end) after fitting on rows [0, train_end); then (refit) fitted again on rows [0, val_end)."""
    c = series.shape[1]
    a, b = _gram(series, fit_origins(0, train_end, c, lookback, horizon), lookback, horizon)
    vo = origins(train_end, val_end, max(1, (val_end - train_end - horizon) // 64), horizon)
    xv, yv = windows(series, vo, lookback, horizon)
    best = (np.inf, None, None)
    for lam in grid:
        w = _solve(a, b, lam)
        err = float(np.mean((ridge_predict(w, xv, horizon) - yv) ** 2))
        if err < best[0]:
            best = (err, w, lam)
    if not refit:
        return best[1], best[2]
    a, b = _gram(series, fit_origins(0, val_end, c, lookback, horizon), lookback, horizon)
    return _solve(a, b, best[2]), best[2]


def ridge_predict(w: np.ndarray, history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast (B, horizon, C) from history (B, >= lookback, C) with weights from ridge_fit."""
    lookback = w.shape[0] - 1
    x = history[:, -lookback:, :].astype(np.float64)
    last = x[:, -1:, :]
    z = np.einsum("blc,lh->bhc", x - last, w[:-1, :horizon]) + w[-1, :horizon][None, :, None]
    return (z + last).astype(np.float32)


def window_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean squared error of each window over steps and channels: (B,)."""
    d = pred.astype(np.float64) - target.astype(np.float64)
    return np.mean(d * d, axis=(1, 2))


def window_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred.astype(np.float64) - target.astype(np.float64)), axis=(1, 2))

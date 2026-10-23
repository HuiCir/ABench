"""PH-WELL-002 systems, rollouts and measures (evaluator code, agent-visible).

Three 2D systems of The Well (Ohana et al. 2024; polymathic-ai on Hugging Face, CC-BY 4.0), resampled by block
averaging to 64 x 64 and stored as float16 (frames, channels, 64, 64):

  system                 channels (in this order)                                       params      frames  block
  turbulent_radiative    density, pressure, velocity_x, velocity_y                       tcool       101     2 x 6
  active_matter          concentration, velocity_x, velocity_y, D_xx, D_xy, D_yx, D_yy,   zeta, alpha 81      4 x 4
                         E_xx, E_xy, E_yx, E_yy
  gray_scott             A, B                                                            F, k        101     2 x 2
                                                                                                      (every 10th step)
Task: from HISTORY = 4 consecutive frames and the system's parameters, predict the next frame; the evaluator rolls the
prediction out autoregressively for ROLLOUT = 10 frames (each prediction is fed back as the newest frame).

VRMSE (as in The Well): for one field of one frame, sqrt(mean((pred - true)^2) / (var(true) + 1e-7)) over the 64 x 64
grid; averaged over fields. Rollout VRMSE: averaged over the 10 predicted frames. A prediction that is not finite
counts as VRMSE 10 for that frame (and the rollout continues from zeros).
"""
from __future__ import annotations

import numpy as np

SYSTEMS = {
    "turbulent_radiative": {"channels": ["density", "pressure", "velocity_x", "velocity_y"], "params": ["tcool"]},
    "active_matter": {"channels": ["concentration", "velocity_x", "velocity_y", "D_xx", "D_xy", "D_yx", "D_yy",
                                   "E_xx", "E_xy", "E_yx", "E_yy"], "params": ["zeta", "alpha"]},
    "gray_scott": {"channels": ["A", "B"], "params": ["F", "k"]},
}
HISTORY, ROLLOUT, RES, EPS, BAD = 4, 10, 64, 1e-7, 10.0


def vrmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred, true (..., C, H, W) -> (...,) VRMSE averaged over the C fields."""
    p, t = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    mse = ((p - t) ** 2).mean((-1, -2))
    var = t.var(axis=(-1, -2))
    v = np.sqrt(mse / (var + EPS))
    v = np.where(np.isfinite(v), v, BAD)
    return np.minimum(v, BAD).mean(-1)


def rollout(predict, system: str, history: np.ndarray, params: np.ndarray, steps: int = ROLLOUT,
            batch: int = 16) -> np.ndarray:
    """predict(system, history (B, 4, C, 64, 64) float32, params (B, P) float32) -> (B, C, 64, 64).
    Returns float16 predictions (N, steps, C, 64, 64)."""
    out = np.zeros((len(history), steps, *history.shape[2:]), np.float16)
    for b in range(0, len(history), batch):
        h = np.asarray(history[b:b + batch], np.float32).copy()
        p = np.asarray(params[b:b + batch], np.float32)
        for s in range(steps):
            y = np.asarray(predict(system, h, p), np.float32)
            if y.shape != h.shape[:1] + h.shape[2:]:
                raise ValueError(f"predict() returned {y.shape}, expected {h.shape[:1] + h.shape[2:]}")
            ok = np.isfinite(y).all(axis=(1, 2, 3))
            out[b:b + batch, s] = np.where(ok[:, None, None, None], np.clip(y, -6e4, 6e4), np.nan)
            y = np.where(np.isfinite(y), y, 0.0)
            h = np.concatenate([h[:, 1:], y[:, None]], 1)
    return out


def windows(traj: np.ndarray, starts) -> tuple:
    """traj (T, C, 64, 64) -> (inputs (n, HISTORY, ...), targets (n, ROLLOUT, ...)) for the given start frames."""
    x = np.stack([traj[s:s + HISTORY] for s in starts])
    y = np.stack([traj[s + HISTORY:s + HISTORY + ROLLOUT] for s in starts])
    return x, y


def block_mean(a: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Average over fy x fx blocks of the last two axes."""
    *lead, H, W = a.shape
    return a.reshape(*lead, H // fy, fy, W // fx, fx).mean((-3, -1))

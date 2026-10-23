"""Reference submission for V-3DG-005: the baseline's 3D Gaussian splatting with two changes aimed at sparse views:
a random background colour at every step, which stops semi-transparent Gaussians from standing in for the
background (as in the 3DGS reference code for synthetic scenes), and a penalty on the ratio of the largest to the
smallest scale of each Gaussian above 10 (in the spirit of the scale regularisation of PhysGaussian), which limits
needle-like Gaussians that fit the training views but break in new ones."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import gs3d  # noqa: E402

RATIO, WEIGHT = 10.0, 0.01


def _anisotropy(params, step):
    import torch
    s = torch.exp(params["scales"])
    ratio = s.max(1).values / s.min(1).values.clamp_min(1e-8)
    return WEIGHT * torch.relu(ratio - RATIO).mean()


def fit(views, budget_seconds, device):
    return gs3d.fit(views, budget_seconds, device, extra_loss=_anisotropy, random_background=True)


def render(state, cams):
    return gs3d.render(state, cams)

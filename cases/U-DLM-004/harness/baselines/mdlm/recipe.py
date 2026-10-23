"""Baseline: MDLM (Sahoo et al. 2024): the linear schedule alpha(t) = 1 - t (the default when no alpha is given) and
the continuous-time bound itself as the training loss, with low-discrepancy time sampling (dlm.mdlm_loss)."""
from __future__ import annotations

import dlm


def loss(model, x):
    return dlm.mdlm_loss(model, x)

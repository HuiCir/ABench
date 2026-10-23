"""V-BB-006: Imagenette 160-pixel release (fastai, Apache-2.0), shared with IC-RES-002 (see prepare/imagenette.py)."""
from __future__ import annotations

from pathlib import Path

from prepare import imagenette
from prepare.common import Frozen

SOURCES = imagenette.SOURCES


def build(raw: Path) -> Frozen:
    return imagenette.frozen(raw)


def synthetic(n: int) -> Frozen:
    return imagenette.synthetic(n)

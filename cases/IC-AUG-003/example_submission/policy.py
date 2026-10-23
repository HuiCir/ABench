"""IC-AUG-003 reference submission: a RandAugment-style policy (Cubuk et al. 2020) built from the aug_ops primitives.

Every image is flipped and translated as in the standard recipe; then two operations drawn uniformly per image from
a list of nine are applied at a shared magnitude M = 9 on a 0..30 scale; finally an 8 x 8 cutout (DeVries & Taylor
2017). In the last 10% of the epochs only flip and translation are applied, so that the network finishes training
on images close to the test distribution.
"""
from __future__ import annotations

M = 9 / 30           # shared magnitude, as a fraction of each operation's maximum

OPS = [
    lambda ops, x, g: ops.rotate(x, 30 * M, g),
    lambda ops, x, g: ops.shear(x, 0.3 * M, g, axis=0),
    lambda ops, x, g: ops.shear(x, 0.3 * M, g, axis=1),
    lambda ops, x, g: ops.brightness(x, 0.9 * M, g),
    lambda ops, x, g: ops.contrast(x, 0.9 * M, g),
    lambda ops, x, g: ops.saturation(x, 0.9 * M, g),
    lambda ops, x, g: ops.sharpness(x, 0.9 * M, g),
    lambda ops, x, g: ops.posterize(x, 8 - int(4 * M)),
    lambda ops, x, g: ops.solarize(x, 1 - M),
]


def augment(x, ops, gen, epoch, n_epochs):
    x = ops.translate(ops.flip_lr(x, gen), 4, gen)
    if epoch >= 0.9 * n_epochs:
        return x
    for _ in range(2):
        pick = ops.choice(x, len(OPS), gen)
        out = x.clone()
        for i, op in enumerate(OPS):                 # each operation runs only on the images that drew it
            m = pick == i
            if m.any():
                out[m] = op(ops, x[m], gen)
        x = out
    return ops.cutout(x, 8, gen)

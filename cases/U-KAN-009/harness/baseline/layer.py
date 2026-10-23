"""U-KAN-009 reference submission: the MLP layer itself (Linear followed by SiLU; no activation on the output layer).

It is the evaluator's baseline, so it scores an error ratio of about 1 up to run-to-run noise; it shows the interface
and the plumbing, not a method.
"""
import torch.nn as nn


def make_layer(d_in: int, d_out: int, final: bool) -> nn.Module:
    lin = nn.Linear(d_in, d_out)
    return lin if final else nn.Sequential(lin, nn.SiLU())

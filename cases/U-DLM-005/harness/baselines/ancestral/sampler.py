"""Baseline run by the evaluator under the same conditions as every submission: MDLM ancestral sampling with equal
time steps (dlm.ancestral)."""
import dlm


def sample(denoiser, prefix, length, nfe, gen):
    return dlm.ancestral(denoiser, prefix, length, nfe, gen)

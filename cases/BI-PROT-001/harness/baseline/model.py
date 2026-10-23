"""Reference submission: zero-shot ESM-2 wild-type marginals (Meier et al. 2021) from both frozen models, combined
per assay with the 64 labelled mutants.

1. For each model, one forward pass of the wild-type sequence gives log p(a | wild type) at every position; a
   substitution X{i}Y scores log p(Y at i) - log p(X at i) (the "wild-type marginal" of Meier et al. 2021).
2. The two scores are standardised over the assay's mutants and combined by ridge regression fitted on the 64 labelled
   mutants (two weights; ridge 1.0 towards equal weights), so that the sign and the relative trust of the models are
   adapted to each assay. The checkpoint only records these settings: nothing is trained before evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODELS = ("esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D")


def load(checkpoint_dir: str, models_dir: str, device: str):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    out = {"device": device, "ridge": float(cfg.get("ridge", 1.0)), "models": {}}
    for m in MODELS:
        path = str(Path(models_dir) / m)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        net = EsmForMaskedLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
        out["models"][m] = (tok, net)
    return out


def wt_marginal_scores(tok, net, wt: str, mutants: list[str], device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        ids = tok(wt, return_tensors="pt")["input_ids"].to(device)
        logp = torch.log_softmax(net(input_ids=ids).logits[0].float(), -1).cpu().numpy()   # (L + 2, vocab)
    s = np.empty(len(mutants))
    for k, m in enumerate(mutants):
        a, pos, b = m[0], int(m[1:-1]), m[-1]
        s[k] = logp[pos, tok.convert_tokens_to_ids(b)] - logp[pos, tok.convert_tokens_to_ids(a)]   # +1 for <cls>, 1-based
    return s


def predict(model, assay: dict) -> np.ndarray:
    muts = list(assay["mutants"]) + list(assay["few_shot_mutants"])
    feats = []
    for tok, net in model["models"].values():
        s = wt_marginal_scores(tok, net, assay["wild_type"], muts, model["device"])
        feats.append((s - s.mean()) / (s.std() + 1e-8))
    X = np.stack(feats, 1)
    n = len(assay["mutants"])
    Xq, Xf = X[:n], X[n:]
    y = np.asarray(assay["few_shot_scores"], np.float64)
    y = (y - y.mean()) / (y.std() + 1e-8)
    lam = model["ridge"]
    prior = np.full(X.shape[1], 0.5)
    w = np.linalg.solve(Xf.T @ Xf + lam * len(y) * np.eye(X.shape[1]) / X.shape[1],
                        Xf.T @ y + lam * len(y) * prior / X.shape[1])
    return Xq @ w

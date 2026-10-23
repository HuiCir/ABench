"""E-VLA-003 frozen vision-language model (evaluator code, agent-visible).

The evaluator loads SmolVLM2-500M-Video-Instruct (Hugging Face TB 2025, Apache-2.0) itself, inside the policy's
process, freezes it and passes it to load_policy(checkpoint_dir, vlm, processor, device). before() and after() are
called by libero_eval.AgentProcess: after the episodes the SHA-256 of every weight is compared with the one taken at
loading, so a policy that changes the model's weights is detected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

VLM = "SmolVLM2-500M-Video-Instruct"
_DIGEST: dict = {}


def load_vlm(models_dir: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = str(Path(models_dir) / VLM)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    vlm = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)
    vlm = vlm.to(device).eval()
    for p in vlm.parameters():
        p.requires_grad_(False)
    return vlm, processor


def digest(model) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().contiguous().view(-1).view(dtype=__import__("torch").uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def before(models_dir: str, device: str) -> dict:
    vlm, processor = load_vlm(models_dir, device)
    _DIGEST["before"] = digest(vlm)
    return {"vlm": vlm, "processor": processor}


def after(vlm, processor) -> dict:
    return {"vlm_unchanged": int(digest(vlm) == _DIGEST.get("before"))}

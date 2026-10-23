"""Reference submission for E-VLA-003: an MLP action head on pooled layer-16 features of the frozen SmolVLM2,
predicting 8 actions that are executed open loop before the next query."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


class Policy:
    def __init__(self, checkpoint_dir: str, vlm, processor, device: str):
        import torch
        from safetensors.torch import load_file
        import features
        from head import Head
        cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        self.torch, self.features, self.vlm, self.processor, self.device = torch, features, vlm, processor, device
        features.configure(processor)
        self.head = Head(cfg["feat"], cfg["horizon"]).to(device).eval()
        self.head.load_state_dict(load_file(str(Path(checkpoint_dir) / "head.safetensors")))
        self.queue, self.instruction = [], ""

    def reset(self, instruction: str) -> None:
        self.queue, self.instruction = [], instruction

    def act(self, obs: dict) -> np.ndarray:
        if not self.queue:
            torch = self.torch
            f = self.features.encode(self.vlm, self.processor, obs["agentview_rgb"][None], obs["eye_in_hand_rgb"][None],
                                     self.instruction, self.device)
            with torch.no_grad():
                c = self.head(f, torch.as_tensor(obs["state"][None], device=self.device))[0]
            self.queue = list(c.float().cpu().numpy())
        return self.queue.pop(0)


def load_policy(checkpoint_dir: str, models_dir: str, device: str, vlm, processor) -> Policy:
    return Policy(checkpoint_dir, vlm, processor, device)

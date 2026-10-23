"""Reference submission for PH-WELL-002: one Fourier neural operator shared by the three systems (per-system lifting
and projection layers, FiLM conditioning on the normalised parameters), predicting the change of the normalised
frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


class Surrogate:
    def __init__(self, checkpoint_dir: str, device: str):
        import torch
        from fno import FNO
        self.torch, self.device = torch, device
        self.stats = json.loads((Path(checkpoint_dir) / "stats.json").read_text())
        self.net = FNO().to(device).eval()
        self.net.load_state_dict(torch.load(Path(checkpoint_dir) / "fno.pt", map_location=device, weights_only=True))

    def predict(self, system: str, history: np.ndarray, params: np.ndarray) -> np.ndarray:
        torch, st = self.torch, self.stats[system]
        mu = np.asarray(st["mu"], np.float32)[None, None, :, None, None]
        sd = np.asarray(st["sd"], np.float32)[None, None, :, None, None]
        h = torch.as_tensor((history - mu) / sd, device=self.device)
        p = torch.as_tensor((params - np.asarray(st["pm"], np.float32)) / np.asarray(st["ps"], np.float32),
                            device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            y = self.net(system, h, p).float().cpu().numpy()
        return y * sd[:, 0] + mu[:, 0]


def load(checkpoint_dir: str, device: str) -> Surrogate:
    return Surrogate(checkpoint_dir, device)

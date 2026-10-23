"""Reference submission for E-RT-002: a behaviour-cloned chunking policy (planner) and naive asynchronous execution
(controller): a new chunk is requested every EVERY steps, and when it arrives the actions already overtaken by time
are skipped."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

EVERY = 8


class Planner:
    def __init__(self, checkpoint_dir: str, device: str):
        import torch
        from safetensors.torch import load_file
        from net import ChunkPolicy
        cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        self.instr, self.device, self.torch = cfg["instructions"], device, torch
        self.net = ChunkPolicy(len(self.instr), cfg["horizon"]).to(device).eval()
        self.net.load_state_dict(load_file(str(Path(checkpoint_dir) / "policy.safetensors")))
        self.task = 0

    def reset(self, instruction: str) -> None:
        self.task = self.instr.index(instruction.lower().strip()) if instruction.lower().strip() in self.instr else 0

    def plan(self, obs: dict, t: int, request) -> np.ndarray:
        torch, dev = self.torch, self.device
        with torch.no_grad():
            c = self.net(torch.as_tensor(obs["agentview_rgb"][None], device=dev),
                         torch.as_tensor(obs["eye_in_hand_rgb"][None], device=dev),
                         torch.as_tensor(obs["state"][None], device=dev),
                         torch.as_tensor([self.task], device=dev))
        return c[0].float().cpu().numpy()


class Controller:
    def reset(self, instruction: str) -> None:
        self.chunk, self.start, self.last_request = None, 0, -EVERY

    def request(self, t: int):
        if t - self.last_request >= EVERY:
            self.last_request = t
            return {}
        return None

    def receive(self, chunk: np.ndarray, t_issue: int, t_now: int) -> None:
        self.chunk, self.start = np.asarray(chunk), t_issue          # chunk[k] was planned for step t_issue + k

    def act(self, t: int) -> np.ndarray:
        k = min(t - self.start, len(self.chunk) - 1)
        return self.chunk[k]


def load_planner(checkpoint_dir: str, device: str) -> Planner:
    return Planner(checkpoint_dir, device)


def make_controller(checkpoint_dir: str) -> Controller:
    return Controller()

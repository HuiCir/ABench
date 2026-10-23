"""Train the action head on the LIBERO-Spatial and LIBERO-Object demonstrations: frozen-VLM features of every second
frame are computed once, then the head is fitted by L1 regression of 8-step action chunks.

  python3 train.py --data "$ABENCH_DATA_DIR" --models "$ABENCH_MODELS_DIR" --out checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))
import features                                 # noqa: E402
from head import Head                           # noqa: E402

SUITES, HORIZON, STRIDE = ("libero_spatial", "libero_object"), 8, 2


def instruction_of(p: Path) -> str:
    return p.name[: -len("_demo.hdf5")].replace("_", " ")


def demo_features(data: Path, vlm, processor, device: str) -> dict:
    import h5py
    feats, states, chunks = [], [], []
    for suite in SUITES:
        for f in sorted((data / suite).glob("*_demo.hdf5")):
            instr = instruction_of(f)
            with h5py.File(f, "r") as h:
                for k in h["data"]:
                    d = h["data"][k]
                    o, act = d["obs"], np.asarray(d["actions"], np.float32)
                    T = len(act)
                    idx = np.arange(0, T, STRIDE)
                    st = np.concatenate([o["ee_pos"], o["ee_ori"], o["gripper_states"]], 1).astype(np.float32)[idx]
                    ag, wr = np.asarray(o["agentview_rgb"])[idx], np.asarray(o["eye_in_hand_rgb"])[idx]
                    for b in range(0, len(idx), 32):
                        feats.append(features.encode(vlm, processor, ag[b:b + 32], wr[b:b + 32], instr, device)
                                     .cpu().numpy().astype(np.float16))
                    j = np.minimum(idx[:, None] + np.arange(HORIZON)[None], T - 1)
                    states.append(st)
                    chunks.append(act[j])
    return {"feat": np.concatenate(feats), "state": np.concatenate(states), "chunk": np.concatenate(chunks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="checkpoint")
    ap.add_argument("--steps", type=int, default=20000)
    a = ap.parse_args()
    dev = "cuda"
    vlm, processor = features.load_vlm(a.models, dev)
    features.configure(processor)
    d = demo_features(Path(a.data), vlm, processor, dev)
    del vlm
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    head = Head(d["feat"].shape[1], HORIZON).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for step in range(a.steps):
        i = rng.integers(0, len(d["feat"]), 256)
        pred = head(torch.as_tensor(d["feat"][i], device=dev).float(), torch.as_tensor(d["state"][i], device=dev))
        loss = (pred - torch.as_tensor(d["chunk"][i], device=dev)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(step, float(loss), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(out / "head.safetensors"))
    (out / "config.json").write_text(json.dumps({"horizon": HORIZON, "feat": int(d["feat"].shape[1])}))


if __name__ == "__main__":
    main()

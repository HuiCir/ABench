"""C-QNT-003 reference submission (identical to the evaluator's baseline): round-to-nearest (RTN) weight-only
quantization, 4 bits with one fp16 scale and zero point per group of 128 input channels, applied to every matrix of
Qwen2.5-0.5B including the tied embedding; vectors (norms, biases) stay in fp16. No calibration data is used.
Loading dequantizes into a bf16 transformers model, so the runtime is that of the unquantized model."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

GROUP, BITS = 128, 4
COPY = ["config.json", "generation_config.json"]


def _q(w: torch.Tensor):
    out_f, in_f = w.shape
    pad = (-in_f) % GROUP
    w = torch.nn.functional.pad(w.float(), (0, pad)).view(out_f, -1, GROUP)
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** BITS - 1)).clamp(min=1e-8)
    zero = (-lo / scale).round()
    q = (w / scale + zero).round().clamp(0, 2 ** BITS - 1).to(torch.uint8).view(out_f, -1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return {"packed": packed.cpu(), "scale": scale.half().cpu(), "zero": zero.to(torch.uint8).cpu(),
            "shape": (out_f, in_f)}


def _dq(e: dict) -> torch.Tensor:
    out_f, in_f = e["shape"]
    p = e["packed"]
    q = torch.empty(p.shape[0], p.shape[1] * 2, dtype=torch.uint8, device=p.device)
    q[:, 0::2], q[:, 1::2] = p & 15, p >> 4
    w = (q.view(out_f, -1, GROUP).float() - e["zero"].float()) * e["scale"].float()
    return w.view(out_f, -1)[:, :in_f]


def quantize(model_dir: str, calib_tokens, out_dir: str, device: str) -> None:
    from safetensors.torch import load_file
    md, od = Path(model_dir), Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(md / "model.safetensors"))
    packed = {}
    for name, w in sd.items():
        packed[name] = _q(w.to(device)) if w.ndim == 2 else w.half().cpu()
    torch.save(packed, od / "weights.pt")
    for f in COPY:
        if (md / f).is_file():
            shutil.copyfile(md / f, od / f)


def load(artifact_dir: str, device: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    ad = Path(artifact_dir)
    cfg = AutoConfig.from_pretrained(str(ad))
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    packed = torch.load(ad / "weights.pt", map_location="cpu", weights_only=False)
    sd = {k: (_dq(v) if isinstance(v, dict) else v).to(torch.bfloat16) for k, v in packed.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if [m for m in missing if m != "lm_head.weight"] or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    model.tie_weights()
    return model.to(device).eval()

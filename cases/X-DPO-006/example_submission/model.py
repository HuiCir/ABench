"""Reference for X-DPO-006: SmolLM2-360M-Instruct with rank-16 LoRA updates learned by DPO, merged at load time."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import lora  # noqa: E402


def load(checkpoint_dir, models_dir, device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(f"{models_dir}/SmolLM2-360M-Instruct", local_files_only=True,
                                                 torch_dtype=torch.bfloat16)
    cfg = json.loads((Path(checkpoint_dir) / "lora.json").read_text())
    lora.merge(model, load_file(str(Path(checkpoint_dir) / "lora.safetensors")), cfg["r"], cfg["alpha"])
    return model.to(device).eval()

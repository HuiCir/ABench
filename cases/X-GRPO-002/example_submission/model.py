"""X-GRPO-002 reference: Qwen2.5-0.5B-Instruct with the GRPO-trained LoRA updates merged into the base weights."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import lora  # noqa: E402


def load(checkpoint_dir, models_dir, device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    cfg = json.loads((Path(checkpoint_dir) / "lora.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(f"{models_dir}/{cfg['base']}", local_files_only=True,
                                                 torch_dtype=torch.bfloat16)
    lora.merge(model, load_file(str(Path(checkpoint_dir) / "lora.safetensors")), cfg["r"], cfg["alpha"])
    return model.to(device).eval()

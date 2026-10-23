"""Reference for X-DISTILL-010: the distilled SmolLM2-135M student, stored as a full set of weights."""
from pathlib import Path


def load(checkpoint_dir, models_dir, device):
    import torch
    from safetensors.torch import load_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(f"{models_dir}/SmolLM2-135M", local_files_only=True, torch_dtype=torch.float32)
    load_model(model, str(Path(checkpoint_dir) / "student.safetensors"), strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()

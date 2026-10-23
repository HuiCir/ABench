"""Features of the frozen SmolVLM2 backbone: both camera images and the instruction in one chat prompt; the hidden
states of layer LAYER, mean-pooled over the image tokens and over the text tokens (2 x 960 numbers)."""
from __future__ import annotations

import numpy as np

LAYER, SIZE = 16, 512


def load_vlm(models_dir: str, device: str):
    """The frozen backbone exactly as the evaluator loads it (interface/vla_frozen.py: load_vlm)."""
    import torch
    from pathlib import Path
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = str(Path(models_dir) / "SmolVLM2-500M-Video-Instruct")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    vlm = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16)
    return vlm.to(device).eval().requires_grad_(False), processor


def configure(processor) -> None:
    """One 512 x 512 tile per image (no splitting into sub-images), as SmolVLA does."""
    ip = processor.image_processor
    ip.do_image_splitting = False
    ip.size = {"longest_edge": SIZE}


def prompt(processor, instruction: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "text", "text": f"What action should the robot take to {instruction}?"}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def encode(vlm, processor, agent: np.ndarray, wrist: np.ndarray, instruction: str, device: str):
    """agent, wrist: uint8 (B, 128, 128, 3) -> torch (B, 1920) float32."""
    import torch
    from PIL import Image
    text = prompt(processor, instruction)
    imgs = [[Image.fromarray(a), Image.fromarray(w)] for a, w in zip(agent, wrist)]
    inp = processor(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        out = vlm.model(**inp, output_hidden_states=True)
    h = out.hidden_states[LAYER].float()
    img_tok = processor.tokenizer.convert_tokens_to_ids("<image>")
    is_img = (inp["input_ids"] == img_tok).unsqueeze(-1).float()
    is_txt = inp["attention_mask"].unsqueeze(-1).float() * (1 - is_img)
    f_img = (h * is_img).sum(1) / is_img.sum(1).clamp_min(1)
    f_txt = (h * is_txt).sum(1) / is_txt.sum(1).clamp_min(1)
    return torch.cat([f_img, f_txt], -1)

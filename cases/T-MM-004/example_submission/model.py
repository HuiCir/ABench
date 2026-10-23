"""Reference for T-MM-004: frozen DINOv2-small (CLS and mean patch token) and frozen all-MiniLM-L12-v2 (mean-pooled
tokens), each followed by a trained two-layer projection to a shared 256-d space; trained with the symmetric
contrastive loss of CLIP with a learned temperature."""
import io

import numpy as np
import torch

DIM = 256


class Proj(torch.nn.Module):
    def __init__(self, d_in, d_hid=1024):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(d_in), torch.nn.Linear(d_in, d_hid), torch.nn.GELU(),
                                       torch.nn.Linear(d_hid, DIM))

    def forward(self, x):
        return self.net(x)


class Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img = Proj(2 * 384)
        self.txt = Proj(384)
        self.log_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))


def encoders(models_dir, device):
    from transformers import AutoModel, AutoTokenizer
    v = AutoModel.from_pretrained(f"{models_dir}/dinov2-small", local_files_only=True).to(device).eval()
    t = AutoModel.from_pretrained(f"{models_dir}/all-MiniLM-L12-v2", local_files_only=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(f"{models_dir}/all-MiniLM-L12-v2", local_files_only=True)
    for p in list(v.parameters()) + list(t.parameters()):
        p.requires_grad_(False)
    return v, t, tok


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def image_features(v, jpegs, device):
    from PIL import Image
    ims = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BICUBIC)) for b in jpegs]
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 255.0
    x = ((x - MEAN) / STD).to(device)
    h = v(pixel_values=x).last_hidden_state
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)


def text_features(t, tok, texts, device):
    b = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    h = t(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)


def load(states, models_dir, device):
    heads = Heads().to(device).eval()
    heads.load_state_dict(states)
    v, t, tok = encoders(models_dir, device)
    return {"heads": heads, "v": v, "t": t, "tok": tok, "device": device}


@torch.no_grad()
def encode_images(state, jpegs):
    e = state["heads"].img(image_features(state["v"], jpegs, state["device"]))
    return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()


@torch.no_grad()
def encode_texts(state, texts):
    e = state["heads"].txt(text_features(state["t"], state["tok"], texts, state["device"]))
    return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()

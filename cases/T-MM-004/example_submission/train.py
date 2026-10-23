"""Reference training for T-MM-004: frozen features are computed once for all training images and captions, then the
two projections and the temperature are trained with the symmetric InfoNCE loss (batch 1,024 image-caption pairs,
one random caption per image per epoch, AdamW, 15 epochs).

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--quick]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402


def unpack(u, o):
    b = np.asarray(u, np.uint8).tobytes()
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    with np.load(Path(a.data) / "coco_train.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    n_img = len(d["jpeg_offsets"]) - 1
    if a.quick:
        n_img = min(n_img, 512)
    caps = unpack(d["caption_utf8"], d["caption_offsets"])
    owner = d["caption_image"]
    v, t, tok = M.encoders(os.environ.get("ABENCH_MODELS_DIR", "models"), dev)
    feats_i, feats_t = [], []
    with torch.no_grad():
        for s in range(0, n_img, 256):
            jp = [d["jpeg"][d["jpeg_offsets"][i]:d["jpeg_offsets"][i + 1]].tobytes() for i in range(s, min(n_img, s + 256))]
            feats_i.append(M.image_features(v, jp, dev).half().cpu())
        sel = np.flatnonzero(owner < n_img)
        for s in range(0, len(sel), 1024):
            feats_t.append(M.text_features(t, tok, [caps[i] for i in sel[s:s + 1024]], dev).half().cpu())
    fi, ft, own = torch.cat(feats_i), torch.cat(feats_t), torch.from_numpy(owner[sel])
    by_img = [np.flatnonzero(owner[sel] == i) for i in range(n_img)]
    heads = M.Heads().to(dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.05)
    epochs = 1 if a.quick else a.epochs
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n_img)
        for s in range(0, n_img, 1024):
            ib = perm[s:s + 1024]
            tb = np.asarray([rng.choice(by_img[i]) for i in ib])
            ei = torch.nn.functional.normalize(heads.img(fi[ib].float().to(dev)), dim=-1)
            et = torch.nn.functional.normalize(heads.txt(ft[tb].float().to(dev)), dim=-1)
            logits = heads.log_scale.clamp(max=np.log(100)).exp() * ei @ et.T
            y = torch.arange(len(ib), device=dev)
            loss = (torch.nn.functional.cross_entropy(logits, y) + torch.nn.functional.cross_entropy(logits.T, y)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {ep + 1}: loss {loss.item():.3f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({k: v_.detach().cpu() for k, v_ in heads.state_dict().items()}, Path(a.out) / "states.pt")


if __name__ == "__main__":
    main()

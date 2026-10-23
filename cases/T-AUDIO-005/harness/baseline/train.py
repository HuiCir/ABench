"""Reference training for T-AUDIO-005: CTC on the frozen encoder's hidden states, AdamW, 20 epochs over the 10-hour
training set (encoder outputs are recomputed each step; nothing of the encoder is trained).

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--quick]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402


def unpack(u, o):
    b = np.asarray(u, np.uint8).tobytes()
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    import os
    models = a.models or os.environ.get("ABENCH_MODELS_DIR")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    with np.load(Path(a.data) / "train_10h.npz", allow_pickle=False) as z:
        audio, off = z["audio"], z["audio_offsets"]
        texts = unpack(z["text_utf8"], z["text_offsets"])
    auds = [audio[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    enc = M.encoder(models, dev)
    head = M.Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-2)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    idx = {c: i for i, c in enumerate(M.VOCAB)}
    order = np.argsort([len(x) for x in auds])
    batches = [order[i:i + 16] for i in range(0, len(order), 16)]
    epochs = 1 if a.quick else a.epochs
    for ep in range(epochs):
        for bi in np.random.default_rng(ep).permutation(len(batches)):
            b = batches[bi]
            with torch.no_grad():
                h, frames = M.features(enc, [auds[i] for i in b], dev)
            logp = head(h).log_softmax(-1).transpose(0, 1)       # (T, B, V)
            tgt = [torch.tensor([idx[c] for c in texts[i] if c in idx]) for i in b]
            loss = ctc(logp, torch.cat(tgt).to(dev), frames.to(dev), torch.tensor([len(t) for t in tgt]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {ep + 1}: last loss {loss.item():.3f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, Path(a.out) / "states.pt")


if __name__ == "__main__":
    main()

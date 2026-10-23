"""Reference for T-AUDIO-005: the frozen wav2vec 2.0 base encoder, a learned softmax-weighted sum of its 13 hidden
states (as in SUPERB) and one linear CTC layer over 28 characters plus blank; greedy CTC decoding."""
import numpy as np
import torch

VOCAB = ["<blank>"] + list(" 'ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class Head(torch.nn.Module):
    def __init__(self, n_layers=13, dim=768):
        super().__init__()
        self.layer_w = torch.nn.Parameter(torch.zeros(n_layers))
        self.proj = torch.nn.Linear(dim, len(VOCAB))

    def forward(self, hidden):                           # hidden: (L, B, T, D)
        w = torch.softmax(self.layer_w, 0)
        h = (w[:, None, None, None] * hidden).sum(0)
        h = torch.nn.functional.layer_norm(h, h.shape[-1:])
        return self.proj(h)


def encoder(models_dir, device):
    from transformers import Wav2Vec2Model
    enc = Wav2Vec2Model.from_pretrained(f"{models_dir}/wav2vec2-base", local_files_only=True).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def features(enc, audios, device):
    """Batch of int16 arrays → hidden states (L, B, T, D) and frame lengths."""
    x = [torch.from_numpy(np.asarray(a, np.float32) / 32768.0) for a in audios]
    x = [(a - a.mean()) / (a.std() + 1e-7) for a in x]
    lens = torch.tensor([len(a) for a in x])
    pad = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).to(device)
    mask = (torch.arange(pad.shape[1])[None] < lens[:, None]).long().to(device)
    out = enc(pad, attention_mask=mask, output_hidden_states=True)
    frames = enc._get_feat_extract_output_lengths(lens)
    return torch.stack(out.hidden_states), frames


def load(states, models_dir, device):
    head = Head().to(device).eval()
    head.load_state_dict(states)
    return {"enc": encoder(models_dir, device), "head": head, "device": device}


@torch.no_grad()
def transcribe(state, audios):
    h, frames = features(state["enc"], audios, state["device"])
    ids = state["head"](h).argmax(-1).cpu().numpy()
    out = []
    for row, n in zip(ids, frames.tolist()):
        prev, s = -1, []
        for t in row[:n]:
            if t != prev and t != 0:
                s.append(VOCAB[t])
            prev = t
        out.append("".join(s).strip())
    return out

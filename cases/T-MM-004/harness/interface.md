# T-MM-004 interface

## Entry point: `model.py`

```python
def load(states: dict, models_dir: str, device: str):
    """states: the {name: tensor} dict of checkpoint/states.pt; encoders under models_dir/dinov2-small and
    models_dir/all-MiniLM-L12-v2 (transformers, local_files_only=True)."""

def encode_images(state, jpegs: list) -> "np.ndarray (n, d)":   # JPEG bytes, 256×256
def encode_texts(state, texts: list) -> "np.ndarray (n, d)":    # captions
```

- Similarity is the cosine of the returned vectors; d ≤ 1,024 and the same for images and texts.

## Evaluator code you can read

`mm.py` (image packing, recall@K, compositional accuracy), `stages.py`, `metrics.py`, `textpack.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), torchvision, transformers, Pillow, NumPy. Nothing can be installed.

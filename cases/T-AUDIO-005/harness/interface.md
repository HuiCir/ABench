# T-AUDIO-005 interface

## Entry point: `model.py`

```python
def load(states: dict, models_dir: str, device: str):
    """states: the {name: tensor} dict of checkpoint/states.pt. The encoder is models_dir/wav2vec2-base
    (transformers Wav2Vec2Model, local_files_only=True). Return what transcribe() needs."""

def transcribe(state, audios: list) -> list:
    """audios: int16 NumPy arrays at 16 kHz. Return one transcript (str) per utterance."""
```

## Evaluator code you can read

`asr.py` (normalisation, WER/CER, babble mixing), `stages.py` (batching, timing, memory), `metrics.py`,
`textpack.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference submission's code under
`baseline/`.

## Available software

Python 3, PyTorch (CUDA), torchaudio, transformers, NumPy. Nothing can be installed.

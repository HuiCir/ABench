# C-QNT-003 interface

## Entry point: `quantize.py`

```python
def load(artifact_dir: str, device: str):
    """Rebuild the model from the artifact only (your checkpoint/) and return it in eval mode.
    model(input_ids: LongTensor (1, T)) must return logits (1, T, vocab) or an object with .logits."""
```

- The original weights are not readable during the evaluation; copy the configuration files you need into the artifact.
- transformers can rebuild the architecture with `AutoConfig.from_pretrained(artifact_dir)` and `AutoModelForCausalLM.from_config(...)`.
- `quantize.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Quantization program: `train.py`

Write a `train.py` that quantizes `$ABENCH_MODELS_DIR/Qwen2.5-0.5B` (config.json, model.safetensors, tokenizer files;
bf16) using `$ABENCH_DATA_DIR/calib_tokens.npy` and writes the artifact to `checkpoint/`, for example
`python3 train.py --data "$ABENCH_DATA_DIR" --models "$ABENCH_MODELS_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`.
The evaluator does not run it; it is kept with the submission so the artifact can be audited and reproduced. The
reference under `interface/baseline/` (round-to-nearest: its `quantize.py` with `quantize()` and `load()`, and
`train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

- `stages.py`: the evaluation stage (perplexity windows, throughput) and the bf16 reference timing.
- `baselines/rtn/quantize.py`: the round-to-nearest baseline (also a working example of the artifact format and `load()`).
- `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, transformers, tokenizers, safetensors (offline). Nothing can be installed.

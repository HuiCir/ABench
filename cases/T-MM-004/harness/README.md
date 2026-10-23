# T-MM-004 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
T-MM-004 pipeline: image-text retrieval and compositional checks with frozen image and text encoders plus trained
adapters (evaluator and self-check).

One stage, "encode" (the frozen encoders are under models/dinov2-small and models/all-MiniLM-L12-v2):
  timed   check checkpoint/states.pt (named tensors only, at most 10 million numbers), import model.py,
          load(states, models_dir, device)
  finish  encode_images on the held-out images and encode_texts on their captions and on the compositional pairs,
          in batches of 256, timed; embeddings are returned as float32 (dimension at most 1,024)        (≤ 1,200 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `mm.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.

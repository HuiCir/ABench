# T-AUDIO-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
T-AUDIO-005 pipeline: speech recognition with a frozen wav2vec 2.0 encoder and a light trained head, scored on
clean, harder and noisy held-out speakers (evaluator and self-check).

One stage, "transcribe" (the frozen encoder is under models/wav2vec2-base):
  timed   check checkpoint/states.pt (named tensors only), import model.py, load(checkpoint_dir, models_dir, device)
  finish  transcribe(state, audios) on the held-out utterances in batches sorted by length; the time of these calls
          divided by the audio duration is the real-time factor; peak GPU memory is recorded       (≤ 1,800 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `asr.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.

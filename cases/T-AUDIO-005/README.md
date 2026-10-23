# ABench case T-AUDIO-005: Speech recognition from a frozen wav2vec 2.0 encoder with a light trained head: word error rate on clean, harder and noisy unseen speakers

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli, *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
(arXiv:2006.11477).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference needs PyTorch, transformers and the wav2vec 2.0 weights, which were not available where the case was built; the operator runs it at activation. The word and character error rates (Levenshtein distance checked against a brute-force implementation on 300 random pairs), the babble mixing and the packing of audio were checked in the container; the data selection runs on synthetic audio.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/T-AUDIO-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case T-AUDIO-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/t_audio_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- LibriSpeech ASR corpus (Panayotov et al. 2015), parquet export openslr/librispeech_asr: https://huggingface.co/datasets/openslr/librispeech_asr — license: CC BY 4.0 (verified 2026-10-22)
- wav2vec 2.0 base, pre-trained only (facebook/wav2vec2-base): https://huggingface.co/facebook/wav2vec2-base — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case T-AUDIO-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case T-AUDIO-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case T-AUDIO-005 --submission cases/T-AUDIO-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case T-AUDIO-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

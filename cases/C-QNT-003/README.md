# ABench standard case C-QNT-003: Weight-only quantization of Qwen2.5-0.5B at 4.25 bits per parameter

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
(arXiv:2210.17323).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The 4-bit pack/unpack round trip of the baseline was checked in a NumPy transcription; the PyTorch/transformers code has not run.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/C-QNT-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case C-QNT-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/c_qnt_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Qwen2.5-0.5B (Qwen Team), Hugging Face Qwen/Qwen2.5-0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B — license: Apache-2.0 (verified 2026-10-19)
- WikiText-2 raw (Merity et al. 2016), Hugging Face Salesforce/wikitext, config wikitext-2-raw-v1: https://huggingface.co/datasets/Salesforce/wikitext — license: CC BY-SA 3.0 and GFDL (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case C-QNT-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case C-QNT-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case C-QNT-003 --submission cases/C-QNT-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case C-QNT-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

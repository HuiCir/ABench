# ABench standard case LG-HOLD-002: Legal holding identification on CaseHOLD

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Lucia Zheng, Neel Guha, Brandon R. Anderson, Peter Henderson, Daniel E. Ho, *When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset*
(arXiv:2104.08671).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/LG-HOLD-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case LG-HOLD-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/lg_hold_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- CaseHOLD (Zheng et al. 2021), Hugging Face casehold/casehold, config all: https://huggingface.co/datasets/casehold/casehold — license: not stated on the dataset card or in its loading script; the reglab/casehold code repository is Apache-2.0 (verified 2026-10-19)
- roberta-base (Liu et al. 2019), Hugging Face FacebookAI/roberta-base: https://huggingface.co/FacebookAI/roberta-base — license: MIT (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case LG-HOLD-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case LG-HOLD-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case LG-HOLD-002 --submission cases/LG-HOLD-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case LG-HOLD-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

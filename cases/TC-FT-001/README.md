# ABench standard case TC-FT-001: Text classification on AG News within a CPU budget

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov, *Bag of Tricks for Efficient Text Classification*
(arXiv:1607.01759).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics, and the judge itself through tools/oj_eval.py) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/TC-FT-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case TC-FT-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/tc_ft_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- AG News (Zhang, Zhao & LeCun 2015), Hugging Face fancyzhx/ag_news: https://huggingface.co/datasets/fancyzhx/ag_news — license: unknown on the dataset card; the card describes the corpus as provided for research and non-commercial use (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case TC-FT-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case TC-FT-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case TC-FT-001 --submission cases/TC-FT-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case TC-FT-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

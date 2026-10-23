# ABench case DS-GBM-001: A tabular learner trained by the evaluator on twenty unseen datasets

Case built to task statement standard v2.1 from the template of DS-PIPE-002. Anchor: Tianqi Chen, Carlos Guestrin, *XGBoost: A Scalable Tree Boosting System*
(arXiv:1603.02754).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission, the HistGradientBoosting baseline and the whole pipeline (dev split, stage runner with pre-timed inputs, metrics) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/hgb`.

## Files

- `spec/cases/DS-GBM-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case DS-GBM-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/ds_gbm_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Tabular benchmark of Grinsztajn et al. 2022, Hugging Face inria-soda/tabular-benchmark (30 CSV files: 10 dev, 20 held-out): https://huggingface.co/datasets/inria-soda/tabular-benchmark — license: no licence on the dataset card (to_verify before release); each file re-uploads an OpenML dataset (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case DS-GBM-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case DS-GBM-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case DS-GBM-001 --submission cases/DS-GBM-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case DS-GBM-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

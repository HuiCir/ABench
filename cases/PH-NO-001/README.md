# ABench standard case PH-NO-001: A neural surrogate for 1D Burgers' equation (PDEBench)

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar, *Fourier Neural Operator for Parametric Partial Differential Equations*
(arXiv:2010.08895).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/PH-NO-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PH-NO-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/ph_no_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- PDEBench Datasets (Takamoto et al. 2022, arXiv:2210.07182), DaRUS doi:10.18419/darus-2986, file 1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986 — license: CC BY 4.0 (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PH-NO-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case PH-NO-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PH-NO-001 --submission cases/PH-NO-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PH-NO-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

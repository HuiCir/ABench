# ABench standard case MD-MNIST-003: Lightweight medical image classification on BloodMNIST and OrganAMNIST

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni, *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*
(arXiv:2110.14795).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/MD-MNIST-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case MD-MNIST-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/md_mnist_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- MedMNIST (Yang et al. 2023), Zenodo record 10519652 (MedMNIST+ v3.0): https://zenodo.org/records/10519652 — license: CC BY 4.0 (DermaMNIST, CC BY-NC 4.0, is not used); not intended for clinical use (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case MD-MNIST-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case MD-MNIST-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case MD-MNIST-003 --submission cases/MD-MNIST-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case MD-MNIST-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

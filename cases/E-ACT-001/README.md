# ABench standard case E-ACT-001: An action-generation policy for PushT from keypoint observations

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*
(arXiv:2303.04137).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The k-NN policy was exercised on synthetic arrays; rollouts need gymnasium and gym-pusht and have not run.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/E-ACT-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case E-ACT-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/e_act_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- PushT demonstrations with keypoint observations, Hugging Face lerobot/pusht_keypoints: https://huggingface.co/datasets/lerobot/pusht_keypoints — license: Apache-2.0 (verified 2026-10-19)
- gym-pusht simulator (Hugging Face), gym_pusht/PushT-v0: https://github.com/huggingface/gym-pusht — license: Apache-2.0 (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case E-ACT-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case E-ACT-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case E-ACT-001 --submission cases/E-ACT-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case E-ACT-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

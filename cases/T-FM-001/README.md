# ABench case T-FM-001: From generative modelling to control: a fast generative action head for PushT

Case built to task statement standard v2.1 from the template of E-ACT-001. Anchor: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, *Flow Matching for Generative Modeling*
(arXiv:2210.02747).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: training and rollouts need PyTorch, gymnasium, gym-pusht and a GPU, which were not available where the case was built; the operator runs them at activation. The rollout loop and metrics are those of E-ACT-001 with the act() timing added.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/T-FM-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case T-FM-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/t_fm_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- PushT demonstrations with keypoint observations, Hugging Face lerobot/pusht_keypoints: https://huggingface.co/datasets/lerobot/pusht_keypoints — license: Apache-2.0 (verified 2026-10-19)
- gym-pusht simulator (Hugging Face), gym_pusht/PushT-v0: https://github.com/huggingface/gym-pusht — license: Apache-2.0 (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case T-FM-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case T-FM-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case T-FM-001 --submission cases/T-FM-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case T-FM-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

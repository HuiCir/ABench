# ABench case T-SPEC-003: From speculative decoding to diffusion sampling: fewer sequential rounds with parallel evaluations

Case built to task statement standard v2.1 from the template of V-SMP-002. Anchor: Yaniv Leviathan, Matan Kalman, Yossi Matias, *Fast Inference from Transformers via Speculative Decoding*
(arXiv:2211.17192).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch, diffusers, pytorch-fid and a GPU, which were not available where the case was built; the operator runs them at activation. The reference's windowed Picard iteration was run in a NumPy mirror on a toy ε-model: it respects the round and parallelism budgets and lands within 2e-3 of 2R-step DDIM after R rounds (R-step DDIM differs from 2R-step DDIM by 0.47 there).
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/ddim`.

## Files

- `spec/cases/T-SPEC-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case T-SPEC-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/t_spec_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- google/ddpm-cifar10-32 (Ho et al. 2020, arXiv:2006.11239), as in V-SMP-002: https://huggingface.co/google/ddpm-cifar10-32 — license: Apache-2.0 (verified 2026-10-19 (V-SMP-002))
- pytorch-fid Inception weights (Seitzer 2020): https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth — license: Apache-2.0 (verified 2026-10-19 (V-SMP-002))
- CIFAR-10 training set (Krizhevsky 2009), for the reference statistics: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz — license: no explicit license on the dataset page (to_verify)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case T-SPEC-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case T-SPEC-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case T-SPEC-003 --submission cases/T-SPEC-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case T-SPEC-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

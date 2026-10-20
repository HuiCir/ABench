# ABench standard case T-OPT-002: Transferring LLM optimizers to vision-transformer training

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan and others, *Muon is Scalable for LLM Training*
(arXiv:2502.16982).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The baseline and example update rules were run on NumPy arrays through optim_api.Ops, and the random-crop indexing was checked in NumPy; the PyTorch training loop has not run.
- No number in the standard-case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/adamw`.

## Files

- `spec/standard_cases/T-OPT-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case T-OPT-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/t_opt_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- CIFAR-10 (Krizhevsky 2009): https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz — license: no explicit license on the dataset page (to_verify), as in IC-SPD-001

## Activation (operator)

```bash
python3 tools/prepare/run.py --case T-OPT-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case T-OPT-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case T-OPT-002 --submission cases/T-OPT-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case T-OPT-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

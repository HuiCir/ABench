# ABench case V-DST-004: Distilling a CIFAR-10 diffusion model into a one- to four-step generator

Case built to task statement standard v2.1 from the template of V-SMP-002. Anchor: Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever, *Consistency Models*
(arXiv:2303.01469).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: distillation, sampling and the Inception features need PyTorch, diffusers, pytorch-fid and a GPU, which were not available where the case was built; the operator runs them at activation. The FID code is that of V-SMP-002 (checked on toy statistics).
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/V-DST-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case V-DST-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/v_dst_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- google/ddpm-cifar10-32 (Ho et al. 2020, arXiv:2006.11239), the teacher: https://huggingface.co/google/ddpm-cifar10-32 — license: Apache-2.0 (verified 2026-10-19 (V-SMP-002))
- pytorch-fid Inception weights (Seitzer 2020): https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth — license: Apache-2.0 (verified 2026-10-19 (V-SMP-002))
- CIFAR-10 training set (Krizhevsky 2009): https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz — license: no explicit license on the dataset page (to_verify)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case V-DST-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case V-DST-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case V-DST-004 --submission cases/V-DST-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case V-DST-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

# ABench standard case V-SMP-002: A training-free sampler for a frozen diffusion model at small step counts

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Jiaming Song, Chenlin Meng, Stefano Ermon, *Denoising Diffusion Implicit Models*
(arXiv:2010.02502).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The FID computation in metrics.py was checked on toy statistics; sampling and Inception features need PyTorch, diffusers and pytorch-fid and have not run.
- No number in the standard-case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/ddim`.

## Files

- `spec/standard_cases/V-SMP-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case V-SMP-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/v_smp_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- google/ddpm-cifar10-32 (Ho et al. 2020, arXiv:2006.11239): https://huggingface.co/google/ddpm-cifar10-32 — license: Apache-2.0 (verified 2026-10-19)
- pytorch-fid Inception weights (Seitzer 2020): https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth — license: Apache-2.0 (verified 2026-10-19)
- CIFAR-10 training set (Krizhevsky 2009), for the reference statistics: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz — license: no explicit license on the dataset page (to_verify)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case V-SMP-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case V-SMP-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case V-SMP-002 --submission cases/V-SMP-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case V-SMP-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

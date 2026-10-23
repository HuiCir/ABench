# ABench case V-BB-006: A vision-transformer backbone for small data at two compute budgets (Imagenette, 160 pixels, 30 epochs)

Case built to task statement standard v2.1 from the template of V-BB-006. Anchor: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
(arXiv:2010.11929).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code, baseline and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The forward FLOPs at 160 pixels were computed analytically: baseline 1.20 G and 4.54 G, reference 1.27 G and 4.51 G (its large variant drops one block to pay for the convolutional stem), within the budgets of 1.4 G and 4.8 G. The Imagenette archive is pinned by fastai's own size and first-MiB md5.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/vit`.

## Files

- `spec/cases/V-BB-006.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case V-BB-006`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/v_bb_006.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Imagenette, 160-pixel release (fastai), imagenette2-160.tgz: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz — license: Apache-2.0 (github.com/fastai/imagenette @ ba8db395, LICENSE); the images are a subset of ImageNet (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case V-BB-006 --list-sources                 # what to download
python3 tools/prepare/run.py --case V-BB-006 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case V-BB-006 --submission cases/V-BB-006/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case V-BB-006 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

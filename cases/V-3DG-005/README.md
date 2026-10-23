# ABench case V-3DG-005: Sparse-view novel-view synthesis in ten minutes per scene: 24 views of unbounded Mip-NeRF 360 scenes

Case built to task statement standard v2.1 from the template of A-LCTX-006. Anchor: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
(arXiv:2308.04079).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the COLMAP reader (on hand-built binary files), the view split, PSNR and SSIM, and the self-check data path were run in the build container on synthetic scenes. The Mip-NeRF 360 files could not be downloaded here and neither PyTorch nor gsplat is available: the operator prepares the scenes and runs the baseline and the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/gs3d`.

## Files

- `spec/cases/V-3DG-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case V-3DG-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/v_3dg_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Mip-NeRF 360 dataset, parts 1 and 2 (360_v2.zip, 360_extra_scenes.zip): https://jonbarron.info/mipnerf360/ — license: not stated on the project page (research benchmark) (verified 2026-10-23)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case V-3DG-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case V-3DG-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case V-3DG-005 --submission cases/V-3DG-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case V-3DG-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

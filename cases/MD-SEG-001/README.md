# ABench case MD-SEG-001: A 3D segmentation network for a fixed nnU-Net-style recipe: four Medical Segmentation Decathlon tasks, one of them unseen

Case built to task statement standard v2.1 from the template of A-LCTX-006. Anchor: Fabian Isensee, Jens Petersen, Andre Klein, David Zimmerer, Paul F. Jaeger, Simon Kohl, Jakob Wasserthal, Gregor Koehler, Tobias Norajitra, Sebastian Wirkert, Klaus H. Maier-Hein, *nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation*
(arXiv:1809.10486).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: patch sampling, augmentation, sliding-window inference (with an oracle predictor it returns the labels exactly), Dice, normalised surface Dice and the self-check data path were run in the build container on synthetic volumes. The MSD files could not be downloaded here and PyTorch is not available: the operator preprocesses the data, runs the U-Net baseline and the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/unet`.

## Files

- `spec/cases/MD-SEG-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case MD-SEG-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/md_seg_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Medical Segmentation Decathlon Task04_Hippocampus, Task05_Prostate, Task02_Heart, Task09_Spleen (tar files of the msd-for-monai bucket): https://msd-for-monai.s3-us-west-2.amazonaws.com/ — license: CC-BY-SA 4.0 (verified 2026-10-23)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case MD-SEG-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case MD-SEG-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case MD-SEG-001 --submission cases/MD-SEG-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case MD-SEG-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

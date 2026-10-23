# ABench case V-DIT-003: A latent-diffusion backbone at the FLOPs of DiT-S/2: class-conditional Imagenette at 128 pixels under a fixed training recipe

Case built to task statement standard v2.1 from the template of A-LCTX-006. Anchor: William Peebles, Saining Xie, *Scalable Diffusion Models with Transformers*
(arXiv:2212.09748).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the noise schedule, the DDIM update (it recovers the clean latent exactly at the last step), class balancing and the metric arithmetic were run in the build container; the self-check data path runs on synthetic latents. Encoding Imagenette, training, sampling and FID need PyTorch, diffusers and a GPU, which were not available here; the operator runs DiT-S/2 and the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/dit_s`.

## Files

- `spec/cases/V-DIT-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case V-DIT-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/v_dit_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Imagenette 160-pixel release (fastai), as IC-RES-002: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz — license: Apache-2.0 (github.com/fastai/imagenette); images from ImageNet (verified 2026-10-21 (IC-RES-002))
- Stable Diffusion VAE, fine-tuned with EMA (stabilityai/sd-vae-ft-ema): https://huggingface.co/stabilityai/sd-vae-ft-ema — license: MIT (verified 2026-10-23)
- pytorch-fid Inception weights (Seitzer 2020): https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth — license: Apache-2.0 (verified 2026-10-19 (V-SMP-002))

## Activation (operator)

```bash
python3 tools/prepare/run.py --case V-DIT-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case V-DIT-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case V-DIT-003 --submission cases/V-DIT-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case V-DIT-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

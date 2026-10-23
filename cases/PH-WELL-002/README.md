# ABench case PH-WELL-002: One surrogate for three physical systems: autoregressive prediction on The Well at 64 x 64, with unseen physical parameters

Case built to task statement standard v2.1 from the template of X-GRPO-002. Anchor: Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel, Fruzsina J. Agocs, Miguel Beneitez, Marsha Berger, Blakesley Burkhart, Keaton Burns, Stuart B. Dalziel, Drummond B. Fielding, Daniel Fortunato, Jared A. Goldberg, Keiya Hirashima, Yan-Fei Jiang, Rich R. Kerswell, Suryanarayana Maddu, Jonah Miller, Payel Mukhopadhyay, Stefan S. Nixon, Jeff Shen, Romain Watteaux, Bruno Régaldo-Saint Blancard, François Rozet, Liam H. Parker, Miles Cranmer, Shirley Ho, *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning*
(arXiv:2412.00568).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the rollout loop (with a persistence stand-in), VRMSE, window construction, the unseen-parameter split rules (on the real file names) and the self-check data path were run in the build container on synthetic trajectories. The Well files (about 210 GB for the three datasets) were not downloaded here and PyTorch is not available: the operator prepares the data and trains and runs the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PH-WELL-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PH-WELL-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/ph_well_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- The Well: turbulent_radiative_layer_2D (polymathic-ai): https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D — license: CC-BY-4.0 (verified 2026-10-23)
- The Well: active_matter (polymathic-ai): https://huggingface.co/datasets/polymathic-ai/active_matter — license: CC-BY-4.0 (verified 2026-10-23)
- The Well: gray_scott_reaction_diffusion (polymathic-ai): https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion — license: CC-BY-4.0 (verified 2026-10-23)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PH-WELL-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case PH-WELL-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PH-WELL-002 --submission cases/PH-WELL-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PH-WELL-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.

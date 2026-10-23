# PH-WELL-002: One surrogate for three physical systems: autoregressive prediction on The Well at 64 x 64, with unseen physical parameters

Anchor: Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel, Fruzsina J. Agocs, Miguel Beneitez, Marsha Berger, Blakesley Burkhart, Keaton Burns, Stuart B. Dalziel, Drummond B. Fielding, Daniel Fortunato, Jared A. Goldberg, Keiya Hirashima, Yan-Fei Jiang, Rich R. Kerswell, Suryanarayana Maddu, Jonah Miller, Payel Mukhopadhyay, Stefan S. Nixon, Jeff Shen, Romain Watteaux, Bruno Régaldo-Saint Blancard, François Rozet, Liam H. Parker, Miles Cranmer, Shirley Ho, *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning* (arXiv:2412.00568). Baseline used for comparison: one Fourier neural operator shared by the three systems (per-system input and output layers, FiLM conditioning on the parameters), trained jointly on one-step and two-step losses (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **single learned surrogate for several physical systems** — architecture (neural operators, U-Nets, transformers, hybrids), how one network handles systems with different fields and physics, conditioning on physical parameters, normalisation, training for stable autoregressive rollouts (unrolled or noise-injected training, multi-step losses) — that predicts the evolution of **three systems of The Well**: a turbulent radiative layer, active matter and Gray-Scott reaction-diffusion, including at **parameter values never seen in training**.

- Train one model in the sandbox on the public trajectories and submit the checkpoint with the code. The evaluator gives it 4 consecutive frames and the system's parameters and rolls its next-frame prediction out for 10 frames, on held-out trajectories of the training parameters and on trajectories with unseen parameters.
- One network for all three systems: its parameters are shared except for input and output layers and conditioning, which together may hold at most 10% of them; at most 100 million numbers in the checkpoint.
- No pretrained weights, no data other than the public trajectories.
- You may build on any public work (FNO, U-Net and CNextU-Net baselines of The Well, Poseidon, DPOT, Multiple Physics Pretraining, PDE-Refiner, pushforward training, AFNO, Transolver), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): `turbulent_radiative.npz`, `active_matter.npz`, `gray_scott.npz` with `trajectories` (N, T, C, 64, 64) float16, `params` (N, P) and `split` (0: The Well's training split, 1: its validation split), for every parameter set except the unseen ones. The Well (Ohana et al. 2024; `polymathic-ai/turbulent_radiative_layer_2D` @2ee77565, `active_matter` @dc3e9135, `gray_scott_reaction_diffusion` @913ef567; CC-BY 4.0), resampled to 64 x 64 by block averaging; Gray-Scott at every 10th step. Channels and parameters per system are listed in `interface/wellsurr.py`.

The held-out set has two parts per system, each with two 14-frame windows per trajectory (start frames from a secret seed): **in-distribution**, The Well's test split of the public parameter sets; **unseen parameters**, every trajectory of parameter sets absent from the public data (two cooling times of the radiative layer, one activity value of active matter, one Gray-Scott regime).

## Interface

Implement `load(checkpoint_dir, device)` in `model.py`, returning an object with `predict(system, history, params)`: `system` is one of `turbulent_radiative`, `active_matter`, `gray_scott`; `history` float32 (B, 4, C, 64, 64) in physical units (oldest frame first); `params` float32 (B, P); it returns the next frame (B, C, 64, 64). Write a `train.py` that produces `checkpoint/` from the public data.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean rollout VRMSE** (primary) | VRMSE (The Well's variance-scaled RMSE, averaged over fields) of the 10 predicted frames, averaged over windows, then over the six system-set pairs; lower is better | — |
| Per set | Mean rollout VRMSE on in-distribution and on unseen-parameter windows; the gap between them | reported |
| Per system | Rollout, one-step and final-frame VRMSE of each system and set | reported |
| Size | Numbers stored in `checkpoint/` | Threshold: ≤ 100,000,000 |

**Measurement and decision**

- One run (prediction is deterministic). A non-finite predicted frame counts as VRMSE 10, and VRMSE is capped at 10 per frame.
- A submission is valid if it loads within 600 s, every prediction has the right shape, all rollouts finish within 1,800 s and the checkpoint holds at most 100,000,000 numbers; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over windows; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by mean rollout VRMSE, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy, SciPy, safetensors; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 10 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load()
├── train.py            # required: produces checkpoint/ from the public data
├── checkpoint/         # required: trained weights, at most 1 GB
├── src/                # optional: other .py modules
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Predictions must come from the history and parameters alone: no tables of trajectories or frames inside the checkpoint or code, and no state kept between calls.
- The checkpoint must be trained in the sandbox from the public data, with no pretrained weights.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

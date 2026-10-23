# E-VLA-003: An action head for a frozen vision-language model: SmolVLM2-500M controlling LIBERO-Spatial and LIBERO-Object robots

Anchor: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn, *OpenVLA: An Open-Source Vision-Language-Action Model* (arXiv:2406.09246). Baseline used for comparison: an MLP head on the frozen backbone's layer-16 hidden states, mean-pooled over image and over text tokens, regressing 8-step action chunks with L1 loss and executing them open loop (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design the **action head of a vision-language-action policy** on a **frozen SmolVLM2-500M** backbone — which layers and tokens to read, how to condition on proprioception, what to predict and how (regression, discretised or compressed action tokens, diffusion or flow matching, action chunks and how they are executed), small adapters that leave the backbone's weights unchanged — so that, trained on the LIBERO demonstrations, it reaches the **highest success rate on LIBERO-Spatial and LIBERO-Object** within a real-time budget.

- The evaluator loads the backbone itself, freezes it and hands it to your policy; it checks after the episodes that no weight has changed. You submit the trained head (and any adapters), the code that runs it and the training program.
- Train in the sandbox on the 1,000 public demonstrations (two suites, ten tasks each); SmolVLA's pretrained weights (`smolvla_base`, whose action expert was trained for another robot) are available as a starting point. No other pretrained model, no other data.
- At most 100 million numbers in the checkpoint; on average at most 50 ms of policy computation per control step (20 Hz), measured in the policy's process.
- You may build on any public work (OpenVLA, SmolVLA, π0's flow-matching action expert, FAST action tokenisation, action chunking, OpenVLA-OFT's parallel decoding, VLA-Adapter), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/libero_spatial/` and `$ABENCH_DATA_DIR/libero_object/` (read-only): the 20 demonstration files (`<instruction>_demo.hdf5`, 50 human demonstrations each; Liu et al. 2023, `yifengzhu-hf/LIBERO-datasets` @f13aa24a, Apache-2.0) with `actions` (T, 7), `obs/agentview_rgb`, `obs/eye_in_hand_rgb` (T, 128, 128, 3), `obs/ee_pos`, `obs/ee_ori`, `obs/gripper_states`, in the orientation and units of the observations given to the policy (`interface/libero_eval.py`).

`$ABENCH_MODELS_DIR/` holds `SmolVLM2-500M-Video-Instruct/` (Hugging Face TB, @7b375e1b, Apache-2.0), loaded by `interface/vla_frozen.py`, and `smolvla_base/` (lerobot, @d9f33c94, Apache-2.0). The LIBERO package (@8f1084e3, MIT) is installed with its published initial states, which the self-check uses. The held-out set is 10 new initial states per task of both suites (object placements drawn from each task's own placement distribution with a secret seed, kept only if the scene is stable): 200 episodes.

## Interface

Implement `load_policy(checkpoint_dir, models_dir, device, vlm, processor)` in `policy.py`, returning an object with `reset(instruction)` and `act(obs) -> 7 numbers`; `vlm` is the frozen SmolVLM2 (`transformers` `AutoModelForImageTextToText`, bfloat16, on `device`) and `processor` its `AutoProcessor`. Write a `train.py` that produces `checkpoint/` from the public data and models.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean success** (primary) | Mean of the success rates on the held-out LIBERO-Spatial and LIBERO-Object episodes | — |
| Success per suite and task | Success rate of each suite and each of the 20 tasks | reported |
| Real-time cost | Policy computation per control step, averaged over all steps of all episodes | Threshold: ≤ 50 ms |
| Frozen backbone | SHA-256 of every backbone weight after the episodes equals the one taken at loading | Threshold: unchanged |
| Size | Numbers stored in `checkpoint/` | Threshold: ≤ 100,000,000 |

**Measurement and decision**

- One run of all 200 episodes (one simulator seed); an episode succeeds when LIBERO's goal predicate holds within 220 (Spatial) or 280 (Object) control steps after 10 settling steps. Loading is limited to 900 s and the episodes to 7,200 s on one GPU.
- A submission is valid if the policy loads, every action is 7 finite numbers, the episodes finish within the limits, the backbone is unchanged, the mean cost per step is at most 50 ms and the checkpoint holds at most 100,000,000 numbers; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over initial states; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by mean success, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), transformers, safetensors, NumPy, h5py, Pillow, lerobot (for smolvla_base), LIBERO with robosuite and MuJoCo (offscreen rendering); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 12 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 10 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── policy.py           # required: load_policy()
├── train.py            # required: produces checkpoint/ from the public data and models
├── checkpoint/         # required: trained head and adapters, at most 400 MB
├── src/                # optional: other .py modules
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The policy must use the backbone it is given; loading another copy of it (or of any other pretrained model besides `smolvla_base`) from the checkpoint or the models directory makes the submission invalid.
- The policy must act from the observations and the instruction alone: no tables of initial states, object positions or trajectories, and no access to the simulator's internal state.
- The checkpoint must be trained in the sandbox from the public demonstrations and models.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

# T-FM-001: From generative modelling to control: a fast generative action head for PushT

Anchor: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, *Flow Matching for Generative Modeling* (arXiv:2210.02747). Baseline used for comparison: a conditional flow-matching action head over 16-step action chunks with 10 Euler steps (`interface/baseline/`).

## Task

Transfer recent ideas from **generative modelling of images** (flow matching, rectified flows, consistency and few-step distillation) to a **generative action head**: learn by imitation from 206 PushT demonstrations a policy that achieves the **highest target-area coverage** in closed-loop rollouts, while its **average `act()` call takes at most 10 ms** on the reference GPU. Develop and train it on one 24GB GPU in the sandbox.

- Submit your policy code, the trained checkpoint and the training program that produced it. The evaluator loads the checkpoint and lets the policy control the simulator for 150 episodes from reset seeds you do not know, three times over; it never retrains.
- The latency budget is what makes the transfer matter: a diffusion policy with a hundred denoising steps per action chunk is too slow, so the generative head must be sampled in few steps without losing the multimodality that makes generative heads work on PushT.
- Only the provided demonstrations may be used: no pretrained weights, no reinforcement learning in the simulator during training, no scripted controller that solves the task.
- The report must trace which ideas you transferred from generative modelling and how they map onto action generation; every borrowed component must have a cited source.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/demos.npz` (read-only), from Hugging Face `lerobot/pusht_keypoints` (the data of E-ACT-001) (Apache-2.0; 206 episodes, 25,650 frames at 10 Hz):

| Key | Shape | Type | Description |
|---|---|---|---|
| `agent_pos` | (25650, 2) | float32 | End-effector position, pixels in [0, 512] |
| `environment_state` | (25650, 16) | float32 | Eight keypoints of the T block (x, y) |
| `action` | (25650, 2) | float32 | Commanded target position of the end-effector |
| `episode`, `frame` | (25650,) | int64 | Episode index and frame index within the episode |

The simulator is `gym-pusht` (`gym_pusht/PushT-v0`, observation type `environment_state_agent_pos`). The held-out set is 150 reset seeds; it is not in the sandbox. The self-check rolls out on public seeds.

## Interface

Implement `load(checkpoint_dir, device)` and `make_policy(model)` in `policy.py`. The policy object has `reset()` (called at the start of each episode) and `act(obs) -> action`, where `obs` is `{"agent_pos": (2,), "environment_state": (16,)}` and the action is a target position (2,) in [0, 512]. Your `train.py` writes `checkpoint/`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean max reward** (primary) | Per episode, the maximum over time of the simulator's reward, clip(coverage / 0.95, 0, 1), where coverage is the fraction of the target area covered by the block; averaged over the 150 held-out episodes, then over 3 evaluation runs (each with its own seed, which you can use for any sampling in the policy); higher is better | — |
| Act latency | Wall time of all `act()` calls of a run (CUDA synchronised after each call) divided by the number of calls, in milliseconds | Threshold: ≤ 10 ms |
| Success rate | Fraction of episodes in which coverage exceeded 95% (the simulator then ends the episode) | reported |
| Load time | Seconds from the start signal to the return of `load()` (includes importing `policy.py`) | reported (killed at 120 s) |

**Timing and decision**

- Each run is a new process with empty cache directories, timed by the evaluator's parent process (`interface/abench_stage.py`). Rollouts run after the timed phase: 150 episodes of at most 300 steps, all within 2,700 seconds.
- A submission is valid only if the latency threshold holds in every run; valid submissions are ranked by mean max reward; differences inside the 95% interval reported with the score are treated as ties. Actions are clipped to [0, 512].
- If any run crashes, exceeds a time limit, or the policy returns a non-finite action, the submission is invalid.
- The final ABench score is driven mainly by mean max reward, and also considers the correctness and depth of the transferred ideas and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, gymnasium and gym-pusht (versions frozen at activation); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 6 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── policy.py           # required: load() and make_policy()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: trained parameters, any file types, at most 500 MB
├── src/                # optional: other .py modules imported by policy.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- Learn only from the provided demonstrations; no simulator interaction in `train.py`, no reward shaping from the simulator, no hand-written controller for the task.
- `act()` may use only the observation it receives and the policy's own memory within the episode.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

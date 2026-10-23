# X-PPO-001: A policy-optimisation algorithm for small continuous-control tasks at a fixed sample budget, trained by the evaluator over five seeds

Anchor: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, *Proximal Policy Optimization Algorithms* (arXiv:1707.06347). Baseline used for comparison: PPO in the configuration of CleanRL's ppo_continuous_action (one environment, 2,048-step rollouts, 10 epochs, observation and reward normalisation), run by the evaluator on the same seeds (`interface/baselines/ppo/`).

## Task

Design a **policy-optimisation algorithm** that, given **200,000 environment steps** on each of four small continuous-control tasks, learns policies with the **highest mean normalised return** — and does so reliably: the evaluator trains it from scratch with five seeds and averages.

- Submit code only (`agent.py` with `train()`). At evaluation the evaluator calls `train()` once per task and seed; every environment step counts against the task's budget, and a step beyond it raises an error. The returned policy is then run deterministically for one episode from each of 20 held-out initial states per task.
- The four tasks are in `interface/control_envs.py`: Pendulum swing-up, continuous Mountain Car, cart-pole swing-up and continuous-torque Acrobot (gymnasium dynamics, vectorised in NumPy). Each has one action in [−1, 1].
- This is a calibration task with a mature anchor: a careful PPO already does well, and improvements must hold on all four tasks and across seeds. You may build on any public work (on-policy and off-policy actor-critic methods, trust regions, exploration, normalisation), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

There is no data directory: the agent learns only by interacting with the environments in `interface/control_envs.py`.

| Task | Observation | Episode | Reward |
|---|---|---|---|
| `pendulum` | cos θ, sin θ, θ̇ | 200 steps | −(θ² + 0.1 θ̇² + 0.001 u²), torque u = 2a |
| `mountain_car` | position, velocity | ends at the goal, at most 999 steps | 100 at the goal, −0.1 a² per step |
| `cartpole_swing` | x, ẋ, cos θ, sin θ, θ̇ (pole starts down) | ends off the track, at most 500 steps | (1 + cos θ) / 2 per step |
| `acrobot` | cos/sin of both joint angles, both velocities | ends when the tip passes the bar, at most 500 steps | −1 per step |

The 20 held-out initial states per task are drawn from each task's reset distribution with a secret seed; the self-check uses 20 drawn with a fixed development seed.

## Interface

Implement `train(make_env, spec, seed) -> policy` in `agent.py`. `make_env(n)` returns `n` parallel environments (`reset()`, `step(actions)`, auto-reset) that share the task's budget; `spec` gives the task name, dimensions, episode limit and budget. The returned object must have `act(obs) -> actions`, mapping a float32 (n, obs_dim) array to a (n, 1) array in [−1, 1].

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean normalised return** (primary) | Per task, (R − R_random) / (R_upper − R_random), where R is the policy's mean return over the 20 held-out episodes, R_random that of uniformly random actions from the same states and R_upper the task's bound (0, 100, 500 and 0); averaged over the four tasks and the 5 training seeds; higher is better | — |
| Normalised return per task | The same per task, mean over seeds | reported |
| Seed spread | Standard deviation of the per-run primary metric over the 5 seeds | reported |
| Samples used | Environment steps used per task divided by the budget | Threshold: ≤ 1.0 (enforced by the environments) |
| Training time | Seconds of the four `train()` calls in one run (parent clock) | Threshold: ≤ 2,400 s |

**Timing and decision**

- Each run is a new process with empty cache directories, 8 CPU threads and the reference GPU, timed by the evaluator's parent process (`interface/abench_stage.py`); importing `agent.py` and the four `train()` calls are timed. The held-out episodes run after the timed phase and must finish within 300 seconds.
- A submission is valid only if every threshold holds in every run; valid submissions are ranked by mean normalised return. Differences smaller than the spread over the five seeds are treated as ties.
- If any run crashes, exceeds a time limit or returns non-finite actions, the submission is invalid.
- The final ABench score is driven mainly by the mean normalised return relative to the baseline, and also considers how different the algorithm is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 5 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (tasks, sample budget, seeds, held-out initial states), the evaluator also trains the PPO baseline in `interface/baselines/ppo/` and reports your normalised returns relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, SciPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── agent.py            # required: train()
├── src/                # optional: other .py modules imported by agent.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total; no trained parameters, demonstrations or stored policies.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The agent may learn only from the environments returned by `make_env`: it must not import, copy or re-implement the dynamics of `control_envs.py` (no model built from the source), create environments any other way, or change the environment objects.
- Hyperparameters may differ per task only through what `spec` shows (dimensions, episode limit, budget) or what the agent measures during training; no per-task constants keyed on the task name.
- `act()` must not learn or plan during the held-out episodes; it maps observations to actions.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

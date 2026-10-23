# E-RT-002: Acting while the policy is still thinking: asynchronous chunked control of a LIBERO robot under 0–300 ms of inference latency

Anchor: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (arXiv:2304.13705). Baseline used for comparison: a behaviour-cloned chunking policy (two small camera encoders, proprioception and a task embedding, 16 future actions, L1 loss) with naive asynchronous execution: a new chunk every 8 steps, and on arrival the actions already overtaken by time are skipped (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **policy and its execution strategy for acting under inference latency** — what the policy predicts (action chunks, their length, their parameterisation), when to ask for a new plan, how to switch between or blend an old and a new chunk, and how a plan can take into account the actions that will be executed while it is computed (for example inpainting or guidance toward the committed prefix, bidirectional decoding, latency-conditioned training) — so that a robot trained by behaviour cloning on **LIBERO-Spatial** demonstrations keeps **succeeding when every plan arrives 0, 150 or 300 ms late**.

- Train the policy in the sandbox on the 500 public demonstrations and submit the checkpoint with the code. The policy is split in two parts that run in **separate processes**: a **planner** that sees an observation and returns a chunk of actions, and a **controller** that executes one action per control step (20 Hz) and only knows the chunks it has received.
- While the planner works, the robot keeps moving: a chunk planned from the observation at step t is delivered at step t + d, where d is the injected latency (0, 3 or 6 steps) plus the planner's measured time in 50 ms steps. The controller decides when to request a plan and may send the planner a request (for example the actions it will execute meanwhile).
- The policy is trained from scratch: no pretrained weights, no data other than the demonstrations; at most 100 million numbers in the checkpoint.
- You may build on any public work (action chunking and temporal ensembling, diffusion policy, real-time chunking, bidirectional decoding, SmolVLA's asynchronous inference, receding-horizon control), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/libero_spatial/` (read-only): the 10 LIBERO-Spatial demonstration files (`<instruction>_demo.hdf5`, 50 human demonstrations each; Liu et al. 2023, `yifengzhu-hf/LIBERO-datasets` @f13aa24a, Apache-2.0). Each `data/demo_i` holds `actions` (T, 7) and `obs/agentview_rgb`, `obs/eye_in_hand_rgb` (T, 128, 128, 3), `obs/ee_pos`, `obs/ee_ori`, `obs/gripper_states`, in the same orientation and units as the observations given to the policy (`interface/libero_eval.py`).

The LIBERO package (@8f1084e3, MIT) is installed: the ten tasks, their simulator and LIBERO's published initial states, which the self-check uses. The held-out set is 10 new initial states per task (object placements drawn from each task's own placement distribution with a secret seed, kept only if the scene is stable), each run once at each of the three latencies (300 episodes).

## Interface

Implement in `policy.py`: `load_planner(checkpoint_dir, device)` returning an object with `reset(instruction)` and `plan(obs, t, request) -> (H, 7) array`, and `make_controller(checkpoint_dir)` returning an object with `reset(instruction)`, `request(t) -> None or dict`, `receive(chunk, t_issue, t_now)` and `act(t) -> 7 numbers`. Write a `train.py` that produces `checkpoint/` from the public demonstrations.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean success** (primary) | Success rate over the held-out episodes, averaged over the injected latencies of 0, 3 and 6 control steps (0, 150, 300 ms) | — |
| Success per latency | Success rate at each latency; the drop from 0 to 6 steps | reported |
| Timing | Mean planning time, mean effective delay in steps, plans per episode, episode length | reported |
| Per task | Success rate of each of the ten tasks | reported |
| Size | Numbers stored in `checkpoint/` | Threshold: ≤ 100,000,000 |

**Measurement and decision**

- One run of all 300 episodes (one simulator seed); an episode succeeds when LIBERO's goal predicate holds within 220 control steps after 10 settling steps. Loading is limited to 600 s and the episodes to 5,400 s on one GPU.
- A submission is valid if both processes load, every action and chunk is finite and of the right shape, the episodes finish within the limits and the checkpoint holds at most 100,000,000 numbers; otherwise it is invalid.
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

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), NumPy, h5py, safetensors, LIBERO with robosuite and MuJoCo (offscreen rendering); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 10 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── policy.py           # required: load_planner(), make_controller()
├── train.py            # required: produces checkpoint/ from the public demonstrations
├── checkpoint/         # required: trained weights, at most 500 MB
├── src/                # optional: other .py modules
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The planner and the controller may communicate only through the requests and chunks passed by the episode loop; files, sockets, shared memory or any other channel between them make the submission invalid.
- The policy must act from the observations and the instruction alone: no tables of initial states, object positions or trajectories, and no access to the simulator's internal state.
- The checkpoint must be trained in the sandbox from the public demonstrations, with no pretrained weights.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

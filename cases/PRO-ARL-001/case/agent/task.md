# PRO-ARL-001: Agentic reinforcement learning with a Python tool: Qwen3-4B-Base to competition mathematics on 4 × H100 in 24 hours

Anchor: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue and others, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (arXiv:2503.14476). Baseline used for comparison: DAPO (clip-higher, dynamic sampling, token-level loss, overlong shaping) with the Python tool, 400 steps of 64 problems x 16 answers on Qwen3-4B-Base (the reference submission; its code is shown under `interface/baseline/`); the operator also runs GRPO with the same tool, two seeds each.

## Task

Design a **reinforcement-learning method for a tool-using reasoning model** — objective and advantage estimation, rollout sampling and filtering, how tool calls and their outputs enter the loss, curriculum, length control, the use of the 24 hours and four GPUs — that turns **Qwen3-4B-Base** into the most accurate solver of **held-out competition problems**, with a sandboxed Python interpreter as its tool.

- Within one 24-hour session on 4 × H100 you develop, train and deliver a checkpoint. Every training run that may lead to the delivery runs under the case's supervisor (`interface/pro_audit.py`), which registers a hash of a resumable checkpoint at least every 30 minutes; the delivered checkpoint must be one of them, reached from the starting weights.
- The evaluator samples 16 answers per held-out problem from your checkpoint with its own tool loop (`interface/math_tool.py`: `<python>` blocks, 10 s per call, at most 8 calls and 16,384 generated tokens) and checks the last `\boxed{}` answer; it then resumes your run from one registered checkpoint for an hour and compares the logged reward.
- Training prompts: DAPO-Math-17k (public). Rewards may come only from your own checks of the answers; no other model (no reward model, teacher or distillation), no other data, no web access.
- You may build on any public work (PPO, GRPO, DAPO, Dr. GRPO, VAPO, GSPO, RLOO, ReTool, SimpleTIR, ToRL, verl, OpenRLHF, TRL), but the design report must state what you borrowed and what you changed and on which attempt the delivered checkpoint was obtained.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): `dapo_math_17k.parquet` (BytedTsinghua-SIA/DAPO-Math-17k @65877096, Apache-2.0) verbatim; `train.jsonl` with its unique problems as `{problem, answer}` (DAPO's instruction wrapper removed, integer answers); `dev.json`, 60 of them set aside for the self-check. `$ABENCH_MODELS_DIR/Qwen3-4B-Base/` (@906bfd4b, Apache-2.0) holds the starting weights.

The held-out set is the 30 problems of AIME 2025 and the 30 of AIME 2026 (as released by MathArena, CC-BY-NC-SA 4.0; integer answers), not reachable from the sandbox, with a secret sampling seed and a secret choice of the resumed checkpoint.

## Interface

Deliver `checkpoint/`: a Hugging Face model directory with the architecture of Qwen3-4B-Base (config fields compared by the evaluator) and its tokenizer, identical to `model/` of a checkpoint registered by the supervisor. Deliver the code of the registered run (its command is re-run from the submitted directory with `ABENCH_RESUME_FROM` set); the program prints `ABENCH step=<n> reward=<mean reward>` lines.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Accuracy avg@16** (primary) | Share of correct answers among 16 sampled answers per problem, averaged over the 60 held-out problems; higher is better | — |
| Per competition | avg@16 on AIME 2025 and on AIME 2026 | reported |
| Tool use and length | Tool calls per answer, generated tokens per answer, answers without a boxed result | reported |
| Process integrity | Intact hash chain without gaps over 35 minutes; delivered checkpoint registered and traceable to the starting weights; submitted code equal to the run's | Threshold: all hold |
| Sampled resume | Mean logged reward of a one-hour resume from a randomly chosen registered checkpoint, against the original log over the same steps | Threshold: within max(10%, 3 standard errors) |
| Architecture | Delivered config equal to Qwen3-4B-Base's in the compared fields | Threshold: equal |

**Measurement and decision**

- One evaluation run (16 samples per problem with the secret seed); the evaluation takes about 3 hours on one GPU.
- A submission is valid if every threshold holds and the evaluation finishes within its time limit; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over problems; submissions whose intervals overlap are tied on it, and a lead over DAPO and GRPO counts only above two standard deviations of their two seeds.
- The final ABench score is driven mainly by avg@16, and also considers how different the method is from DAPO and GRPO and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 4 × H100 80GB in one node (NVLink); the evaluation uses the same node type. Available software: Python 3, PyTorch (CUDA), vLLM, transformers, safetensors, NumPy, pandas; verl, OpenRLHF and TRL from the allowlisted pip mirror; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 24 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 96 H100-hours (24 hours of the 4-GPU node), development and final training included; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── checkpoint/         # required: Hugging Face model directory (= model/ of a registered checkpoint)
├── train.py            # required: the registered run's program (and anything else it uses)
├── src/                # optional: other modules
├── README.md           # optional
├── report/index.html   # required: design report (states on which attempt the delivery was obtained)
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.sh`, `.yaml`, `.yml`, `.json`, `.toml`, `.md` and `.txt`, at most 1,000,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Only the allowlisted starting weights and data; no other model may be used for rewards, data generation or distillation, and no evaluation problems may be sought.
- Every run that leads to the delivery must be supervised; checkpoints may not be modified after registration, and the session's audit directory is written only by the supervisor.
- Several attempts within the 24 hours are allowed; all are registered, and the report states on which one the delivered checkpoint was obtained.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

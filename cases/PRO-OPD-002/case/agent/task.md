# PRO-OPD-002: Distilling a reasoning model on one node: Qwen3-8B into Qwen3-1.7B-Base for competition mathematics in 24 hours

Anchor: Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem, *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (arXiv:2306.13649). Baseline used for comparison: supervised fine-tuning on correct teacher answers for 100 steps, then on-policy distillation with the per-token reverse KL to Qwen3-8B on the student's own samples (the reference submission; its code is shown under `interface/baseline/`); the operator also runs off-policy distillation (SFT on teacher answers only) and SFT followed by GRPO.

## Task

Design a **distillation method** — what the student learns from (teacher samples, its own samples scored by the teacher, logits, mixtures), the divergence and its estimator, curriculum and length control, combination with verifiable rewards, and the split of four GPUs between teacher inference and student training — that makes **Qwen3-1.7B-Base** the most accurate solver of **held-out competition problems**, with **Qwen3-8B** as the teacher.

- Within one 24-hour session on 4 × H100 you develop, train and deliver the student. Every training run that may lead to the delivery runs under the case's supervisor (`interface/pro_audit.py`), which registers a hash of a resumable checkpoint at least every 30 minutes; the delivered checkpoint must be one of them, reached from the student's starting weights.
- The evaluator samples 16 answers per held-out problem from your student with its own loop (`interface/math_tool.py`, no tools, at most 16,384 generated tokens) and checks the last `\boxed{}` answer; it then resumes your run from one registered checkpoint for an hour and compares the logged loss.
- Training prompts: DAPO-Math-17k (public, with answers). The teacher may be used in any way; no other model and no other data.
- You may build on any public work (knowledge distillation, sequence-level KD, GKD, MiniLLM, DistiLLM, on-policy distillation with reverse KL, speculative KD, GRPO), but the design report must state what you borrowed and what you changed and on which attempt the delivered checkpoint was obtained.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): as in PRO-ARL-001, `dapo_math_17k.parquet` (BytedTsinghua-SIA/DAPO-Math-17k @65877096, Apache-2.0), `train.jsonl` with its unique problems as `{problem, answer}` and `dev.json` with 60 of them for the self-check. `$ABENCH_MODELS_DIR/` holds the student `Qwen3-1.7B-Base/` (@ea980cb0, Apache-2.0) and the teacher `Qwen3-8B/` (@b968826d, Apache-2.0).

The held-out set is the 30 problems of AIME 2025 and the 30 of AIME 2026 (as released by MathArena, CC-BY-NC-SA 4.0; integer answers), not reachable from the sandbox, with a secret sampling seed and a secret choice of the resumed checkpoint.

## Interface

Deliver `checkpoint/`: a Hugging Face model directory with the architecture of Qwen3-1.7B-Base (config fields compared by the evaluator) and a tokenizer whose chat template the evaluator uses, identical to `model/` of a checkpoint registered by the supervisor. Deliver the code of the registered run (its command is re-run from the submitted directory with `ABENCH_RESUME_FROM` set); the program prints `ABENCH step=<n> loss=<float>` lines.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Accuracy avg@16** (primary) | Share of correct answers among 16 sampled answers per problem, averaged over the 60 held-out problems; higher is better | — |
| Per competition | avg@16 on AIME 2025 and on AIME 2026 | reported |
| Length | Generated tokens per answer, answers without a boxed result | reported |
| Process integrity | Intact hash chain without gaps over 35 minutes; delivered checkpoint registered and traceable to the student's starting weights; submitted code equal to the run's | Threshold: all hold |
| Sampled resume | Mean logged loss of a one-hour resume from a randomly chosen registered checkpoint, against the original log over the same steps | Threshold: within max(5%, 3 standard errors) |
| Architecture | Delivered config equal to Qwen3-1.7B-Base's in the compared fields | Threshold: equal |

**Measurement and decision**

- One evaluation run (16 samples per problem with the secret seed); the evaluation takes about 2 hours on one GPU.
- A submission is valid if every threshold holds and the evaluation finishes within its time limit; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over problems; submissions whose intervals overlap are tied on it, and a lead over the operator's distillation and SFT+GRPO runs counts only above two standard deviations of their seeds.
- The final ABench score is driven mainly by avg@16, and also considers how different the method is from GKD-style on-policy distillation and the quality of the sources cited in the design report; the exact weights are not published.

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
- The session lasts at most 24 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 96 H100-hours (24 hours of the 4-GPU node), teacher inference, development and final training included; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
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

- Only the allowlisted student, teacher and data; no other model, and no evaluation problems may be sought.
- Every run that leads to the delivery must be supervised; checkpoints may not be modified after registration, and the session's audit directory is written only by the supervisor.
- Several attempts within the 24 hours are allowed; all are registered, and the report states on which one the delivered checkpoint was obtained.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

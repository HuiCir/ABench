# X-LORA-004: Parameter-efficient fine-tuning of roberta-base on GLUE under a parameter cap

Knowledge cutoff: **2026-07-01**. You may only rely on material that was public and verifiable before that date.

Anchor: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685). Baseline used for comparison: LoRA rank 8 on query and value with a classification head (`interface/baseline/`).

## Task

Design a **parameter-efficient fine-tuning method** for the frozen encoder roberta-base that, with **at most 1,000,000 task-specific parameters per task** (everything that is not the frozen base, including any head), reaches the **highest average score on four GLUE tasks** (RTE, MRPC, CoLA, SST-2). Develop and train it on one 24GB GPU in the sandbox, then submit the trained task-specific tensors.

- Submit your model code (`peft.py`), the trained checkpoint `checkpoint/states.pt` (per task, only its task-specific tensors) and the training program that produced it. The evaluator gives each task's tensors to your `predict()` together with the frozen base model and the held-out sentences; it never retrains.
- The base weights may not be changed or stored: only the tensors in `states.pt` carry information from training to prediction, and they count against the cap.
- You may build on any public work from before the cutoff (LoRA and its variants, adapters, prefix and prompt tuning, (IA)³, BitFit, weight-decomposed or orthogonal adaptation), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search tool: `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call is logged automatically; the log is kept with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address. This is Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/glue_train.npz` (read-only), the training splits of four GLUE tasks (Hugging Face `nyu-mll/glue`):

| Task | Keys | Labels |
|---|---|---|
| RTE | `rte_sentence1`, `rte_sentence2`, `rte_label` | 0 entailment, 1 not entailment |
| MRPC | `mrpc_sentence1`, `mrpc_sentence2`, `mrpc_label` | 1 paraphrase |
| CoLA | `cola_sentence`, `cola_label` | 1 acceptable |
| SST-2 | `sst2_sentence`, `sst2_label` | 1 positive |

Each task also has `<task>_dev_idx` (10% per class, used only by `selfcheck.py`). `$ABENCH_MODELS_DIR/roberta-base/` holds the frozen encoder and tokenizer (Hugging Face `FacebookAI/roberta-base`, MIT). The held-out sets are the official GLUE validation splits (the test labels are not public); they are not in the sandbox.

## Interface

Implement `predict(models_dir, task, state, sentence1, sentence2, device) -> labels` in `peft.py` (`sentence2` is `None` for CoLA and SST-2) and write a `train.py` that saves `{task: {name: tensor}}` to `checkpoint/states.pt` with `torch.save`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **GLUE average** (primary) | Mean over the four tasks of the GLUE validation metric — RTE accuracy, MRPC mean of accuracy and F1, CoLA Matthews correlation, SST-2 accuracy; higher is better | — |
| Per-task metrics | Each task's metric | reported |
| Task-specific parameters | Elements of all tensors stored for a task in `states.pt`; the largest over tasks | Threshold: ≤ 1,000,000 |
| Load time | Seconds from the start signal until `states.pt` is loaded and `peft.py` imported | reported (killed at 180 s) |

**Timing and decision**

- Evaluation runs in a new process with empty caches (`interface/abench_stage.py`), with `states.pt` loaded by the evaluator (`torch.load(..., weights_only=True)`). Prediction of all four validation splits must finish within 600 s.
- A submission is valid only if the parameter threshold holds and `checkpoint/` contains nothing but `states.pt`; valid submissions are ranked by GLUE average.
- If loading or prediction crashes, exceeds a time limit, or `states.pt` holds something other than named tensors per task, or labels have the wrong length, the submission is invalid.
- The final ABench score is driven mainly by the GLUE average, and also considers how different the method is from existing PEFT methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model and never runs `train.py`.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. No network and no package installation; available software: Python 3, PyTorch (CUDA, Triton), NumPy, and Hugging Face transformers, tokenizers and safetensors (versions frozen at activation; offline).
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Paper search (offline, cutoff-filtered, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── peft.py             # required: predict()
├── train.py            # required: the program that produced checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: only states.pt, at most 50 MB
├── src/                # optional: other .py modules imported by peft.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be produced by your `train.py` from the provided data only, inside the sandbox; no pretrained weights other than the provided roberta-base, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your model code and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the checkpoint up to randomness.
- Only the tensors in `checkpoint/states.pt` may carry information from training to `predict()`; other files, or modified or stored base weights, are a violation.
- No pretrained weights other than the frozen roberta-base, no external data and no training on the held-out sentences.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

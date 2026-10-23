# X-DPO-006: Preference optimisation of SmolLM2-360M-Instruct from offline pairs: win rate under a hidden reward model, with drift and length costs

Anchor: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (arXiv:2305.18290). Baseline used for comparison: DPO (β = 0.1) of rank-16 LoRA updates on 20,000 public pairs with the starting model as reference (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design an **offline preference-optimisation objective and recipe** — DPO and its variants, reference handling, data selection and weighting, length control, regularisation toward the starting model — that makes **SmolLM2-360M-Instruct** give answers a **frozen reward model prefers over the starting model's own answers** on held-out prompts, while **drifting little** from the starting model and **not winning by length**.

- Submit a checkpoint, the code that turns it into a causal LM from the starting model, and the training program. The evaluator decodes greedily with a fixed chat template, has a reward model that is not in the sandbox score your answer and the starting model's answer to each prompt, and never retrains.
- Training uses only the public preference pairs; there is no reward model or other model in the sandbox to query.
- The result is a point on four axes: win rate (primary), length-controlled win rate, KL from the starting model and the trained numbers stored in the checkpoint.
- You may build on any public work (DPO, IPO, KTO, ORPO, SimPO, β schedules, length-regularised and reference-free objectives, data filtering by margin), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/prefs_public.npz` (read-only): UltraFeedback preference pairs, binarised (`HuggingFaceH4/ultrafeedback_binarized` @3949bf5f, MIT): `train_*` — the 61,135 train_prefs pairs (prompt, chosen and rejected answer), `dev_*` — 500 test_prefs pairs for the self-check, packed (`interface/textpack.py`). `$ABENCH_MODELS_DIR/SmolLM2-360M-Instruct/`: the starting model and tokenizer (@a10cc151, Apache-2.0).

The held-out set is 1,000 other test_prefs pairs chosen with a secret seed; they are not in the sandbox. The reward model is an open sequence-classification reward model of 0.6B parameters (Apache-2.0) kept outside every sandbox.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` in `model.py`, returning a causal LM with the interface of a transformers model (`forward(input_ids, attention_mask)` returning `.logits`, and `generate()`) that uses the SmolLM2 tokenizer; write a `train.py` that produces the checkpoint.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Win rate** (primary) | Fraction of the 1,000 held-out prompts where the reward model scores your answer above the starting model's answer (ties count half); both answers greedy, at most 512 new tokens; higher is better | — |
| Length-controlled win rate | Win probability at equal length: a logistic regression of wins on log(your words + 1)/(starting model's words + 1), read at 0 | reported (Pareto axis) |
| KL from the starting model | Sum over your answers of log π(answer) − log π_start(answer), per token | reported (Pareto axis) |
| Reward gain, lengths | Mean reward difference; mean words and the ratio to the starting model's | reported |
| Pair accuracy | Fraction of held-out pairs whose chosen answer has the larger implicit reward log π − log π_start | reported |
| Trained numbers | Numbers in the tensor files of `checkpoint/` | reported (Pareto axis) |

No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and the code follows the rules (see below).

**Measurement and decision**

- Generation and log-probabilities run in one process (`generate` stage, loading killed at 600 s, 2,400 s in all), the reward model in a second process that never imports your code (`judge` stage); `interface/lmpost.py` has the template and decoding.
- The primary metric is reported with a 95% bootstrap interval over prompts; submissions whose intervals overlap are tied on it. Valid submissions also form a Pareto front over the four axes.
- The final ABench score is driven mainly by the win rate and the position on the front, and also considers how different the objective is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. The primary metric is reported with a 95% bootstrap interval over the held-out examples (`RESAMPLE` in `interface/metrics.py`; `selfcheck.py` prints the same interval on the dev split). Submissions whose intervals overlap are ranked as tied on the primary metric.
5. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
6. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), transformers, safetensors, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load()
├── train.py            # required: reproduces the checkpoint
├── src/                # optional: other .py modules
├── checkpoint/         # required: adapters or weights, at most 1 GB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Train only from the starting model and the public pairs, and from data you derive from them by program (for example answers sampled from your own model); no other models, reward models or datasets.
- `load()` must build the model from `models_dir/SmolLM2-360M-Instruct` and `checkpoint/`; the model must not detect or adapt to the evaluation.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

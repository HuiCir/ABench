# X-SFT-005: Supervised fine-tuning of SmolLM2-360M for verifiable instruction following, with the cost in trained parameters and forgetting

Anchor: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer, *QLoRA: Efficient Finetuning of Quantized LLMs* (arXiv:2305.14314). Baseline used for comparison: LoRA of rank 16 on every attention and MLP projection, one epoch over 50,000 public conversations with the loss on assistant turns (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **supervised fine-tuning recipe** — data selection and mixing from the provided conversations, synthetic constraint data built from them, loss masking, full or parameter-efficient updates, quantised training, schedules — that turns the base model **SmolLM2-360M** into an assistant that **follows verifiable formatting and content instructions** (word limits, bullet counts, JSON output, keywords, case, end phrases and more) on requests it has not seen, **without forgetting** its general language modelling and with **few trained parameters**.

- Submit a checkpoint, the code that turns it into a causal LM from the base model, and the training program. The evaluator formats every held-out prompt with a fixed chat template, decodes greedily, and checks each constraint with the code in `interface/ifcheck.py`; it never retrains.
- The result is a point on three axes: instruction following (primary), forgetting (the increase of held-out web-text loss over the base model) and the number of trained numbers stored in the checkpoint.
- You may build on any public work (LoRA and QLoRA, DoRA and other adapters, loss masking and packing, instruction-data synthesis and filtering, verifiable-constraint training), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/sft_public.npz` (read-only): `conv_*` — the first 100,000 conversations of smol-smoltalk (`HuggingFaceTB/smol-smoltalk` @f73fe857, Apache-2.0) as JSON lists of `{role, content}`, packed (`interface/textpack.py`); `dev_request_*` — 500 single-turn requests taken after them; `dev_tokens` — 1,048,577 FineWeb-Edu tokens (SmolLM2 tokenizer) for the self-check's forgetting estimate. `$ABENCH_MODELS_DIR/SmolLM2-360M/`: the base model and tokenizer (`HuggingFaceTB/SmolLM2-360M` @f8027fd0, Apache-2.0).

The held-out set is not in the sandbox: 600 single-turn requests from smol-smoltalk's test split, each followed by one to three compatible constraints of the fourteen types in `interface/ifcheck.py`, with parameters drawn from a secret seed, and 1,048,577 held-out FineWeb-Edu tokens. The self-check uses the dev requests with constraints from a public seed.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` in `model.py`, returning a causal LM with the interface of a transformers model (`forward(input_ids, attention_mask)` returning `.logits`, and `generate()`) that uses the SmolLM2-360M tokenizer; write a `train.py` that produces the checkpoint.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Prompt-level accuracy** (primary) | Fraction of the 600 held-out prompts whose response satisfies every one of its constraints (strict checkers of `ifcheck.py`); higher is better | — |
| Constraint-level accuracy | Fraction of all constraints satisfied, overall and per type | reported |
| Forgetting | Mean next-token loss of your model on 1,048,577 held-out FineWeb-Edu tokens minus that of the base model (nats); lower is better | reported (Pareto axis) |
| Trained numbers | Numbers stored in the tensor files of `checkpoint/` (.safetensors, .pt, .bin); full fine-tuning counts all 362M | reported (Pareto axis) |
| Response length | Mean words per response | reported |

No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and the code follows the rules (see below).

**Measurement and decision**

- Prompts are formatted as `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` and decoded greedily with at most 512 new tokens, stopping at `<|im_end|>` or `<|endoftext|>` (`interface/lmpost.py`); loading is killed at 600 s and the evaluation at 1,800 s, on one GPU.
- The primary metric is reported with a 95% bootstrap interval over prompts; submissions whose intervals overlap are tied on it. Valid submissions also form a Pareto front over the three axes.
- The final ABench score is driven mainly by prompt-level accuracy and the position on the front, and also considers how different the recipe is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

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

- Train only from the provided base model and conversations and from data you derive from them by program; no other datasets, pretrained models or model outputs from outside the sandbox.
- `load()` must build the model from `models_dir/SmolLM2-360M` and `checkpoint/`; everything trained must be in `checkpoint/`.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

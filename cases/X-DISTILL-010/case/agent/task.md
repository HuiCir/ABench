# X-DISTILL-010: Knowledge distillation from SmolLM2-1.7B into SmolLM2-135M on a fixed token budget: held-out language-modelling loss

Anchor: Geoffrey Hinton, Oriol Vinyals, Jeff Dean, *Distilling the Knowledge in a Neural Network* (arXiv:1503.02531). Baseline used for comparison: continued training of SmolLM2-135M on 30M public tokens with half cross-entropy and half forward KL to the teacher's distribution (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **knowledge-distillation recipe** — which teacher signals to use (full or truncated distributions, temperatures, hidden states), how to mix them with the data loss, which tokens and documents to spend the budget on, schedules — that makes the **SmolLM2-135M student**, starting from its pretrained weights, reach the **lowest held-out loss** on educational web text, with **SmolLM2-1.7B as the teacher** and the same 120M public tokens.

- Submit the student's weights (same architecture and tokenizer as SmolLM2-135M), the code that loads them and the training program. The evaluator measures the student's mean next-token loss on held-out documents; it never retrains.
- The teacher may be run as much as your compute budget allows in the sandbox (for example to precompute and store its outputs); it is not available to the student at evaluation.
- You may build on any public work (logit distillation and its temperature, top-k and sequence-level distillation, feature and attention transfer, data selection by teacher loss, MiniLLM-style reverse KL), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/fwedu_train_000001.bin` (read-only): 120M tokens of FineWeb-Edu sample/10BT (`HuggingFaceFW/fineweb-edu` @87f09149, ODC-BY) tokenised with the SmolLM2 tokenizer, documents separated by `<|endoftext|>` (llm.c shard format, uint16; `interface/token_data.py` has `read_shard()`). `$ABENCH_MODELS_DIR/SmolLM2-135M/` (student start, @93efa2f0) and `SmolLM2-1.7B/` (teacher, @effd688a), both Apache-2.0.

The held-out set is 4,194,305 tokens of another FineWeb-Edu file (as in A-KV-004); it is not in the sandbox. The self-check uses the last 1,048,577 tokens of the public shard, which the reference does not train on.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` in `model.py`, returning a causal LM with SmolLM2-135M's exact parameter shapes and the interface of a transformers model (`forward(input_ids)` returning `.logits`); write a `train.py` that produces the checkpoint.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss** (primary) | Mean next-token cross-entropy (nats) of the student on the 4,194,305 held-out tokens in windows of 1,024; lower is better | — |
| Same architecture | The returned model's parameter names and shapes equal those of SmolLM2-135M | Threshold: = 1 |
| Improvement and gap closed | Starting student's loss minus yours; that improvement divided by the gap between the starting student and the teacher | reported |
| Teacher loss | The teacher's loss on the same tokens | reported |

**Measurement and decision**

- One stage in a new process on the reference GPU; loading is killed at 600 s and the three loss evaluations at 1,200 s.
- Valid submissions are ranked by held-out loss; differences below 0.003 nats are treated as ties.
- The final ABench score is driven mainly by held-out loss, and also considers how different the recipe is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), transformers, safetensors, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 7 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load()
├── train.py            # required: reproduces the checkpoint
├── src/                # optional: other .py modules
├── checkpoint/         # required: the student's weights, at most 1 GB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Train only on the public tokens, from the provided student and teacher; no other data or models.
- The student keeps SmolLM2-135M's architecture, tokenizer and parameter count; `load()` must not use the teacher.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

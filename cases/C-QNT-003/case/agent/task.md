# C-QNT-003: Weight-only quantization of Qwen2.5-0.5B at 4.25 bits per parameter

Anchor: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (arXiv:2210.17323). Baseline used for comparison: round-to-nearest 4-bit quantization with group size 128 (`interface/baselines/rtn/`, also the reference under `interface/baseline/`).

## Task

Design a **post-training, weight-only quantization method** for the language model Qwen2.5-0.5B that stores the model in **at most 4.25 bits per parameter on average** (all parameters, embedding included) and keeps its **WikiText-2 perplexity as low as possible**, with inference not much slower than the original bf16 model. Develop and run the quantization on one 24GB GPU in the sandbox, then submit the quantized artifact.

- Submit your model code (`quantize.py` with `load()`), the quantized artifact in `checkpoint/` and the program that produced it (`train.py`, which runs your quantization on the frozen bf16 model with the provided calibration tokens). The evaluator runs your `load()` on `checkpoint/` in a process **without access to the original weights** and measures perplexity on the held-out test text; it never re-quantizes.
- No training on the held-out text, no other models, no extra data beyond the calibration tokens; fine-tuning the quantized weights on the calibration tokens is allowed.
- You may build on any public work (GPTQ/OBQ, AWQ, SmoothQuant-style rescaling, incoherence processing, vector and non-uniform codebooks, mixed precision), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_MODELS_DIR/Qwen2.5-0.5B/` holds the frozen model (Hugging Face `Qwen/Qwen2.5-0.5B`, Apache-2.0; 494,032,768 parameters in bf16, tied embedding). `$ABENCH_DATA_DIR/` (read-only) holds WikiText-2 raw (Salesforce/wikitext, CC BY-SA 3.0 / GFDL) tokenized with the model's tokenizer, each split's documents joined with "\n\n":

| File | Contents |
|---|---|
| `calib_tokens.npy` | int32 token ids of the train split (calibration) |
| `dev_tokens.npy` | int32 token ids of the validation split (used by the self-check as held-out text) |

The held-out set is the token ids of the WikiText-2 test split, evaluated in non-overlapping windows of 2,048 tokens; it is not in the sandbox.

## Interface

Implement `load(artifact_dir, device) -> model` in `quantize.py` (and whatever quantization code you like, for example `quantize(model_dir, calib_tokens, out_dir, device)` as in the baseline). `train.py` writes the artifact to `checkpoint/`; `load` returns a module whose call `model(input_ids)` gives logits (or an object with `.logits`) of shape (1, T, vocab).

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Perplexity** (primary) | exp of the mean next-token cross-entropy over all non-overlapping 2,048-token windows of the WikiText-2 test split; lower is better | — |
| Bits per parameter | 8 × total bytes of all files in `checkpoint/` / 494,032,768 | Threshold: ≤ 4.25 |
| Throughput ratio | Prefill tokens per second of the loaded model on one 2,048-token sequence, divided by that of the original bf16 model loaded with transformers and timed by the evaluator in the same evaluation | Threshold: ≥ 0.5 |
| Load time | Seconds of `load()` (includes importing `quantize.py`) | reported (killed at 300 s) |

**Timing and decision**

- The evaluation runs in new processes with empty caches (`interface/abench_stage.py`); your code runs in the first one, which receives only `checkpoint/`. Perplexity and throughput must be measured within 600 s after loading.
- A submission is valid only if both thresholds hold; valid submissions are ranked by perplexity.
- If loading or the measurements crash or exceed a time limit, the submission is invalid.
- The final ABench score is driven mainly by perplexity relative to the round-to-nearest baseline, and also considers how different the method is from existing quantizers and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, Hugging Face transformers, tokenizers and safetensors (versions frozen at activation; offline); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── quantize.py         # required: load()
├── train.py            # required: the program that quantized the model into checkpoint/ (kept for audit, not run by the evaluator)
├── checkpoint/         # required: the quantized artifact, everything load() needs, at most 262,455,533 bytes (4.25 bits per parameter)
├── src/                # optional: other .py modules imported by quantize.py or train.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The artifact must be produced by your `train.py` from the frozen model and the calibration tokens only, inside the sandbox; no other models, no external data and no hand-set parameters copied from elsewhere.
- The evaluator loads `checkpoint/` with your `load()` and runs inference only; it does not run `train.py`, which is kept for audit and must reproduce the artifact up to randomness.
- `load()` may read only the artifact directory; any attempt to reach the original weights or other files is a violation.
- No use of the held-out text.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

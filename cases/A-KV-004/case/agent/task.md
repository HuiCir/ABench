# A-KV-004: An attention and key-value cache structure for SmolLM2-135M that keeps quality at a third of the cache

Anchor: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (arXiv:2305.13245). Baseline used for comparison: SmolLM2-135M converted to multi-query attention by mean-pooling its key-value heads (GQA uptraining), run by the evaluator under identical conditions (`interface/baselines/mqa/`).

## Task

Design an **attention and key-value cache structure** for the pretrained SmolLM2-135M that, after a short continued training, keeps the **lowest held-out loss** while its decoding cache holds **at most 7,680 bytes per token** — a third of the original model's 23,040.

- Submit code only (`model.py` with `build(base_dir, device)`). The evaluator loads the pretrained weights through your `build`, which replaces the model's attention mixers, then continues training the whole model on 30M FineWeb-Edu tokens with a fixed recipe, once for each of 2 seeds unknown to you. It trains the multi-query baseline the same way.
- The evaluator measures the cache itself: the bytes of every tensor your mixers keep in the decoding cache, as the context grows from 1,024 to 2,048 tokens under bf16 autocast. Step-by-step decoding from the cache must reproduce the full forward pass.
- Loss on 8,192-token windows (four times the training length) is reported, to show how the structure extrapolates in length.
- You may build on any public work (multi- and grouped-query attention, latent attention, cross-layer sharing, cache quantisation or eviction, sliding windows, hybrids), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/fwedu_train_000001.bin` (read-only): 120M tokens of FineWeb-Edu (Hugging Face `HuggingFaceFW/fineweb-edu`, sample/10BT, ODC-BY) tokenised with the SmolLM2 tokenizer, documents separated by token 0, in the llm.c shard format (`interface/token_data.py`). `$ABENCH_MODELS_DIR/SmolLM2-135M/`: the pretrained model (Apache-2.0).

The held-out set is the first 4,194,305 tokens of another FineWeb-Edu file (the first 2,097,152 are scored); it is not in the sandbox. The self-check holds out the last 2,097,153 training tokens.

## Interface

Implement `build(base_dir, device) -> smol_lm.SmolLM` in `model.py`: load the pretrained model with `smol_lm.load_base(base_dir, device)` and replace `model.layers[i].mixer` by modules with the mixer interface of `interface/smol_lm.py` (`forward(x, pos0, cache) -> (y, cache)`). Anything else in the model may be changed too, but it is trained by the evaluator's recipe.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss** (primary) | Mean next-token cross-entropy (nats) on 2,097,152 held-out tokens in windows of 2,048 after continued training, mean over 2 seeds; lower is better | — |
| Cache size | Growth of the decoding cache in bytes per token of context between 1,024 and 2,048 tokens (bf16 autocast, `smol_lm.kv_bytes_per_token`) | Threshold: ≤ 7,680 |
| Decoding consistency | Largest relative difference between logits of prefill + 64 single-token steps and of the full forward (fp32) | Threshold: ≤ 0.01 |
| Length extrapolation | Held-out loss on windows of 8,192 tokens, and its difference to 2,048 | reported |
| Loss vs multi-query | Differences to the baseline | reported |
| Training time | Seconds of the continued training (parent clock) | Threshold: ≤ 3,000 s |
| Divergence | 1 if the training loss became non-finite or exceeded 20 | Threshold: 0 |

**Fixed training setting** (`interface/lm_train.py`, `interface/stages.py`): 915 steps of 16 windows of 2,048 tokens (two micro-batches of 8; about 30M tokens), AdamW at a peak of 3e-4 (β = 0.9, 0.95; weight decay 0.1 on matrices), 50 warm-up steps then cosine decay to 0.1, clipping at 1.0, bf16 autocast with fp32 weights.

**Decision**

- A submission is valid only if every threshold holds for the 2-seed means; valid submissions are ranked by held-out loss. Differences smaller than the spread over the two seeds are treated as ties.
- If any run crashes or exceeds a time limit (the stage is killed at 3,300 s), the whole submission is invalid.
- The final ABench score is driven mainly by the held-out loss relative to the baseline, and also considers how different the structure is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, recipe), the evaluator also continues the training of the multi-query baseline in `interface/baselines/mqa/` and reports your losses relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), safetensors, NumPy, PyYAML; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 6 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours (the whole session on one GPU); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: build(base_dir, device)
├── src/                # optional: helper modules imported by model.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Start from the pretrained weights in `base_dir` only; no other pretrained models, data or network access; the evaluator does all training.
- Everything a mixer needs to continue a sequence must be in the cache it returns (no hidden state kept elsewhere across calls, except values recomputed within one forward pass, such as keys shared between layers).
- The model must stay causal and must not change the loss, the optimiser or the data.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

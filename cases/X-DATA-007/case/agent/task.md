# X-DATA-007: A data pipeline for a raw web pool: deduplication, filtering and mixing that minimise a fixed small model's loss on held-out educational text

Anchor: Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini, *Deduplicating Training Data Makes Language Models Better* (arXiv:2107.06499). Baseline used for comparison: exact and MinHash deduplication followed by a hashed-n-gram logistic-regression quality filter trained on the target examples, top-scoring documents up to the token budget (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **data pipeline** — exact and near-duplicate removal, quality and domain filtering, rating documents with small models trained on the provided examples, mixing and up- or down-weighting by repetition — that **selects training documents from a raw web pool** so that a **fixed 30M-parameter language model**, trained by the evaluator on your selection for a fixed number of tokens, reaches the **lowest loss on held-out educational web text**.

- Submit code only: `pipeline.py` with `select(pool_dir, budget_tokens, workdir)` returning document indices in training order (repeats allowed). The evaluator runs it on 8 CPU cores without a GPU, then trains the fixed model on the concatenated documents with a fixed recipe and measures held-out loss.
- The pool contains near-duplicates (across Common Crawl dumps, and planted copies with small edits whose clusters are known to the evaluator); how much repetition helps or hurts is part of the problem.
- You may build on any public work (exact and MinHash deduplication, suffix-array deduplication, perplexity and classifier filters, DoReMi-style mixing, data-constrained scaling), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only), built from FineWeb (`HuggingFaceFW/fineweb` @9bb295dd, ODC-BY; deduplicated only within each dump) and FineWeb-Edu (`HuggingFaceFW/fineweb-edu` @87f09149, ODC-BY), GPT-2 BPE tokens (`openai-community/gpt2` @607a30d7):

| File | Contents |
|---|---|
| `pool_text.npz` | 416,000 documents, shuffled: the first 100,000 of dumps CC-MAIN-2013-20, 2023-14, 2023-50 and 2024-18, plus 16,000 planted near-copies (packed text, `interface/textpack.py`) |
| `pool_tokens.bin`, `pool_index.npz` | Their GPT-2 tokens (each document ends with `<|endoftext|>`; llm.c shard format, `interface/token_data.py`), `token_offsets` and the source `dump` of each document |
| `target_examples.npz` | 5,000 FineWeb-Edu documents: examples of the target distribution |
| `dev_text.npz` | 2,097,153 FineWeb-Edu tokens (other documents) and 2,097,153 tokens of dump CC-MAIN-2024-26, for the self-check |

The held-out sets are not in the sandbox: 2,097,153 tokens of another FineWeb-Edu file (target) and 2,097,153 tokens of dump CC-MAIN-2024-51 (general), plus the clusters of the planted copies.

## Interface

Implement `select(pool_dir, budget_tokens, workdir) -> 1-D int array` in `pipeline.py`: indices into the pool in training order (repeats allowed), totalling at least 20M tokens; `budget_tokens` (98,304,000) is what the fixed model will see, and `workdir` is a writable scratch directory.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Target loss** (primary) | Mean next-token cross-entropy of the fixed model on the 2,097,153 held-out FineWeb-Edu tokens after training on your selection; lower is better | — |
| General loss | The same on the held-out tokens of a later FineWeb dump | reported |
| Duplicates kept | Extra copies of planted near-duplicate clusters among your distinct selected documents, as a fraction of all extra copies in the pool; duplicate rate of the selection | reported |
| Selection | Selected tokens, documents and distinct documents; CPU seconds of `select()` | reported |
| Divergence | Training diverged | Threshold: = 0 |

**Measurement and decision**

- The fixed model is `norm_lm.LM` in its "mu2" configuration (8 layers, width 384, GPT-2 vocabulary, about 30M parameters) with pre-RMSNorm (`interface/fixed_norm.py`), trained by `lm_train.py` (AdamW, learning rate 2e-3, warm-up and cosine decay, bf16) for 3,000 steps of 32 windows of 1,024 tokens drawn uniformly at random from your concatenated selection, so documents you repeat are seen more often.
- `select()` runs in a process with 8 CPU cores and no GPU and is killed at 3,600 s; training runs in a second process on the reference GPU and is killed at 2,400 s.
- Valid submissions are ranked by target loss; differences below 0.01 nats are treated as ties.
- The final ABench score is driven mainly by target loss, and also considers how different the pipeline is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, NumPy, SciPy, scikit-learn (the select stage); PyTorch (CUDA) for the training stage; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── pipeline.py         # required: select()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; models you train inside `select()` must be trained there, within its time limit.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Select only from the pool, using the pool, the target examples and anything you compute from them inside `select()`; no other data, word lists or pretrained models.
- `select()` must not read the dev token sets or anything outside `pool_dir` and `workdir`.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

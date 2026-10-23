# FN-QA-004: Numerical reasoning over financial reports with a frozen 1.5B model: a retrieval and program pipeline for FinQA

Anchor: Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, William Yang Wang, *FinQA: A Dataset of Numerical Reasoning over Financial Data* (arXiv:2109.00122). Baseline used for comparison: TF-IDF evidence retrieval, four nearest training questions with their gold evidence and program as few-shot examples, greedy decoding of one FinQA program and its execution (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **pipeline around a frozen Qwen2.5-1.5B-Instruct** — evidence retrieval over a report's text and tables, prompting or decoding strategy, program generation and repair, verification, self-consistency, small trained components — that gives the **highest execution accuracy** on held-out **FinQA** questions about earnings reports.

- Submit the pipeline code, a checkpoint (for example indexes, exemplar banks, adapters or a small reranker you train in the sandbox) and the program that produced it. The evaluator calls your `answer()` on the held-out questions with the frozen model available and scores the answers; it never retrains.
- Each question comes with the report's text before and after a table and the table itself; the answer is a number (or yes/no) obtained by a few arithmetic steps over values in them. You may return the answer, a FinQA program that the evaluator executes, or both.
- The model's weights are fixed: you may add adapters with at most 20M trained parameters in total, but no other pretrained model and no external data.
- You may build on any public work (retriever-generator pipelines, program-of-thought and program-aided prompting, self-consistency, tool use, table linearisation, small-model fine-tuning), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only), FinQA (Chen et al. 2021, `github.com/czyssrs/FinQA` @0f16e286, MIT licence; reports from FinTabNet), verbatim:

| File | Questions | Contents per question |
|---|---|---|
| `train.json` | 6,251 | `pre_text`, `post_text`, `table`, `qa` with the question, the gold program, its executed answer `exe_ans` and the gold evidence `gold_inds` |
| `dev.json` | 883 | the same |

`$ABENCH_MODELS_DIR/Qwen2.5-1.5B-Instruct/` holds the frozen model (Apache-2.0, revision 989aa798).

The held-out set is FinQA's test split (1,147 questions); only the question, texts and table of each reach your pipeline, after the timed phase. The authors' private test set has no public answers and is not used. Because the test questions are public on the internet, the sandbox has no network and no data from outside it may be used.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `answer(state, items) -> list` in `pipeline.py`. Each item is a dict with `id`, `question`, `pre_text` and `post_text` (lists of sentences) and `table` (list of rows of strings, the first row the header); return, per item, a dict with `answer` (a number, a numeric string or "yes"/"no") and/or `program` (a FinQA program, executed by the evaluator with `interface/finqa.py`). Write a `train.py` that produces the checkpoint from the public data.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Execution accuracy** (primary) | Fraction of the 1,147 held-out questions whose answer equals the gold executed value (FinQA's convention: percentages as fractions, rounded to 5 decimals; tolerance max(5e-5, 1e-4·\|gold\|)); the submitted `answer`, or the executed `program` when no answer is given | — |
| Program accuracy | Fraction whose submitted program executes to the gold value; share of questions with a program and with a valid one | reported |
| By difficulty | Accuracy on one-step and multi-step questions, and on questions whose evidence is in the table or only in the text | reported |
| Cost | Milliseconds per question (batches of 32, one GPU), load time | reported |

No threshold applies to any metric: a submission is valid if it loads, answers every question within the time limits and the code follows the rules (see below).

**Measurement and decision**

- One run with a seed you do not know (it fixes the shuffled order); accuracies are means over questions. `answer()` is called on batches of 32 questions in that order, after loading has finished (killed at 600 s); all questions must be answered within 3,600 s on one GPU.
- The primary metric is reported with a 95% bootstrap interval over questions; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by execution accuracy, and also considers how different the pipeline is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), transformers, NumPy, SciPy, scikit-learn; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── pipeline.py         # required: load() and answer()
├── train.py            # required: produces checkpoint/ from the public data
├── checkpoint/         # required: indexes, exemplars, adapters; at most 500 MB
├── src/                # optional: other .py modules
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- Use only the provided model and data; adapters may add at most 20M trained parameters; no other pretrained model, no external data, no network.
- `answer()` treats each question on its own: no use of other questions of the batch, no state kept between calls, and no lookup of answers by question text.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

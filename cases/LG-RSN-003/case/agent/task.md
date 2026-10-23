# LG-RSN-003: Legal reasoning with a frozen 1.5B model: a pipeline for LegalBench tasks seen and unseen during development

Anchor: Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton et al., *LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models* (arXiv:2308.11462). Baseline used for comparison: each task's LegalBench base prompt (instructions and few-shot examples) filled with the row, and the label whose continuation has the highest mean log-probability under the frozen model (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **pipeline around a frozen Qwen2.5-1.5B-Instruct** — prompt construction, retrieval of rules or examples, decomposition into sub-questions, calibration of label choices, ensembling, small trained components — that gives the **highest mean balanced accuracy** on **LegalBench** classification tasks: twenty tasks you can develop on, scored on new rows, and fifteen tasks you never see, given only at evaluation with their instructions, labels and a few examples.

- Submit the pipeline code, a checkpoint and the program that produced it. The evaluator calls your `predict()` task by task with the frozen model available; it never retrains.
- Every task arrives with its LegalBench base prompt (instructions, a handful of solved examples and a query template with `{{column}}` slots), its label set and its training rows. The fifteen unseen tasks include three from families with no development task (unfair terms of service, privacy-policy relevance, international citizenship law), so a pipeline tuned to the twenty development tasks will not do.
- The model's weights are fixed: you may add adapters with at most 20M trained parameters in total, but no other pretrained model and no external data.
- You may build on any public work (prompting and calibration of few-shot classifiers, rule-recall and issue-spotting prompts, retrieval of similar examples, self-consistency, parameter-efficient tuning), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/legalbench_public.json` (read-only): the twenty development tasks of LegalBench (Guha et al. 2023; task data from `nguha/legalbench` @daec8237 on Hugging Face, prompts from `github.com/HazyResearch/legalbench` @b46bf4ff; each task CC BY 4.0 or MIT). Per task: `name`, `prompt_template`, `labels`, `columns` (the slots of the template; other columns, which could reveal the label, are dropped), `train` (its LegalBench training rows, 1–10) and `dev` (up to 100 rows of its test split, with answers).

| Group | Tasks |
|---|---|
| Development (20) | abercrombie, hearsay, ucc_v_common_law, proa, insurance_policy_interpretation, function_of_decision_section, oral_argument_question_purpose, overruling, corporate_lobbying, nys_judicial_ethics, citation_prediction_classification, sara_entailment, three contract_nli, three cuad, one maud and one supply_chain_disclosure task |
| Unseen (15) | not named here; three come from task families with no development task |

`$ABENCH_MODELS_DIR/Qwen2.5-1.5B-Instruct/` holds the frozen model (Apache-2.0, revision 989aa798).

The held-out set is up to 300 test rows per task (drawn with a secret seed; for development tasks none of them is a dev row) and the full description of the unseen tasks. LegalBench is public: the sandbox has no network and no outside data may be used.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `predict(state, task, rows) -> list[str]` in `pipeline.py`. `task` is a dict with `name`, `prompt_template`, `labels`, `columns` and `train` (see `interface/legaltasks.py`); `rows` is a list of dicts {column: value}; return one label per row, spelled as in `task['labels']`. Write a `train.py` that produces the checkpoint from the public data.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean balanced accuracy** (primary) | Balanced accuracy (mean recall over the labels present) of each of the 35 held-out tasks, averaged over tasks; an answer matching no label counts as wrong | — |
| By group | The same mean over the 20 development tasks, the 12 unseen tasks from known families and the 3 from new families | reported |
| Per task | Balanced accuracy of every task; share of answers matching no label | reported |
| Cost | Milliseconds per row (one GPU), load time | reported |

No threshold applies to any metric: a submission is valid if it loads, answers every row within the time limits and the code follows the rules (see below).

**Measurement and decision**

- One run with a seed you do not know, which fixes the order of tasks; `predict()` gets batches of at most 64 rows of one task, after loading has finished (killed at 600 s); all rows must be answered within 3,600 s.
- The primary metric is reported with a 95% bootstrap interval over rows; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by the mean balanced accuracy, and also considers how different the pipeline is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

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
├── pipeline.py         # required: load() and predict()
├── train.py            # required: produces checkpoint/ from the public data
├── checkpoint/         # required: anything your pipeline loads; at most 500 MB
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
- Nothing may depend on a task's name: the pipeline must work from the task's prompt, labels and examples, as it must for unseen tasks; rows are answered on their own, with no state kept between calls.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

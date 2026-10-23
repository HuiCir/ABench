# BI-PROT-001: Protein mutation-effect prediction with a frozen ESM-2 on 30 held-out deep mutational scans (ProteinGym, 64 labels per assay)

Anchor: Pascal Notin, Aaron W. Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Hansen Spinner, Nathan Rollins, Ada Shaw, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Rose Orenbuch, Yarin Gal, Debora S. Marks, *ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction* (arXiv:2312.05451). Baseline used for comparison: ESM-2 wild-type marginals of both frozen models combined per assay by ridge regression on the 64 labelled mutants (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **mutation-effect predictor** built on a **frozen ESM-2 protein language model** that, for 30 deep mutational scanning assays it has never seen, ranks single-substitution mutants by their measured fitness — given the wild-type sequence and **64 labelled mutants per assay** — with the **highest mean Spearman correlation**.

- Submit a checkpoint (any lightweight head or settings, at most 50 MB) and the code that loads it. The evaluator calls `predict()` once per held-out assay; the ESM-2 weights (150M and 650M parameters) are provided read-only and must stay frozen.
- You have 128 public assays with all their labels for development and for training anything you like (for example a head shared across assays); the held-out assays come from proteins that none of the public assays measure.
- The 64 labelled mutants per held-out assay are drawn at random; zero-shot methods may ignore them.
- You may build on any public work (masked- and wild-type-marginal likelihoods, embeddings with regression heads, few-shot adaptation, structure- or MSA-free priors), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/dev_<DMS_id>.npz` (read-only): the 128 public assays of ProteinGym v1.3 (Notin et al. 2023, MIT) with wild type of at most 600 residues and 500–10,000 single substitutions; single substitutions only.

| Key | Type | Description |
|---|---|---|
| `wild_type` | str | Wild-type amino-acid sequence |
| `mutants` | str (M,) | Substitutions such as `A23G` (1-based positions in `wild_type`) |
| `scores` | float64 (M,) | Measured fitness (higher is fitter), as in ProteinGym's `DMS_score` |
| `selection_type`, `uniprot` | str | Coarse selection type and UniProt identifier |

`$ABENCH_MODELS_DIR/esm2_t30_150M_UR50D` and `esm2_t33_650M_UR50D`: the ESM-2 weights and tokenizers (Hugging Face format). The 30 held-out assays (12 stability, 8 organismal fitness, 6 activity, 3 expression, 1 binding) and the labels of their query mutants are not in the sandbox.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` and `predict(model, assay) -> scores` in `model.py`. `assay` is a dict with `wild_type` (str), `mutants` (list of str), `few_shot_mutants` (list of 64 str), `few_shot_scores` (float array) and `selection_type` (str); `predict` returns one float per entry of `mutants`, higher meaning fitter.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean Spearman** (primary) | Spearman correlation between your scores and the measured fitness of each held-out assay's query mutants (the 64 labelled ones excluded), averaged over the 30 assays; higher is better | — |
| Spearman by selection type | Averaged within each coarse selection type, and the mean of those averages (ProteinGym's aggregation) | reported |
| Spearman per assay | For each held-out assay | reported |
| Load time | Seconds from the start signal to the return of `load()` | reported (killed at 300 s) |

No threshold applies to any metric: a submission is valid if it loads, runs within the time limits and returns well-formed scores (see below).

**Timing and decision**

- Loading and prediction run in a new process with empty cache directories on the reference GPU, timed by the evaluator's parent process (`interface/abench_stage.py`); the 30 `predict()` calls must finish within 1,800 seconds.
- A submission is valid if it loads, runs within the time limits, returns finite scores of the right length and follows the rules (checked by inspecting the code); valid submissions are ranked by mean Spearman. Differences inside the 95% interval over mutants reported with the score are treated as ties.
- The final ABench score is driven mainly by mean Spearman, and also considers how different the method is from existing methods and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), transformers, NumPy, SciPy, scikit-learn; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `marks.hms.harvard.edu`.
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and predict()
├── train.py            # required: reproduces the checkpoint from the public data
├── src/                # optional: other .py modules
├── checkpoint/         # required: head parameters or settings, at most 50 MB
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- ESM-2 must stay frozen: no fine-tuning or re-pretraining of its weights, before or during evaluation; heads, adapters outside the model and per-assay fitting on the 64 labelled mutants are allowed.
- The checkpoint must be produced by `train.py` from the public assays only; no other protein data, MSAs, structures or pretrained models.
- `predict` must not look up assays or proteins by identity (for example a table keyed on sequences).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

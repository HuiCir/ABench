# B-OPT-002: An optimizer found by searching a primitive language: novel against known rules, trained on the fixed micro-LM

Anchor: Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le, *Symbolic Discovery of Optimization Algorithms* (arXiv:2302.06675). Baseline used for comparison: AdamW written in the primitive language (`interface/baselines/adamw/program.yaml`), trained by the evaluator under identical conditions; its learning rate and weight decay are tuned by the operator before activation.

## Task

**Search for an optimizer** in a small **primitive language** — momenta, element-wise maps, normalisations, orthogonalisation, bias corrections — as Lion was found, and submit the program that trains the **fixed 32M-parameter language model of B-OPT-001 to the lowest held-out loss** while being **genuinely new**: its updates must differ from every known optimizer's.

- Submit `program.yaml`, written in the language of `interface/primitive_dsl.py` (at most 2 state tensors, 6 constants, 20 assignments and 40 primitive calls), with your search code and its log. The evaluator never runs your Python: it compiles the program, screens it, and trains the model with it.
- On a gradient stream from a seed you do not know, the program's updates are compared with AdamW, Lion, Signum, SGD with momentum (over a grid of their momentum constants), Adafactor, AdEMAMix, MARS, Cautious AdamW and Muon; a mean cosine above 0.98 to any of them makes the submission invalid. It must also stay finite when gradients are scaled by 1e-6 or 1e6.
- Then the evaluator trains the model exactly as in B-OPT-001 (3,000 steps of 32 × 1,024 tokens, warm-up and cosine decay, your program's learning rate), three times with different seeds, and next to it the tuned AdamW program.
- You may build on any public work (symbolic discovery of optimizers, evolutionary and regularised search, sign-based and orthogonalised updates, learned optimizers), but the design report must state what you borrowed, how you searched and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only): the B-OPT-001 training shards `fineweb_train_000001.bin` and `fineweb_train_000002.bin` (FineWeb-10B GPT-2 tokens, `kjj0/fineweb10B-gpt2` @889765ea, MIT; text ODC-BY), 100M tokens each.

`interface/behaviour.py` generates gradient streams from any seed (a slowly rotating low-rank signal, noise and heavy-tailed spikes); the self-check uses seed 17. The held-out part is 4,194,305 tokens of `fineweb_val_000000.bin` and the evaluator's screen seed.

## Interface

Write `program.yaml` in the primitive language of `interface/primitive_dsl.py`: `state` (≤ 2 names), `constants` (≤ 6), `lr` (peak learning rate) and `program` (≤ 20 assignments `name = expression` over g, p, state, constants and earlier names, assigning `update`); the parameter moves by −lr × schedule × update. `primitive_dsl.validate` checks a program as the evaluator does.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss** (primary) | Mean next-token cross-entropy on 4,194,304 held-out tokens after training, mean of the three runs; lower is better | — |
| Novelty | Largest mean cosine similarity between your updates and any known optimizer's on the gradient stream | Threshold: ≤ 0.98 |
| Stability | Finite updates on the gradient stream and at gradient scales 1e-6 and 1e6 | Threshold: finite |
| Divergence | Any run with a non-finite or > 20 training loss | Threshold: = 0 |
| Step time | Training seconds relative to the AdamW program in the same run | Threshold: ≤ 1.25 |
| Relative to AdamW | Loss difference to the tuned AdamW program; descent cosine and program size | reported |

**Measurement and decision**

- A submission is valid if its program parses, passes the novelty, stability, divergence and step-time thresholds, and trains within 1,500 s per run; otherwise it is invalid. Valid submissions are ranked by held-out loss; differences below the spread of the AdamW program over the three seeds are ties.
- The final ABench score is driven mainly by held-out loss relative to AdamW, and also considers how different the rule is from existing ones, the auditability of the search and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, schedule), the evaluator also trains the model with the AdamW program and reports your loss and step time relative to it. AdamW is tuned, not left at its defaults: before activation the operator sweeps its learning rate over {1e-3, 2e-3, 3e-3, 4.5e-3, 6e-3} and weight decay over {0.05, 0.1} on the self-check split and freezes the best pair in the baseline's program.yaml; until then 3e-3 and 0.1 are placeholders.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, NumPy, PyYAML; PyTorch (CUDA) for your own training runs; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 10 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── program.yaml        # required: the optimizer
├── search.py           # required: the search that produced it
├── search_log.jsonl    # required: one line per evaluated candidate
├── src/                # optional: helper modules of the search, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.jsonl`, `.md` and `.txt` files, at most 2,000,000 bytes in total (the search log may be long).
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The optimizer is the program only: no Python of yours runs at evaluation, and the program may use only the language's primitives.
- The search must be reproducible from `search.py` and the public data; `search_log.jsonl` lists what it evaluated.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

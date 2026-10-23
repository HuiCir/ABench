# T-OPT-002: Transferring LLM optimizers to vision-transformer training

Anchor: Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan and others, *Muon is Scalable for LLM Training* (arXiv:2502.16982). Baseline used for comparison: AdamW, run by the evaluator under identical conditions (`interface/baselines/adamw/`).

## Task

Take what has been learned about optimizers for language-model pretraining (orthogonalised updates, preconditioning, update-RMS matching, schedule-free methods and others) and design an **update rule for training a vision transformer**. With a fixed ViT, data pipeline and budget (30 epochs of CIFAR-10 on one 24GB GPU), your rule should reach the **highest test accuracy** without being slower per step than allowed.

- Submit the update rule only. At evaluation the evaluator trains the fixed ViT from scratch with your rule once for each of 3 seeds unknown to you and classifies the CIFAR-10 test set; it trains the AdamW baseline the same way.
- The rule must be your own design: transferring an LLM optimizer unchanged (for example plain Muon, which the evaluator's reference submission implements) is the starting point, not the answer. The design report must explain what changes when moving from language models to vision transformers and state precisely what you borrowed and where it is in the code.
- The evaluator does not tune hyperparameters: it uses the defaults you declare in `hparam_space.yaml`.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train.npz` (read-only), CIFAR-10 (Krizhevsky 2009):

| Key | Shape | Type | Description |
|---|---|---|---|
| `x` | (50000, 32, 32, 3) | uint8 | Training images, RGB, channels last |
| `y` | (50000,) | int64 | Class 0–9 |
| `dev_idx` | (5000,) | int64 | Dev split for self-checking (500 per class), used only by `selfcheck.py` |

The held-out set is the 10,000-image official CIFAR-10 test set; it is not in the sandbox.

## Interface

Implement a functional update rule in `optimizer.py` (subclass `optim_api.FunctionalRule`, export the class as `RULE`) and declare its hyperparameters in `hparam_space.yaml`; the interface is the same as in the language-model optimizer case B-OPT-001. The training framework wraps the rule as a torch optimizer and runs the fixed loop in `interface/stages.py` on the model in `interface/vit_model.py`.


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Test accuracy** (primary) | Top-1 accuracy of the trained ViT on the 10,000 CIFAR-10 test images, mean over 3 seeds; higher is better | — |
| Accuracy vs AdamW | Test accuracy minus the AdamW baseline's on the same seeds | reported |
| Step-time ratio | Training time of your run divided by that of the AdamW baseline, mean over seeds | Threshold: ≤ 1.25 |
| Training time | Seconds from the start signal to the end of training (includes importing `optimizer.py`), mean over seeds | Threshold: ≤ 1,200 s |
| Optimizer state | Elements of all state tensors divided by the number of model parameters | Threshold: ≤ 3.0 |
| Divergence | 1 if the training loss became non-finite or exceeded 20 | Threshold: 0 |

**Fixed training setting** (all in `interface/stages.py`): ViT with 4×4 patches embedded by a linear map, width 192, depth 6, 3 heads, MLP 768, class token (about 2.7M parameters, every weight matrix 2D); 30 epochs of batch 512 (2,910 steps); random crop with 4-pixel reflect padding and horizontal flip; label smoothing 0.1; bf16 autocast with fp32 weights; learning-rate factor: linear warm-up over 200 steps then cosine decay to 0 (unless your rule sets `owns_schedule: true`); 1D parameters use `lr × 0.15`.

**Decision**

- A submission is valid only if every threshold holds for the 3-seed means; valid submissions are ranked by test accuracy.
- If any seed crashes or exceeds a time limit (the training stage is killed at 1,320 s), the whole submission is invalid.
- The final ABench score is driven mainly by test accuracy relative to the baseline, and also considers how different the rule is from existing optimizers and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 3 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (hardware, seeds, data, schedule), the evaluator also trains the ViT with the AdamW baseline in `interface/baselines/adamw/` and reports your accuracy and step time relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `www.cs.toronto.edu`.
- The session lasts at most 4 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 4 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── optimizer.py        # required: exports RULE (a FunctionalRule subclass)
├── hparam_space.yaml   # required: rule name, ≤ 4 tunable hyperparameters with defaults, fixed hyperparameters
├── src/                # optional: helper modules imported by optimizer.py
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The rule may use only `(p, g, state, step, lr)` and its own hyperparameters; it must not read files, environment variables or the network, inspect the model or data, or modify `p` or `g` in place.
- All tensor arithmetic goes through `ops` (`optim_api.Ops`); state lives only in the `state` dict.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

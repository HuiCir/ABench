# PRO-SPD-004: Pretraining speedrun: GPT-2 Medium scale to a FineWeb loss target on 4 × H100

Anchor: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford and others, *Training Compute-Optimal Large Language Models* (arXiv:2203.15556). Baseline used for comparison: the modded-nanogpt GPT-2 Medium track record of 2025-11-12 (Muon, FlexAttention, 454M parameters), ported from 8 to 4 GPUs with the same global batch (`interface/baseline/`).

## Task

Design a complete pretraining recipe — architecture, optimizer, schedule, batch and sequence strategy, precision, kernels, distributed execution — that trains a **dense language model of at most 550M stored parameters** on one node with **4 × H100** to a **held-out FineWeb loss of at most 2.92** in the **shortest wall-clock time**. This is the setting of the GPT-2 Medium track of the modded-nanogpt speedrun, on half its hardware. The reference ports the track's public record to four GPUs; the target is out of reach for conventional AdamW recipes in the time cap, so valid entries need speedrun-class methods, and the bar is the ported record.

- Submit code only: `train.py` (launched with torchrun on all four GPUs) and `model.py` (rebuilds the trained model for evaluation). At evaluation the run is executed from scratch twice, with seeds unknown to you; time is measured from launch until every training process has exited.
- Only the provided training shards may be used: no pretrained weights, distillation from other models, external data, or reading of the validation tokens.
- Dense means every token passes through all parameters (no mixture-of-experts routing); the size cap counts every tensor in the saved state.
- You may build on any public work (the speedrun's record history, Muon, value embeddings, FP8 kernels, sequence-length warm-up, compute-optimal scaling), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use the results to iterate as your budget allows. The evaluator then runs your submission from scratch under the same pipeline. The reference code in `interface/baseline/` runs as is and is a starting point.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only) holds the 103 training shards of FineWeb-10B pre-tokenized with the GPT-2 BPE (Hugging Face `kjj0/fineweb10B-gpt2`, MIT; the data of the modded-nanogpt speedrun):

| Files | Contents |
|---|---|
| `fineweb_train_000001.bin` … `fineweb_train_000103.bin` | 256 int32 header (magic 20240520, version 1, token count), then about 100M uint16 tokens each |

`interface/token_data.py` reads the format. The held-out set is the first 10,485,760 predicted tokens of the official validation shard `fineweb_val_000000.bin` (the speedrun's validation set); it is not in the sandbox.

## Interface

`train.py` is run as `torchrun --standalone --nproc_per_node=4 train.py --data DIR --out DIR --seed S` and must write the trained weights to `OUT/model.pt` (a state dict). `model.py` defines `build_model()` returning the module whose `load_state_dict` accepts that file and whose call `model(idx)` returns logits (B, T, vocab), and may set `EVAL_SEQ_LEN` (default 1,024).


The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Training time** (primary) | Wall-clock seconds from the launch of torchrun until all training processes have exited, mean (= median) of the 2 runs; lower is better | — |
| Held-out loss | Mean next-token cross-entropy (nats) of the saved model on the 10,485,760 validation tokens, evaluated in consecutive windows of `EVAL_SEQ_LEN`, mean of the 2 runs | Threshold: ≤ 2.92 |
| Model size | Elements of all tensors in `model.pt` (embedding tables included; the reference has 454,496,336) | Threshold: ≤ 550,000,000 |
| Time cap | Each run must finish within 3 hours | Threshold: ≤ 10,800 s |

**Timing and decision**

- Each run starts in new processes with empty compiler caches (`interface/abench_stage.py`); everything from Python start-up to the last process exit counts, including compilation, data loading and saving `model.pt`.
- The evaluation stage runs in another process on one GPU with only the `out/` directory; it must finish within 20 minutes.
- A submission is valid only if every threshold holds; valid submissions are ranked by training time. Differences inside the 95% interval reported with the score are treated as ties.
- If any run crashes or exceeds a time limit, the whole submission is invalid.
- The final ABench score is driven mainly by training time, and also considers how different the recipe is from existing speedrun records and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your training program from scratch through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware, because the training itself is what this case measures.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 4 × H100 80GB in one node (NVLink); the evaluation uses the same node type. Available software: Python 3, PyTorch >= 2.7 (CUDA, Triton, FlexAttention, torch.distributed / torchrun, torch.compile), NumPy; no packages fetched at run time (the FlashAttention 3 kernel that later records download from the Hugging Face kernel hub is not available); your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 24 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 96 H100-hours (24 hours of the 4-GPU node); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── train.py            # required: training entry point (torchrun)
├── model.py            # required: build_model(), optional EVAL_SEQ_LEN
├── src/                # optional: helper modules
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 400,000 bytes in total; no trained parameters are submitted.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- No pretrained weights, distillation, external data or use of the validation shard; the model must be trained by `train.py` within the timed run.
- `model.py` must compute an honest probability model: logits from the trained weights for the given tokens only, with no access to later tokens of a window and no special handling of the evaluation.
- No processes may outlive `train.py`, and no work may be moved before the launch (for example compiled artifacts cached outside the run directory).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

# PRO-MOE-003: Pretraining a mixture-of-experts language model from scratch on 4 × H100 in 24 hours: lowest FineWeb loss at a fixed compute per token

Anchor: Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen and others, *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (arXiv:2401.06066). Baseline used for comparison: a DeepSeekMoE-style decoder (1 shared and 64 fine-grained routed experts, top-6, expert-level balance loss; about 1.0B stored and 0.17B used parameters per token) trained for 18,000 steps of 512 x 1,024 tokens (the reference submission; its code is shown under `interface/baseline/`); the operator also runs the auxiliary-loss-free balancing recipe (Wang et al. 2024), two seeds each.

## Task

Design a **mixture-of-experts language model and its pretraining recipe** — expert granularity and count, shared experts, routing and load balancing, capacity and token dropping, where and how experts are placed, optimiser and schedule, parallelism and kernels for the four GPUs — that, trained **from scratch** within one 24-hour session on **4 × H100**, reaches the **lowest held-out FineWeb loss** while using at most **0.8 GFLOPs of forward compute per token** and storing at most **2.5 billion parameters**.

- You develop, train and deliver within the session. Every training run that may lead to the delivery runs under the case's supervisor (`interface/pro_audit.py`), which registers a hash of a resumable checkpoint at least every 30 minutes; the delivered checkpoint must be one of them, reached from random initialisation.
- The evaluator rebuilds your model with your `model.py`, loads the delivered state dict, counts its forward FLOPs per token on held-out text (the routing it actually performs) and measures the held-out loss; it then resumes your run from one registered checkpoint for an hour and compares the logged training loss.
- Data: the 103 public FineWeb-10B shards only (as many passes as you like); no pretrained weights, distillation or other data.
- You may build on any public work (GShard, Switch Transformer, ST-MoE, DeepSeekMoE, auxiliary-loss-free balancing, DeepSeek-V3, OLMoE, Mixtral, MegaBlocks, expert-choice routing, Muon), but the design report must state what you borrowed and what you changed and on which attempt the delivered checkpoint was obtained.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only) holds the 103 training shards of FineWeb-10B pre-tokenized with the GPT-2 BPE (Hugging Face `kjj0/fineweb10B-gpt2` @889765ea, MIT; FineWeb text ODC-BY), as in PRO-SPD-004: `fineweb_train_000001.bin` … `fineweb_train_000103.bin`, read by `interface/token_data.py`.

The held-out set is the first 10,485,760 predicted tokens of the official validation shard `fineweb_val_000000.bin`, not in the sandbox, and a secret choice of the resumed checkpoint.

## Interface

Deliver `model.py` with `build_model()` (and optionally `EVAL_SEQ_LEN`, default 1,024) returning the architecture whose `load_state_dict` accepts `checkpoint/model.pt` and whose call `model(idx)` returns logits (B, T, 50257); `checkpoint/` must be identical to `model/` of a registered checkpoint. Deliver the code of the registered run (re-run from the submitted directory with `ABENCH_RESUME_FROM` set); it prints `ABENCH step=<n> loss=<float>` lines.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Held-out loss** (primary) | Mean next-token cross-entropy (nats) on the 10,485,760 held-out tokens in consecutive windows of `EVAL_SEQ_LEN`; lower is better | — |
| Compute per token | Forward FLOPs per token of the delivered model on 8 held-out windows of 1,024 tokens, counted by `torch.utils.flop_counter` | Threshold: ≤ 0.8 × 10^9 |
| Stored size | Elements of all tensors in `checkpoint/model.pt` | Threshold: ≤ 2,500,000,000 |
| Process integrity | Intact hash chain without gaps over 35 minutes; delivered checkpoint registered and reached from random initialisation; submitted code equal to the run's | Threshold: all hold |
| Sampled resume | Mean logged training loss of a one-hour resume from a randomly chosen registered checkpoint, against the original log over the same steps | Threshold: within max(1%, 3 standard errors) |
| Engineering | Throughput, MFU and loss spikes from the supervisor's logs and telemetry | reported |

**Measurement and decision**

- One evaluation on one GPU (deterministic); the evaluation takes well under an hour.
- A submission is valid if every threshold holds and the evaluation finishes within its time limit; otherwise it is invalid.
- The held-out loss of the two reference seeds gives the noise level: a lead counts only above two of their standard deviations.
- The final ABench score is driven mainly by the held-out loss, and also considers how different the design is from DeepSeekMoE-style recipes and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 4 × H100 80GB in one node (NVLink); the evaluation uses the same node type. Available software: Python 3, PyTorch >= 2.7 (CUDA, Triton, torch.distributed, torch.compile), NumPy; MegaBlocks and other kernels from the allowlisted pip mirror; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 24 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 96 H100-hours (24 hours of the 4-GPU node), development and final training included; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── checkpoint/         # required: model.pt (= model/ of a registered checkpoint)
├── model.py            # required: build_model(), optional EVAL_SEQ_LEN
├── train.py            # required: the registered run's program (and anything else it uses)
├── src/                # optional: other modules
├── README.md           # optional
├── report/index.html   # required: design report (states on which attempt the delivery was obtained)
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.sh`, `.yaml`, `.yml`, `.json`, `.toml`, `.md` and `.txt`, at most 1,000,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- No pretrained weights, distillation, external data or use of the validation shard; the delivered weights must come from registered runs started from random initialisation.
- `model.py` must compute an honest probability model: logits from the delivered weights for the given tokens only, with no access to later tokens of a window and no special handling of the evaluation or of the FLOPs count.
- Every run that leads to the delivery must be supervised; checkpoints may not be modified after registration.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

# PRO-SRV-005: Serving Qwen3-8B on 4 × H100: the highest request rate under a latency SLO on replayed production traffic, with exact greedy outputs

Anchor: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, *Efficient Memory Management for Large Language Model Serving with PagedAttention* (arXiv:2309.06180). Baseline used for comparison: vLLM with tensor parallelism over the 4 GPUs, chunked prefill and prefix caching (the reference submission; its code is shown under `interface/baseline/`); the operator also runs a tuned SGLang configuration.

## Task

Design a **serving system for Qwen3-8B** on one node with **4 × H100** — batching and scheduling, KV-cache management, prefill/decode splitting or disaggregation, parallelism (tensor, data, pipeline) and replica layout, kernels, CUDA graphs, admission and preemption — that sustains the **highest request rate** at which **90% of requests** meet a latency SLO (**time to first token ≤ 2 s, time per output token ≤ 100 ms**) on **replayed production traffic**, while producing the **greedy outputs of the model**.

- Within one 24-hour session you develop and deliver `serve.py`, which the evaluator starts on the node and loads with held-out windows of two production traces (conversation and code), replaying each window's arrival pattern at increasing rate factors until the SLO attainment falls below 90%. The whole evaluation is run twice and the median taken.
- Outputs must be greedy decoding of Qwen3-8B in bfloat16: a sample of answers is checked token by token against the evaluator's own forward pass, which rules out lossy quantisation, pruning or a different model. Speculative decoding with exact verification is allowed.
- Any open-source engine may be used and modified (vLLM, SGLang, TensorRT-LLM from the allowlisted mirror), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/dev.npz` (read-only): one window of 2,000 consecutive requests from each of the two Azure LLM inference traces of 2023 (Azure/AzurePublicDataset @79092101, CC-BY 4.0: arrival times, context and generated lengths), with prompts made of FineWeb training documents tokenised by Qwen3's tokenizer and cut to each request's context length (clipped to 16–8,192 tokens; generated lengths to 1–1,024): `<trace>_ids`, `<trace>_offsets`, `<trace>_arrival`, `<trace>_max_new`, and `check_idx`. `$ABENCH_MODELS_DIR/Qwen3-8B/` (@b968826d, Apache-2.0) holds the model.

The held-out set is one other window of 2,000 requests from each trace at a secret position, with prompts from FineWeb validation documents, and a secret sample of 100 requests per trace whose answers are checked.

## Interface

Deliver `serve.py`, started as `python3 serve.py --model MODEL_DIR --port PORT`; it must answer `GET /health` with 200 when ready and `POST /generate` with body `{input_ids, max_new_tokens, ignore_eos}` by streaming newline-delimited JSON `{"ids": [...]}` lines with the new token ids and a final `{"done": true}` (`interface/serve_bench.py`).

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Sustained rate** (primary) | For each trace, the trace's native request rate times the largest rate factor (1, 2, 4, 6, 8, 12, 16, 24, 32; interpolated linearly to the first factor that fails) at which at least 90% of requests have TTFT ≤ 2 s and TPOT ≤ 100 ms; mean of the two traces, median of the two evaluation runs; higher is better | — |
| Per trace | Largest rate factor, peak output tokens per second | reported |
| Greedy consistency | Share of checked answers (up to 256 tokens each) in which every token's logit is within 0.25 of the largest logit of the evaluator's forward pass | Threshold: ≥ 0.97 |
| Completeness | Checked requests answered with exactly the requested number of tokens | Threshold: all |
| Start-up | Seconds until `/health` answers | reported (limit 1,800 s) |

**Measurement and decision**

- The load generator runs on the same node (evaluator code); every request asks for exactly its trace length (`ignore_eos`). The evaluation is run twice from a cold start; the primary metric is the median.
- A submission is valid if the server becomes ready within 1,800 s in both runs, the replay finishes within 7,200 s and both thresholds hold in both runs; otherwise it is invalid.
- The two runs of each reference give the noise level: a lead counts only above two of their standard deviations.
- The final ABench score is driven mainly by the sustained rate, and also considers how different the system is from vLLM and SGLang defaults and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your code through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware; nothing is trained.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- 4 × H100 80GB in one node (NVLink); the evaluation uses the same node type. Available software: Python 3, PyTorch (CUDA, Triton), transformers, NumPy; vLLM, SGLang, FlashInfer and TensorRT-LLM from the allowlisted pip mirror; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run; blocked for this case because its data is distributed there: `github.com` (with its subdomains and `*.githubusercontent.com`).
- The session lasts at most 24 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 96 H100-hours (24 hours of the 4-GPU node); the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── serve.py            # required: the server entry point
├── src/                # optional: other modules, configurations and patches
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Files may only be `.py`, `.sh`, `.yaml`, `.yml`, `.json`, `.toml`, `.md`, `.txt`, `.cu`, `.cpp` and `.h`, at most 2,000,000 bytes in total; no model weights.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The server must use the provided Qwen3-8B weights in bfloat16 (or a lossless re-layout of them); no other model except a draft model trained in the session for speculative decoding with exact verification.
- The server must not see the evaluation in advance: no cached answers, no special handling of the load generator, and no work carried over between the two evaluation runs (each starts with empty caches).
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

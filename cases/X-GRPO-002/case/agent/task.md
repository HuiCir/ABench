# X-GRPO-002: A reinforcement-learning objective for verifiable rewards: post-training Qwen2.5-0.5B-Instruct on Countdown and GSM8K

Anchor: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (arXiv:2402.03300). Baseline used for comparison: GRPO with LoRA (rank 16): 8 sampled responses per prompt, group-standardised verifier rewards, one on-policy update per batch, 200 steps (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **reinforcement-learning method for verifiable rewards** — objective, advantage estimation, sampling and filtering of rollouts, curriculum, length control — that post-trains **Qwen2.5-0.5B-Instruct** in the sandbox to the **highest verified accuracy** on held-out **Countdown** equations and **GSM8K** word problems.

- Submit the trained model (a checkpoint, for example LoRA or full weights), the code that loads it and the training program that produced it. The evaluator formats every prompt, decodes greedily (at most 512 new tokens) and checks the single `<answer>` block of each response with the verifiers in `interface/verify_math.py`; it never retrains.
- Training data: generate as many Countdown instances as you like with the case's generator, and use the GSM8K training split; the held-out Countdown instances come from a secret seed, the held-out GSM8K problems are its test split.
- Rewards may only come from the provided verifiers or your own checks of the same answers; no other pretrained model (no reward model, no teacher), no external data.
- You may build on any public work (PPO, GRPO and its variants such as DAPO and Dr. GRPO, RLOO, rejection-sampling fine-tuning, expert iteration, length penalties, curricula), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/` (read-only):

| File | Contents |
|---|---|
| `gsm8k_train.json` | 7,173 GSM8K training problems (Cobbe et al. 2021, `openai/gsm8k` main @740312ad, MIT): `question`, `solution`, `gold` |
| `dev.json` | Self-check problems: 500 Countdown instances from a public seed and 300 GSM8K training problems set aside |

`$ABENCH_MODELS_DIR/Qwen2.5-0.5B-Instruct/` holds the base model (Apache-2.0, revision 7ae55760). `interface/verify_math.py` has the Countdown generator, the two prompt formats, the answer extraction and the verifiers.

The held-out set is 1,000 Countdown instances drawn with a secret seed (none identical to a dev instance) and the 1,319 GSM8K test problems; only the prompts are given to the model, after the timed phase.

## Interface

Implement `load(checkpoint_dir, models_dir, device)` in `model.py`, returning a transformers-compatible causal LM (`forward(input_ids, attention_mask)` with `.logits`, and `generate()`); the evaluator uses the base model's tokenizer and the ChatML template of `interface/lmpost.py`. Write a `train.py` that produces the checkpoint from the public data and the base model.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean accuracy** (primary) | Mean of the verified accuracy on the 1,000 held-out Countdown instances and on the 1,319 GSM8K test problems | — |
| Accuracy per task | Countdown: the answer is an equation using at least two of the given numbers, each at most once, only + − × ÷ and parentheses, equal to the target; GSM8K: the answer equals the final number | reported |
| Format violations | Share of responses without exactly one `<answer>` block (they score 0) | Threshold: ≤ 0.5 |
| Reward-hacking signs | Responses with several answer blocks, truncated answers, mean response length | reported |
| Size | Numbers stored in the checkpoint | reported |

**Measurement and decision**

- One evaluation run (greedy decoding is deterministic); loading is killed at 600 s and all responses must be generated within 1,800 s on one GPU.
- A submission is valid if it loads, answers within the limits and its format-violation rate is at most 0.5; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over problems; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by mean accuracy, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), transformers, safetensors, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 10 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 8 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load()
├── train.py            # required: produces checkpoint/ from the public data and the base model
├── checkpoint/         # required: trained weights or adapters, at most 2 GB
├── src/                # optional: other .py modules
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- Outside `checkpoint/`, files may only be `.py`, `.yaml`, `.yml`, `.md` and `.txt`, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The checkpoint must be trained from the provided base model in the sandbox; no other pretrained model, no external data, no network.
- The loaded model must answer from the prompt alone: no lookup tables of problems or answers inside the checkpoint or code.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

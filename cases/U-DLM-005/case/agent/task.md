# U-DLM-005: A faster sampler for a frozen masked diffusion language model: text quality at 16, 32 and 64 network evaluations

Anchor: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li, *Large Language Diffusion Models* (arXiv:2502.09992). Baseline used for comparison: MDLM ancestral sampling with equal time steps, run by the evaluator under identical conditions (`interface/baselines/ancestral/`).

## Task

Design a **sampler** — unmasking order and schedule, remasking, temperature and confidence rules, caching, corrector steps, anything that only queries the network — for a **frozen masked diffusion language model** (MDLM, 110M parameters, trained on OpenWebText) that produces the **most natural text with only 16, 32 or 64 network evaluations** per batch, both from scratch and continuing held-out prefixes.

- Submit code only (`sampler.py`). The evaluator calls `sample()` with a counting denoiser that refuses calls beyond the budget, at NFE 16, 32 and 64: 256 unconditional samples of 1,024 tokens and 256 continuations of 256 tokens after held-out 256-token prefixes per budget, twice with seeds you do not know. It runs the ancestral baseline the same way.
- Quality is the generative perplexity of the samples under GPT-2 Large, which is not in the sandbox; text must also keep at least 90% of the token diversity (unigram entropy) of real text, so repetition cannot buy a low perplexity.
- The sampler may only query the denoiser: no other model, no retraining, no access to the scorer.
- You may build on any public work (ancestral and τ-leaping samplers, confidence-based and low-confidence remasking, MaskGIT schedules, predictor-corrector and ReMDM-style remasking, caching across steps, semi-autoregressive blocks), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_MODELS_DIR/mdlm-owt/` holds the frozen MDLM (Sahoo et al. 2024, `kuleshov-group/mdlm-owt` @d0958fa8, Apache-2.0; GPT-2 vocabulary plus mask token 50257; loaded with `trust_remote_code`), and `$ABENCH_MODELS_DIR/gpt2/` GPT-2 small (`openai-community/gpt2` @607a30d7, MIT), which scores samples in the self-check only.

`$ABENCH_DATA_DIR/dev_text.npz` (read-only): `windows`, 64 windows of 1,024 GPT-2 tokens, each the start of a FineWeb document of at least 1,024 tokens (`kjj0/fineweb10B-gpt2` @889765ea train shard 1, MIT; text ODC-BY).

The held-out set is 256 windows drawn the same way from the validation shard with a secret seed: their first 256 tokens are the prefixes, the windows the references for entropy.

## Interface

Implement `sample(denoiser, prefix, length, nfe, gen) -> LongTensor (B, length)` in `sampler.py`. `denoiser(x)` maps a (B, L) LongTensor with mask token 50257 to log-probabilities (B, L, 50257) of the clean tokens (unmasked positions one-hot) and counts one evaluation per call; `prefix` is a (B, 256) LongTensor to keep unchanged, or None; `gen` carries `gen.torch` (a seeded torch.Generator), `gen.device` and `gen.batch` (B when prefix is None). See `interface/dlm.py`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean log generative perplexity** (primary) | Mean over the six settings (NFE 16, 32, 64; unconditional and continuation) of the mean per-token negative log-likelihood of the generated tokens under GPT-2 Large (continuations conditioned on their prefix), averaged over the two seeds; lower is better | — |
| Entropy | Mean unigram entropy of the generated tokens per sample, divided by that of the held-out real text of the same length (1,024 tokens; 256 for continuations); the smallest ratio over the six settings | Threshold: ≥ 0.9 |
| Per setting | Generative perplexity and entropy for each NFE and mode; wall-clock seconds per budget; the real text's perplexity under the scorer | reported |
| Relative to ancestral | Each perplexity divided by the ancestral sampler's in the same run | reported |

**Measurement and decision**

- A submission is valid if it returns fully unmasked samples of the right shape with prefixes unchanged, never exceeds the budget, finishes within 3,000 s per run and meets the entropy threshold; otherwise it is invalid.
- The primary metric is reported with a 95% bootstrap interval over samples; submissions whose intervals overlap are tied on it.
- The final ABench score is driven mainly by the mean log generative perplexity, and also considers how different the sampler is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then runs your code through the stages in `interface/stages.py`, once for each of 2 seeds unknown to you, on the evaluation hardware; nothing is trained.
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Under exactly the same conditions (model, budgets, seeds, prefixes), the evaluator also runs the ancestral sampler in `interface/baselines/ancestral/` and reports your perplexities relative to it.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), transformers, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 5 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 3 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── sampler.py          # required: sample()
├── src/                # optional: helper modules, .py only
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- `sample()` may only use the denoiser it is given and the generator; it must not load models or files, keep state between calls, or look at the prefix of another batch.
- Randomness must come from `gen.torch` so that runs are reproducible.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.

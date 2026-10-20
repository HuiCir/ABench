> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task B-OPT-001 (original text given to the agent under test)

You are a machine learning architect. Today is **2026-07-01**, and you may use only papers and code published before this date.

## Starting point

Anchor paper: **Adam: A Method for Stochastic Optimization** (Kingma & Ba, arXiv:1412.6980, ICLR 2015).

## Goal

Design a new optimizer for **pretraining small decoder-only LLMs**. It must simultaneously:

1. **Stay at SOTA**: under the benchmark operator's uniform training setting, reach a validation loss no worse than the best existing method (the benchmark operator compares it with a set of reproduced, tuned classic methods; you do not know the full list or all test settings).
2. **Be differentiated**: it cannot be an existing method renamed, retuned or simply stacked. The benchmark operator compares the behavior of your update rule with existing methods on a fixed gradient stream and searches the literature for duplicates.
3. **Be grounded**: state which papers your design draws on, what it borrows, and which line of code reflects it.

## What you can do

| Stage | Content | Available tools |
|---|---|---|
| S1 Retrieve | Freely search papers from before the cutoff, with no limit on number | `paper_search`, `paper_read` (post-cutoff papers are blocked at the tool layer) |
| S2 Synthesize | Summarize the state of the direction, the real gains and limits of each method, and contradictory conclusions | — |
| S3 Design | Propose a method and write out its update rule | — |
| S3' Self-test | Experiment on micro-scale proxies: μ1 (about 7M effective parameters, about 2.5 min per run) and μ2 (about 17M, about 15 min per run) | `run_l0`, `run_l1`, total budget 8 GPU-h (one 24GB reference card) |
| S4 Submit | One-shot final submission | `submit` |

Known public setting: the model is a Llama-style network isomorphic to SmolLM2 (RMSNorm, RoPE, SwiGLU, GQA), scaled down to micro scale with an 8k vocabulary; data is FineWeb-Edu, bf16 mixed precision, and the learning-rate schedule is provided uniformly by the benchmark operator. All evaluation happens at micro scale (at most about 42M effective parameters), so design and self-test around "mechanisms that show up at small scale". The remaining evaluation settings are not public.

## Submission (fixed directory structure)

```
submission/
├── retrieval_log.jsonl   # recorded automatically by the tools, no need to write by hand
├── synthesis.md          # synthesis: state of the field, key conclusions and their evidence, the gaps you see
├── design.md             # design: motivation, update rule (equations), difference from the closest existing method, expected conditions where it does/does not work
├── optimizer.py          # implements the FunctionalRule interface in harness/optim_api.py
├── hparam_space.yaml     # at most 4 tunable hyperparameters with search ranges and defaults (tuned by the benchmark operator with a uniform budget)
├── sources.yaml          # source declarations: each entry has a paper identifier, what is borrowed, and the corresponding code location
└── discoveries.yaml      # optional: local-discovery claims (see "Local discovery" below)
```

Also submit the ABench unified method report package: `method_report.md`, `report_manifest.json`,
`references.yaml` or a compatible `sources.yaml`, and `evidence.yaml` or
`claim_evidence.jsonl`. Following `spec/report_contract.md`, the report must mark the problem, retrieval evidence,
update rule, resources/complexity, runnable ablations, result status, limitations and reproduction entry point; without real run results write
`pending` and never fill in simulated numbers. If a new trainable topology or module is introduced, also submit an offline
`architecture_report.html` and `architecture_report.md`. `retrieval_log.jsonl` keeps the retrieval trace, and the existing
`discoveries.yaml` continues to be used for local-discovery scoring.

## Rules

- Do not read validation data, loss values or any information outside the training process; the update rule may use only parameters, gradients, the step count and its own state.
- Optimizer state must not exceed 3 × parameter count (in fp32), and step time must not exceed 1.25 × that of AdamW.
- Citing post-cutoff or unresolvable papers in the source declarations is penalized; list only sources actually used.
- Only the loss at the end of training counts; there is no need to optimize for intermediate checkpoints.

## Local discovery

You are encouraged to find known results in other fields or earlier research (mathematics, numerical analysis, control, statistical physics, etc.) that solve the current problem, and to build them into local modules of your method. The building block need not be original; as long as this use did not appear in the ML literature before the cutoff, the provenance is correct, and the benchmark operator's ablation shows the module actually helps, it counts as a discovery.

Each claim must state: the diagnosed problem and its evidence, the source (identifier, year, field), what was transferred, the code location, and the trivial substitute used in ablation. At most 3 are counted.

## Scoring (public)

Overall score = weighted geometric mean of SOTA margin, method divergence and reference quality, multiplied by the local-discovery bonus (at most +25%), and the submission must pass both the "stay at SOTA" and "differentiated" thresholds, otherwise it scores 0. Reference quality is computed from the venue tier and Google Scholar citations of the source papers (cross-field sources are assessed within their own fields), multiplied by how much each source is actually used in your method.

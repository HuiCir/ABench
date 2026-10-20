> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task B-OPT-002 (original text given to the agent under test)

You are a machine learning architect. Today is **2026-07-01**; you may use only papers and code published before this date.

## Starting point

Anchor paper: **Symbolic Discovery of Optimization Algorithms** (Lion, arXiv:2302.06675, NeurIPS 2023). Lion illustrates an important route: search optimizer programs within a frozen update grammar on small proxies, then transfer the candidates to language-model training.

## Goal

Starting from the frozen symbolic primitives below, design and implement an **auditable program-search process** that finds a new optimizer for small-scale LLM pretraining. It must:

1. reach quality no worse than Lion on the fixed gradient stream and the μ1 proxy;
2. stay at SOTA when final evaluation reuses the shared reference envelope of B-OPT-001;
3. be behaviorally distinguishable from existing rules such as Lion and AdamW;
4. record candidate programs, evaluation counts, proxy scores and random seeds so that the search can be replayed deterministically.

## Frozen search space

Candidate programs may use only the following states and primitives; they may not execute arbitrary Python or read validation loss:

- State: `m_fast`, `m_slow`, each the same shape as the parameter; total state no more than 3× the parameter count;
- EMA: `ema(m, g, beta)` (the recurrence is fixed by the harness; the candidate AST only describes the update expression after the recurrence);
- Unary primitives: `identity`, `sign`, `abs`, `sqrt`, `normalize_rms`;
- Binary primitives: `add`, `sub`, `mul`, `safe_div`, `min`, `max`;
- Constants: `beta_fast`, `beta_slow`, `alpha`, `tau`; `weight_decay` is added by the harness as a fixed outer term and is not a search variable;
- The update must have the form `-lr * (update + weight_decay * p)` and may not carry its own learning-rate schedule.

Search evaluation must run on the fixed gradient stream provided by the benchmark: first compare the direction/spectral proxy, then confirm with independent short μ1 runs. At most 256 candidates, with the search seed fixed at 17; hidden test feedback must not be used for tuning.

## Deliverables

```
submission/
├── retrieval_log.jsonl
├── synthesis.md
├── design.md
├── optimizer.py
├── hparam_space.yaml
├── sources.yaml
├── discoveries.yaml
├── search_log.jsonl       # one line per candidate: program_id, AST, proxy score, seed, time
└── search_spec.yaml       # grammar, budget, target and replay information
```

Also submit the ABench unified method report package: `method_report.md`, `report_manifest.json`,
`references.yaml` or a compatible `sources.yaml`, and `evidence.yaml` or `claim_evidence.jsonl`. The report must show clearly
the frozen primitives, search space, candidate data flow, resource budget, deterministic replay, final candidate and ablations; the retrieval trace is still written to
`retrieval_log.jsonl`, and the search log or self-reported scores must not be treated as server-side final results. If a new trainable topology or module is introduced, also submit an offline
`architecture_report.html` and `architecture_report.md`.

`optimizer.py` must implement the `FunctionalRule` interface in `harness/optim_api.py` and expose `RULE`. Before submission it must pass L0: interface, state cap, 1000-step ill-conditioned quadratic, determinism.

## Scoring

Final evaluation still uses the SOTA margin M, divergence D, reference quality Q and local discovery X of B-OPT-001; in addition, primitive-search efficiency E (evaluations to reach the target proxy quality and the curve AUC) is reported. If the search log is missing, cannot be replayed, or candidates go outside the frozen grammar, the search threshold fails.

# ABench Case B-OPT-002: Searching optimizers from symbolic primitives

This is the second bench specification after B-OPT-001. It reuses the same `FunctionalRule`, reference pool and final-evaluation scoring, but turns the task into **Lion-style primitive program search**: the agent may enumerate candidates only from frozen EMA states, symbolic operations and a finite set of constants, and must submit an auditable, replayable search trace.

- Corpus cutoff: 2026-07-01
- Anchor: Lion / *Symbolic Discovery of Optimization Algorithms* (arXiv:2302.06675)
- Subset: primitive; difficulty high; resource tier high
- Development budget: fixed gradient stream + μ1, 8 GPU-h in total; submission verification about 5.5 GPU-h; reference precompute 0 (plans to reuse the real reference package of B-OPT-001)

## Relation to B-OPT-001

B-OPT-001 asks for a new optimizer to be designed directly; B-OPT-002 turns "propose an update rule" into an auditable program-search problem. Final evaluation still uses the 4 settings, reference envelope, behavioral divergence and source-quality scoring of B-OPT-001. The added search layer checks grammar compliance, log continuity, deterministic replay, the fixed budget, and consistency between the candidate programs and the submitted rule.

## Current status

This directory delivers only the bench specification, harness, example submission and scoring entry point, with no fabricated training curves, reference results or LLM judge results. `example_submission/search_log.jsonl` is a replayable search example on a frozen proxy and does not represent μ1/μ2 training results. Before formal runs, resource snapshots, real reference curves, a real gradient stream, the hidden set and hardware timing must be added.

## Directory

- `case/`: Lion anchor card, frozen grammar, budget, anti-gaming rules and evaluation protocol.
- `harness/`: numpy/torch functional interface compatible with B-OPT-001, and the safe candidate evaluator `primitive_search.py`.
- `example_submission/`: RidgeSign example method, search spec, search log, arXiv HTML design report and source declarations.
- `scoring/`: generic M/D/Q/X scoring, `primitive_score.py` search-trace verification and the real-result report entry point.

Real evaluation runs once the benchmark operator provides the run-result files:

```bash
cd cases/B-OPT-002
python3 scoring/run_case.py \
  --submission example_submission \
  --runs /path/to/real_runs.json \
  --judge /path/to/real_judge.json \
  --out /path/to/report
```

There are currently no real scores. All conclusions should rest on real reference runs, hidden evaluation and reproduction audits.

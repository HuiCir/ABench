# B-OPT-001 three-dimensional scoring demo (predates the standard case)

> Kept from the earlier version of this case's README. It documents the ABench-level scoring design (SOTA margin M, divergence D, reference quality Q) and the simulated end-to-end demo in `sim/` and `outputs/`; those outputs are simulated and are not results. The case's judge is now the standard `harness` pipeline described in README.md.


ABench does not test "reproducing past papers". It asks the agent to work like an architect: **start from a classic paper → retrieve related research openly, throughout the work → synthesize → design a method that is better than and different from existing ones**, which the benchmark operator then evaluates under a uniform setting. This directory is a complete bench skeleton; real resources, reference runs and hidden evaluation still have to be frozen by the benchmark operator. Track B (optimizers), anchor Adam.

- Literature: any date (no cutoff since 2026-10-23; novelty is judged against the literature at session start)
- Verification scale: reduced to a micro-scale ladder per `abench/spec/compute_budget.md` (about 7M / 17M / 42M effective parameters), with one 24GB GPU as the reference card; agent development 8 GPU-h, about 6 GPU-h per submission verification, about 74 GPU-h of reference precompute
- Model structures and datasets are given only as Hugging Face links, **with no download and no training** (`case/resources.yaml`)
- Training results and LLM judge outputs are currently **simulated values** and cannot be used as real leaderboard results; the behavioral probes, code diff, paper weights and scoring logic are only a local process demo

## 1. Task flow

```
S0 Issue        anchor card + task statement (case/anchor_card.yaml, case/task_prompt.md)
 │
S1 Retrieve     the agent retrieves openly with paper_search / paper_read; literature of any date, at any point of the work; everything is logged
 │             → submission/retrieval_log.jsonl
S2 Synthesize   lineage of the field, key conclusions with evidence strength, gaps        → submission/synthesis.md
 │
S3 Design       update rule + differences from the closest existing method + source declarations → report/index.html / optimizer.py / references.yaml
 │   └─ Self-test   L0 automatic checks, μ1/μ2 micro-scale training (8 GPU-h budget)
S4 Submit       one-shot submission; the benchmark operator tunes hyperparameters with the same budget and runs the public μ2 setting + hidden μ1/μ2/μ3 settings
 │
S5 Score        ① SOTA margin M   ② method divergence D   ③ reference quality Q   → thresholds + overall score
 │
S6 Report       outputs/<submission>/score_report.md
```

## 2. Three-dimensional scoring

**① SOTA margin M** (`scoring/sota_margin.py`)
Different agents retrieve different papers, so the comparison is not against opponents the agent found itself, but against the performance of 10 classic methods provided by the benchmark (AdamW, Adafactor, Lion, Sophia, Schedule-Free, AdEMAMix, SOAP, MARS, C-AdamW, Muon; see `case/reference_pool.yaml`) in the given settings:

- Reference methods use a WSD schedule with a decay branch at each of {0.5, 0.7, 1, 1.4, 2}× budget, giving a "budget–final loss" curve; taking the minimum at each cost point forms the **reference envelope**
- Cost = tokens × step-time ratio, so a slow method cannot profit from token efficiency
- Effective speedup S = cost at which the envelope reaches the submission's final loss / the submission's own cost; weighted geometric mean over 4 settings (L2 + 3 hidden L3)
- Thresholds (staying at SOTA): aggregate S ≥ 1, seed lower confidence bound ≥ 0.98, every single setting ≥ 0.9, OOD degradation ≤ 1%, step time ≤ 1.25×
- M = 0.1 + 0.9 · clip(ln S / ln 1.25, 0, 1)

**② Method divergence D** (`scoring/divergence.py`) = 0.5 behavior + 0.15 code + 0.35 mechanism

- Behavior: run the submission and the reference methods side by side on a fixed gradient stream and compare per-step update directions and singular-value distributions. The global view detects "renamed/retuned clones", with its threshold calibrated by the **hyperparameter-perturbation noise floor**; the patch view detects "layer-wise splicing of existing methods"
- Code: AST 5-gram Jaccard, ignoring variable names, low weight
- Mechanism: 3 LLM judges grade 0–3 against the nearest neighbors in the hidden snapshot (`scoring/judge_rubric.md`)

**③ Reference quality Q** (`scoring/paper_weight.py`)

- Paper weight w = (1 − 0.6a)·V + 0.6a·C: V is the venue-tier score, C the percentile of annual Google Scholar citations within the snapshot cohort, a = min(1, age / 2 years); new papers are judged mainly by venue
- Q = 1 − Π(1 − w·u), where u is the usage level assigned by the judges (core 1 / partial 0.5 / decorative 0), counting only the top 8; each unresolvable source multiplies by 0.8

**④ Local discovery X** (`scoring/discovery.py`, defined in `abench/spec/discovery.md`)
Bringing a known result from another field or earlier research into an effective local module also counts as discovery (cf. DeepSeek mHC introducing the 1967 Sinkhorn–Knopp algorithm). Each claim is ablated separately by the benchmark operator and scored as effectiveness × application novelty × provenance correctness × provenance distance. The weight of a cross-field source is assessed within its own field.

**Overall score** = 100 · 1[thresholds] · M^0.5 · D^0.3 · Q^0.2 · (1 + 0.25·X)

## 3. Demo output (not a real evaluation)

| Submission | Role | SOTA | Differentiated | Aggregate speedup | M | D | Q | Overall |
|---|---|---|---|---|---|---|---|---|
| example_submission | Log-gated RankGate | ✓ | ✓ | 1.034× | 0.24 | 0.40 | 0.80 | **43.58** (with discovery bonus X = 0.94) |
| clone_muon | Negative: renamed Muon clone | ✓ | ✗ | 1.011× | 0.14 | 0.16 | 0.99 | **0** |
| rankgate_linear_gate | Negative: early linear-gate version | ✗ | ✗ | 0.776× | 0.10 | 0.35 | 0.80 | **0** |

The example submission claims two local discoveries: stable-rank gating (source: Rudelson & Vershynin 2007) is judged a local discovery; power-iteration estimation of the spectral norm (von Mises 1929) is effective in ablation, but ML has used it the same way before, so it is judged retrieval reuse and earns no discovery score. The linear-gate version claims the same stable-rank gating, but performs better once it is ablated, so it is judged a decorative import.

Both negatives are stopped: the Muon clone differs from Muon behaviorally by exactly the noise floor of "changing a momentum coefficient"; the gate value of the linear-gate version is measured to stay around 0.06, and its behavior is 0.97 similar to Adafactor, a design flaw the probes exposed (the example submission's design.md records the switch to log gating for this reason).

## 4. Directory

```
case/            anchor card, task statement, HF resources, reference pool, evaluation protocol, scoring config, anti-gaming checklist, seed paper snapshot
harness/         submission interface (numpy/torch backends), functional implementations of the reference methods, agent tool specs
example_submission/  example submission (RankGate design report, code, sources and hyperparameter space; no real training results)
scoring/         paper weights, SOTA margin, divergence, judge rubric, per-submission scoring, full demo
sim/             simulated training-result generator, simulated judge output, two negative controls
outputs/         score reports and summary
```

Run (needs only numpy and pyyaml):

```bash
python sim/make_mock_runs.py
python scoring/run_demo.py          # score all three submissions, writes outputs/summary.md
```

## 5. To do before real launch

1. **Verify**: arXiv IDs, venues and first-release dates in the reference pool and snapshot (compiled from memory, marked verify); the `fineweb-edu` config name (`sample-10BT`) and the subdirectory of `bigcode/the-stack-smol`
2. **Collect citations**: Google Scholar has no API, so citations must be scraped manually/semi-automatically and archived on the freeze date; `citations_demo` holds illustrative values
3. **Precompute reference results**: 10 methods × 4 micro-scale settings × 1–3 seeds × WSD branches (about 74 GPU-h), replacing `outputs/mock/`; check fidelity first (`eval_protocol.yaml#proxy_fidelity`)
4. **Publish the 8k BPE vocabulary** to HF
5. **Record a real gradient stream**: replace `synthetic_gradient_stream` with the 10%/50%/90% checkpoints of the μ2 AdamW reference run, and wire the official implementations of SOAP, Sophia and Schedule-Free into the probes (the probes currently cover only 7 methods)
6. **Calibrate**: backtest each coefficient on mature anchors (full-score speedup 1.25, exponents 0.5/0.3/0.2, noise-floor multiplier 1.5)
7. **torch wrapper**: `make_torch_optimizer` has not been executed in this environment; run L0 in the sandbox image

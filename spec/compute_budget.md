# ABench reduced-scale verification protocol (v2026.07)

## 1. Why reduce

An agent cannot validate an idea and train a model in a single submission. It must iterate during development, and each idea often needs dozens of runs. So the real cost of each case is:

```
total cost = agent development runs × number of agents tested + benchmark verification × number of submissions + reference-method precompute (one-off)
```

If any of these terms keeps the paper's original scale (full data, original model size, multi-fold full training), multiplying by the number of agents and iterations makes it unaffordable. This protocol states: **no case trains at the paper's full or large scale**; instead each case is verified within a fixed budget cap using a proxy whose fidelity has been checked.

## 2. Reference hardware and budget caps

**Budget reference card**: one GPU with 24GB of memory (RTX 4090 class). Every Budget case must be completable on this card, and budgets and wall-clock metrics (step time, speedup, latency) are defined on it. Runs on another GPU model, for Budget and for Pro (still an exclusive 4-GPU node), are scored; the GPU model of the agent and the judge is recorded and reported next to the result, so results from different cards can be told apart. `budget_tier: pro` is an explicit exception that is included in the default `all` suite and reported separately by resource class (`spec/pro_set.md`); the runner must not interpret it as "24GB by default, switched conditionally". This also unifies the two earlier conflicting rules of "24GB admission" and "H100 evaluation".

Machine-readable protocols must write the two resource classes as separate fields: `budget_tier: budget|pro`, `hardware.reference`, `wall_clock_limit_h` and `leaderboard` must appear together; if any is missing, preflight must refuse to start rather than infer the hardware from strings or environment variables.

| Resource tier | Agent development budget | Verification per submission (incl. local-discovery ablations) | Reference-method precompute (per case, one-off) |
|---|---|---|---|
| low | ≤ 2 GPU-h (or CPU only) | ≤ 0.5 GPU-h | ≤ 10 GPU-h |
| medium | ≤ 4 GPU-h | ≤ 2 GPU-h | ≤ 30 GPU-h |
| high | ≤ 8 GPU-h | ≤ 6 GPU-h | ≤ 80 GPU-h |

high is the hard cap of the Budget edition; there is no higher tier. A case that does not fit in high has only three options: change the proxy or postpone it to v2 (judgment tasks, T7, were retired on 2026-10-22); tasks that are genuinely industrial essentials go into the Pro resource group in the all suite (resource tier pro, see `pro_set.md`).

**Difficulty level** is independent of the resource tier:

| Difficulty | Meaning |
|---|---|
| low | Entry level and calibration: mature anchor, clear strong baseline; a strong agent should score close to full marks |
| medium | Main evaluation on mature anchors: requires retrieval, synthesis and beating tuned existing methods |
| high | Frontier anchors, non-mainstream paradigms, primitive search, stress/counterfactual, cross-domain transfer and the Pro Set |

## 3. Proxy toolbox

Each case chooses one or more verification methods from the table below and writes them to `verify_plan.proxy` in `index.yaml`.

| Code | Method | Applies to | Example |
|---|---|---|---|
| MICRO | Micro-scale training: 3 sizes (about 7M / 17M / 42M effective parameters), 20 tokens/parameter, 8k vocabulary | Architectures, optimizers, training recipes | B-OPT-001 |
| SYN | Synthetic mechanism probes: MQAR recall, copying, induction heads, length extrapolation, algorithmic tasks | Sequence mixers, recurrent depth, long context | A-HYB-001, U-LOOP-003 |
| SURG | Surgery on pretrained models: take a small HF model (≤ 360M), replace the target module, then short continued training or distillation (≤ 50M tokens) | Components whose differences only show on already-trained representations | A-KV-004, U-TTT-008 |
| FROZEN | Frozen backbone + lightweight head: train only the downstream head or adapter | Biology, medicine, embodied AI and other fields where pretraining is very expensive | BI-PROT-001, E-VLA-003 |
| INFER | Inference only: weights fixed; change only sampling, decoding, quantization or caching | Inference acceleration, diffusion sampling, diffusion-LLM decoding | C-SPEC-001, U-DLM-005 |
| STREAM | Fixed-stream component test: unit-test a component on recorded gradient streams, activation streams or operator workloads | Optimizer behavioral probes, kernels | K-KB-001 |
| SUBSET | Subsetting: a few subtasks, fewer folds, fewer epochs, lower resolution | Medical segmentation, PDEs, visual generation | MD-SEG-001 |
| JUDGE | Judgment: given a published claim, judge whether it holds (retired 2026-10-22 with the T7 cases; claims are now tested through tuned baselines inside design cases) | — | — |

## 4. Proxy fidelity check (the precondition for trusting reduced-scale verification)

The risk of scaling down is that conclusions flip: a method that is better at small scale is not necessarily better at large scale. So before each track goes live, a fidelity check is required:

1. Choose at least 5 pairs of methods from the calibration set with **existing large-scale published conclusions** (for example Muon vs AdamW, SwiGLU vs GELU, RoPE vs learned position embeddings, nnU-Net vs some "new SOTA").
2. Reproduce their relative ranking on the track's proxy ladder.
3. The track goes live only if pairwise agreement is ≥ 80%; tracks at 60%–80% are marked "low fidelity" in reports, with the SOTA margin for reference only; tracks below 60% turn the affected cases into JUDGE tasks or postpone them.

Fidelity results are published with each version, as a statement of how much "micro-scale conclusions" can be trusted.

## 5. Metering during agent development

- During development only L0 checks and μ1/μ2-level proxies can be run; the tool layer deducts reference-card GPU-h from the budget and stops when it runs out.
- Standard cases: `tools/sandbox_runner.py` meters GPU-busy time (nvidia-smi, 5 s samples) and the CPU time of the agent's processes, records both in `agent_run.json`, and marks a session that exceeds its `agent_budget` by more than 5% as not scorable. Single-GPU budgets never exceed the session length.
- The benchmark operator's verification scales (μ3, hidden settings) are not available to the agent.
- Each case allows only one final submission.

## 6. Outer validation (unscored)

Each version cycle, the benchmark operator picks the top 3 submissions of each case and runs one extra validation at 10× scale. The results do not change the scores of the period and are used only for two things: updating the track's fidelity statistics, and marking "reviewed at larger scale" in reports. Large compute is spent only on the few methods that merit it and does not grow with the number of submissions.

## 7. Micro-scale ladder of B-OPT-001 (example)

The 8k BPE vocabulary is trained once by the benchmark operator on FineWeb-Edu and released on HF. Effective compute of the reference card is estimated at 4×10¹³ FLOP/s.

| Tier | Structure | Effective params | Tokens | Time per run | Use |
|---|---|---|---|---|---|
| μ1 | d=256, 6 layers | about 7M | 0.14B | about 2.5 min | agent development, benchmark tuning (16 trials), long-training variant with 8× data |
| μ2 | d=384, 8 layers | about 17M | 0.35B | about 15 min | public verification (3 seeds), 4× batch-size variant, local-discovery ablations (2 seeds each) |
| μ3 | d=512, 12 layers | about 42M | 0.84B | about 1.5 h | hidden scale-extrapolation point (1 seed) |

Verification of one submission takes about 6 GPU-h (the high-tier cap), and precomputing the 10 reference methods takes about 74 GPU-h (μ3 reference curves only go to 1.4× budget). The agent's 8 GPU-h development budget is about 190 μ1 runs or 30 μ2 runs.

# ABench direction coverage and open-issue audit

This audit is based on the current **98 scaffolds**, the task cards of each track, resource profiles and reference pools. Directory-level coverage is already enough to support a first-version split into architecture/training/systems/domains; the bottleneck now is resource freezing, real runners, proxy fidelity and hidden splits, not adding more directions.

Added or narrowed in this round: `T-AUDIO-005` small offline ASR adaptation, `X-DISTILL-010` classic knowledge distillation, and the AWQ/GPTQ/SmoothQuant quantization Pareto of `C-QNT-003`; the earlier SFT, DPO, data recipe, compile, lightweight image-text, fixed small-repository code and reliability cases are kept. Full static preflight gives `cases=98 missing=0 errors=0 warnings=432`, and resource-profile preflight gives `pareto_cases=48 errors=0`. Anchor readiness is 52 `deferred` and 46 `contract_only`; no case has reached `reference_ready` or `hidden_ready`.

## Current scope boundary

- **Not expanded this round**: long context/long-term memory, video understanding or generation, large repository-level software engineering, OSWorld/WebArena/Computer Use, robot simulation, and tasks that need multiple machines or special environments.
- `A-LCTX-006` is kept for direction auditing but is marked `deferred_out_of_scope_v1` and gets no reference budget this round.
- Image generation, sampling, small 3D scenes and lightweight image-text adaptation in the vision track can remain offline method cases; they do not justify a claim of video coverage.
- `X-CODE-008` covers only fixed small Python repositories, compilation and unit tests; it cannot replace large repositories or end-to-end Computer Use.

## Directions still worth prioritizing

### P0: wire in real runners first

1. **Classic post-training chain**: `X-SFT-005` covers full-parameter/LoRA/QLoRA/DoRA, `X-DPO-006` covers DPO/IPO/KTO/ORPO, `X-DISTILL-010` covers hard-label/logit/feature/sequence KD. Only one more RM/RLHF/RLAIF or rejection-sampling case under a shared data and stability protocol is being considered; do not add more preference-optimization variants.
2. **System quantization and compilation**: `C-QNT-003` already specifies AWQ/GPTQ/SmoothQuant vs FP16, 2/3/4 bit, group size, p95 decode, memory and hidden shapes/tasks; `K-COMP-005` handles compile/Inductor/Triton. Next, prioritize real kernels and fixed-hardware measurement rather than new similar quantization cases.
3. **Minimal audio**: `T-AUDIO-005` uses only a frozen ≤100M wav2vec 2.0/HuBERT and short-audio CTC/adapter, with speaker/accent/noise hidden splits. No TTS, real-time streaming or large-scale speech pretraining for now; if resources grow, prefer a separate audio classification/enhancement case over another ASR case of the same kind.
4. **Reliability**: `X-ROBUST-009` already covers calibration, OOD, selective abstention and safety constraints. Prompt injection, jailbreaks, model extraction, privacy/federated learning and reward hacking remain potential gaps, but each needs its own threat model and they should not all be added at once.

### P1: add the classic methods of each direction to the reference pool

- **Architecture**: RoPE/ALiBi, MQA/GQA, MLA, RMSNorm/DeepNorm/QK-Norm, S4/Hyena/RWKV/RetNet, Switch/GShard/DeepSeek MoE. Long-context extrapolation is not an acceptance axis this round.
- **Optimizers**: beyond AdamW, Adafactor, Lion, Sophia, Schedule-Free, AdEMAMix, SOAP, MARS and Muon, the calibration set may keep SGD/Momentum, RMSProp, Adagrad, LAMB/LARS, AdaBelief, Yogi and 8-bit Adam, with unified state memory, weight decay, schedules and multiple seeds.
- **Systems and low precision**: AWQ/GPTQ/SmoothQuant, FP8/FP4, pruning, distillation, FlashAttention, torch.compile/Inductor, Triton, speculative decoding, KV paging and single-GPU serving schedulers. TensorRT-LLM, TVM, cross-GPU communication and power are recorded as future resource items; no new cases for them now.
- **Classic domain baselines**: DS keeps LightGBM/CatBoost, IC keeps ViT/ConvNeXt/MAE/DINO, TC keeps TF-IDF/SVM/NB-SVM/DeBERTa/ModernBERT, GN keeps GraphSAGE/GAT/GIN/PNA, RS keeps BPR/GRU4Rec/BERT4Rec/two-tower/DeepFM/DIN. They suit low-resource calibration and should not be packaged as new SOTA tracks.

## Open issues

1. **Status disconnected from real launch**: all 98 cases are `scaffold`; 52 `deferred` and 46 `contract_only`, with no real reference/hidden evidence.
2. **Placeholder resources and licenses**: models, data, tokenizers, audio front ends, teacher caches, quantization kernels, dependency images and evaluator versions still need frozen URIs, licenses and SHA256.
3. **No server-side timing**: p50/p95 latency, throughput, peak memory, OOM, timeouts, compile time, audio real-time factor, teacher-cache cost and quantization numerical equivalence must be measured.
4. **Hidden splits not in place**: audio needs speaker/accent/noise isolation; distillation needs capability-family, template and length isolation; quantization needs hidden models, shapes, low-batch decode and activation outliers.
5. **Pareto conventions need calibration**: quality floor, infeasible points, reference point, hypervolume, confidence intervals, tie-breaks, throughput/latency direction and GPU-h units must be pre-registered; when throughput is both a quality gain and a cost metric, how it is reported must be explicit.
6. **Proxy fidelity unknown**: every active track needs at least 5 known rankings, multi-seed bootstrap and a noise floor; when ranking agreement is insufficient it is downgraded to judge-based or postponed.
7. **Broken links in older resources**: 17 older cases still reference inputs awaiting release such as `paper_snapshot/seed_neighborhood.jsonl`; a shared preflight should resolve resources case by case instead of only checking static paths.
8. **Insufficient leak and gaming protection**: public CIFAR/GLUE/MovieLens/Cora/legal/medical/protein data may be memorized; time, patient, family, species, user/item or repository-level isolation is needed, and the runner must check input access, caches and undeclared CUDA/network use.
9. **Inconsistent scorer semantics**: some older `eval_protocol` files are still textual metric lists, and the generic runner only returns `scorable: false`; high-priority cases need an evaluator wired in one by one, keeping failure, OOM, timeout and uncertainty fields.
10. **Unclear meaning of date and version freezing**: `cutoff: 2026-07-01` in the index should be stated as a fixed data version rather than the current working date; paper, code and resource hashes must be locked together with the version.

## Recommended order

First do real smoke runs, reference-pool calibration and server-side timing for `C-QNT-003`, `T-AUDIO-005`, `X-DISTILL-010`, `X-SFT-005`, `X-DPO-006` and `K-COMP-005`; then wire in `C-SPEC-001`, `C-KV-002`, `K-FA-003` and the low-resource calibration sets of DS/IC/TC/GN/RS. Long context, video, large repositories, Computer Use, robot simulation and the Pro Set are paused and will be reviewed separately once the shared runner and budget are stable.

A case is promoted from `scaffold` to `proxy_ready` or `reference_ready` only after resource hashes, real reference runs, hidden splits, server-side timing, proxy fidelity and the anti-gaming audit all pass.

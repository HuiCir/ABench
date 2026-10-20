# ABench score report: B-OPT-001 / rankgate

> **Training results and judge outputs are simulated data**; this report only validates the pipeline and does not represent the method's real performance.

## Overall score: 0.0

| Threshold | Result |
|---|---|
| L0 | pass |
| SOTA | fail |
| Differentiated | fail |

Score = 100 · M^0.5 · D^0.3 · Q^0.2 · (1 + 0.25·X) = 100 · 0.1^0.5 · 0.3534^0.3 · 0.7999^0.2 · (1 + 0.25·0.0) = 22.14 × 1.000 = **22.14**

## L0 automatic checks

| Check | Result |
|---|---|
| interface | True |
| state_ratio_max | 2.0 |
| state_memory<=3x | True |
| finite | True |
| quadratic_loss_ratio | 0.0406 |
| descends | True |
| deterministic | True |
| pass | True |

## SOTA margin M = 0.1

Aggregate effective speedup (relative to the reference envelope): **0.7761×**, lower confidence bound 0.7716×

| Setting | Speedup | Per seed | Submission loss | Best reference at 1× budget | Its loss | Envelope-dominant method |
|---|---|---|---|---|---|---|
| L2_main | 0.7308 | [0.7265, 0.7381, 0.7278] | 4.252 | muon | 4.1522 | muon |
| L3a_scale | 0.8158 | [0.8143, 0.8173] | 3.7313 | muon | 3.6836 | muon |
| L3b_long | 0.84 | [0.8638, 0.817] | 4.21 | soap | 4.1686 | muon |
| L3c_bigbatch | 0.7283 | [0.7306, 0.7259] | 4.3176 | muon | 4.2133 | muon |

| SOTA threshold | Result |
|---|---|
| aggregate_speedup>=1.00 | fail |
| lcb>=0.98 | fail |
| per_setting>=0.90 | fail |
| ood_regression<=1% | fail |
| step_time<=1.25x | pass |

OOD: wikitext-103 degraded 0.21%; the-stack-smol/python degraded 1.02%

## Method divergence D = 0.3534

- Behavioral divergence D_beh = 0.0283 (global 0.0283, patch 0.0283), nearest reference: adafactor
- Hyperparameter-perturbation noise floor 0.0745 ({'adamw': 0.0345, 'cautious_adamw': 0.0203, 'lion': 0.0376, 'adafactor': 0.038, 'ademamix': 0.0318, 'mars': 0.0312, 'muon': 0.0745}), clone threshold 0.1117
- Code divergence D_code = 0.7063, nearest reference: adafactor
- Mechanism divergence D_mech = 0.6667 (judge levels [2, 1, 2]), closest prior work: Muon (2502.16982); the hidden snapshot also contains work that switches optimizers per layer, but all of it is static assignment, with no dynamic gating by the spectral structure of the momentum

| Tensor | Nearest reference | Similarity |
|---|---|---|
| layers.4.self_attn.q_proj | adafactor | 0.9727 |
| layers.4.mlp.up_proj | adafactor | 0.9629 |
| layers.20.self_attn.o_proj | adafactor | 0.9748 |
| layers.20.mlp.down_proj | adafactor | 0.9763 |

Mean behavioral similarity to each reference method: adafactor 0.9717, ademamix 0.9469, adamw 0.9468, mars 0.9422, cautious_adamw 0.9219, lion 0.8097, muon 0.5273

| Divergence threshold | Result |
|---|---|
| D>=0.20 | pass |
| D_global>=0.112 (noise floor 0.074 × 1.5) | fail |

## Reference quality Q = 0.7999 (before penalty 0.9999, penalty factor 0.8)

| Source | Status | Usage | Paper weight w | Contribution w·u |
|---|---|---|---|---|
| arXiv:2502.16982 | ok | core | 0.6021 | 0.6021 |
| muon_blog | ok | core | 0.5149 | 0.5149 |
| arXiv:1804.04235 | ok | core | 0.8579 | 0.8579 |
| arXiv:1412.6980 | ok | partial | 0.9842 | 0.4921 |
| arXiv:2509.02046 | ok | partial | 0.8499 | 0.4249 |
| arXiv:2407.07972 | ok | partial | 0.4234 | 0.2117 |
| arXiv:math/0503442 | ok | core | 0.9208 | 0.9208 |
| vonmises1929 | ok | core | 0.7092 | 0.7092 |
| arXiv:1904.00962 | ok | decorative | 0.9211 | 0.0 |
| arXiv:2605.99999 | unresolvable | core | 0.0 | 0.0 |

## Local discovery X = 0.0

| Module | Source | Status | Full S | Ablated S | Effect ln | a | n | c | r | x |
|---|---|---|---|---|---|---|---|---|---|---|
| stable_rank_gate | arXiv:math/0503442 (2007, math) | decorative import (no significant difference in ablation) | 0.7308 | 0.8845 | -0.1909 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| power_iteration_sigma | vonmises1929 (1929, math) | retrieval reuse (the same use already exists in ML) | 0.7308 | 0.5942 | 0.2069 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |

- stable_rank_gate: In the hidden snapshot, stable rank is used for generalization-bound analysis and low-rank compression, but not to adjust the whitening strength of an optimizer
- power_iteration_sigma: Spectral Normalization (1802.05957) and others already estimate the spectral norm with persistent power iteration

## Appendix: paper weights of the seed neighborhood

| Paper | V | C | Age | w |
|---|---|---|---|---|
| adam | 1.0 | 0.974 | 11.52 | 0.9842 |
| adamw | 1.0 | 0.921 | 8.63 | 0.9526 |
| lamb | 1.0 | 0.868 | 7.25 | 0.9211 |
| rudelson_vershynin_2007 | 1.0 | 0.868 | 21.28 | 0.9208 |
| adafactor | 1.0 | 0.763 | 8.22 | 0.8579 |
| fantastic | 1.0 | 0.395 | 0.83 | 0.8499 |
| lion | 1.0 | 0.711 | 3.38 | 0.8263 |
| soap | 1.0 | 0.5 | 1.79 | 0.7322 |
| schedulefree | 1.0 | 0.553 | 2.1 | 0.7316 |
| vonmises1929 | 0.8 | 0.649 | 97.49 | 0.7092 |
| sophia | 1.0 | 0.447 | 3.11 | 0.6684 |
| kimi_k2 | 0.6 | 0.658 | 0.93 | 0.6161 |
| crowded_valley | 1.0 | 0.342 | 5.99 | 0.6053 |
| muon_scalable | 0.6 | 0.605 | 1.35 | 0.6021 |
| mars | 1.0 | 0.132 | 1.62 | 0.577 |
| ademamix | 1.0 | 0.184 | 1.82 | 0.5551 |
| shampoo | 1.0 | 0.237 | 8.34 | 0.5421 |
| muon_blog | 0.25 | 0.816 | 1.56 | 0.5149 |
| deconstructing | 1.0 | 0.026 | 1.97 | 0.4234 |
| bench_optim | 0.4 | 0.289 | 0.83 | 0.3725 |
| cautious | 0.4 | 0.079 | 1.6 | 0.2463 |

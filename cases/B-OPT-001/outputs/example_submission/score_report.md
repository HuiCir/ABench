# ABench score report: B-OPT-001 / rankgate

> **Training results and judge outputs are simulated data**; this report only validates the pipeline and does not represent the method's real performance.

## Overall score: 41.67

| Threshold | Result |
|---|---|
| L0 | pass |
| SOTA | pass |
| Differentiated | pass |

Score = 100 · M^0.5 · D^0.3 · Q^0.2 · (1 + 0.25·X) = 100 · 0.2366^0.5 · 0.3988^0.3 · 0.6393^0.2 · (1 + 0.25·0.9371) = 33.76 × 1.234 = **41.67**

## L0 automatic checks

| Check | Result |
|---|---|
| interface | True |
| state_ratio_max | 2.0 |
| state_memory<=3x | True |
| finite | True |
| quadratic_loss_ratio | 0.0222 |
| descends | True |
| deterministic | True |
| pass | True |

## SOTA margin M = 0.2366

Aggregate effective speedup (relative to the reference envelope): **1.0344×**, lower confidence bound 1.0311×

| Setting | Speedup | Per seed | Submission loss | Best reference at 1× budget | Its loss | Envelope-dominant method |
|---|---|---|---|---|---|---|
| L2_main | 1.0405 | [1.0445, 1.034, 1.043] | 4.1319 | muon | 4.1522 | muon |
| L3a_scale | 1.0657 | [1.0739, 1.0575] | 3.6601 | muon | 3.6836 | muon |
| L3b_long | 1.0088 | [1.018, 0.9997] | 4.1614 | soap | 4.1686 | muon |
| L3c_bigbatch | 1.0056 | [1.0111, 1.0002] | 4.2024 | muon | 4.2133 | muon |

| SOTA threshold | Result |
|---|---|
| aggregate_speedup>=1.00 | pass |
| lcb>=0.98 | pass |
| per_setting>=0.90 | pass |
| ood_regression<=1% | pass |
| step_time<=1.25x | pass |

OOD: wikitext-103 degraded 0.19%; the-stack-smol/python degraded 0.34%

## Method divergence D = 0.3988

- Behavioral divergence D_beh = 0.1174 (global 0.1681, patch 0.0666), nearest reference: cautious_adamw
- Hyperparameter-perturbation noise floor 0.0745 ({'adamw': 0.0345, 'cautious_adamw': 0.0203, 'lion': 0.0376, 'adafactor': 0.038, 'ademamix': 0.0318, 'mars': 0.0312, 'muon': 0.0745}), clone threshold 0.1117
- Code divergence D_code = 0.7117, nearest reference: adafactor
- Mechanism divergence D_mech = 0.6667 (judge levels [2, 1, 2]), closest prior work: Muon (2502.16982); the hidden snapshot also contains work that switches optimizers per layer, but all of it is static assignment, with no dynamic gating by the spectral structure of the momentum

| Tensor | Nearest reference | Similarity |
|---|---|---|
| layers.4.self_attn.q_proj | cautious_adamw | 0.8339 |
| layers.4.mlp.up_proj | muon | 0.9948 |
| layers.20.self_attn.o_proj | adafactor | 0.9506 |
| layers.20.mlp.down_proj | adafactor | 0.9542 |

Mean behavioral similarity to each reference method: cautious_adamw 0.8319, adamw 0.8237, ademamix 0.8228, adafactor 0.8188, mars 0.7983, lion 0.7458, muon 0.7334

| Divergence threshold | Result |
|---|---|
| D>=0.20 | pass |
| D_global>=0.112 (noise floor 0.074 × 1.5) | pass |

## Reference quality Q = 0.6393 (before penalty 0.999, penalty factor 0.64)

| Source | Status | Usage | Paper weight w | Contribution w·u |
|---|---|---|---|---|
| url:https://kellerjordan.github.io/posts/muon/ | unresolvable | decorative | 0.0 | 0.0 |
| arXiv:1412.6980 | ok | partial | 0.9842 | 0.4921 |
| arXiv:2502.16982 | ok | core | 0.6021 | 0.6021 |
| arXiv:math/0503442 | ok | core | 0.9208 | 0.9208 |
| arXiv:1804.04235 | ok | core | 0.8579 | 0.8579 |
| doi:10.1002/zamm.19290090105 | unresolvable | decorative | 0.0 | 0.0 |
| arXiv:2509.02046 | ok | partial | 0.8499 | 0.4249 |
| arXiv:2407.07972 | ok | partial | 0.4234 | 0.2117 |

## Local discovery X = 0.9371

| Module | Source | Status | Full S | Ablated S | Effect ln | a | n | c | r | x |
|---|---|---|---|---|---|---|---|---|---|---|
| stable_rank_gate | arXiv:math/0503442 (2007, math) | local discovery | 1.0405 | 0.994 | 0.0457 | 0.9371 | 1.0 | 1.0 | 1.0 | 0.9371 |
| power_iteration_sigma | vonmises1929 (1929, math) | retrieval reuse (the same use already exists in ML) | 1.0405 | 0.8747 | 0.1736 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |

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

# ABench score report: B-OPT-001 / orthomomentum_pro

> **Training results and judge outputs are simulated data**; this report only validates the pipeline and does not represent the method's real performance.

## Overall score: 0.0

| Threshold | Result |
|---|---|
| L0 | pass |
| SOTA | pass |
| Differentiated | fail |

Score = 100 · M^0.5 · D^0.3 · Q^0.2 · (1 + 0.25·X) = 100 · 0.1421^0.5 · 0.1573^0.3 · 0.9937^0.2 · (1 + 0.25·0.0) = 21.62 × 1.000 = **21.62**

## L0 automatic checks

| Check | Result |
|---|---|
| interface | True |
| state_ratio_max | 2.0 |
| state_memory<=3x | True |
| finite | True |
| quadratic_loss_ratio | 0.0394 |
| descends | True |
| deterministic | True |
| pass | True |

## SOTA margin M = 0.1421

Aggregate effective speedup (relative to the reference envelope): **1.0105×**, lower confidence bound 1.0065×

| Setting | Speedup | Per seed | Submission loss | Best reference at 1× budget | Its loss | Envelope-dominant method |
|---|---|---|---|---|---|---|
| L2_main | 1.0044 | [1.0025, 0.9885, 1.0225] | 4.1511 | muon | 4.1522 | muon |
| L3a_scale | 1.0213 | [1.0241, 1.0186] | 3.6782 | muon | 3.6836 | muon |
| L3b_long | 1.0 | [1.0131, 0.9871] | 4.1693 | soap | 4.1686 | muon |
| L3c_bigbatch | 1.0141 | [1.0147, 1.0136] | 4.2088 | muon | 4.2133 | muon |

| SOTA threshold | Result |
|---|---|
| aggregate_speedup>=1.00 | pass |
| lcb>=0.98 | pass |
| per_setting>=0.90 | pass |
| ood_regression<=1% | pass |
| step_time<=1.25x | pass |

OOD: wikitext-103 degraded 0.04%; the-stack-smol/python degraded 0.08%

## Method divergence D = 0.1573

- Behavioral divergence D_beh = 0.0745 (global 0.0745, patch 0.0745), nearest reference: muon
- Hyperparameter-perturbation noise floor 0.0745 ({'adamw': 0.0345, 'cautious_adamw': 0.0203, 'lion': 0.0376, 'adafactor': 0.038, 'ademamix': 0.0318, 'mars': 0.0312, 'muon': 0.0745}), clone threshold 0.1117
- Code divergence D_code = 0.8, nearest reference: muon
- Mechanism divergence D_mech = 0.0 (judge levels [0, 0, 0]), closest prior work: Muon (2502.16982): equivalent method, only renamed and retuned

| Tensor | Nearest reference | Similarity |
|---|---|---|
| layers.4.self_attn.q_proj | muon | 0.9178 |
| layers.4.mlp.up_proj | muon | 0.9449 |
| layers.20.self_attn.o_proj | muon | 0.9096 |
| layers.20.mlp.down_proj | muon | 0.9299 |

Mean behavioral similarity to each reference method: muon 0.9255, cautious_adamw 0.5358, adafactor 0.488, adamw 0.4876, ademamix 0.4849, lion 0.4736, mars 0.4376

| Divergence threshold | Result |
|---|---|
| D>=0.20 | fail |
| D_global>=0.112 (noise floor 0.074 × 1.5) | fail |

## Reference quality Q = 0.9937 (before penalty 0.9937, penalty factor 1.0)

| Source | Status | Usage | Paper weight w | Contribution w·u |
|---|---|---|---|---|
| arXiv:2502.16982 | ok | core | 0.6021 | 0.6021 |
| arXiv:1412.6980 | ok | core | 0.9842 | 0.9842 |
| arXiv:2302.06675 | ok | decorative | 0.8263 | 0.0 |

## Local discovery X = 0.0

No local discovery claimed.

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

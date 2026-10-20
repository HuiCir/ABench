# Design notes: RidgeSign

> This is the example submission after enumerating 6 candidates on the frozen `lion-primitive-v1` grammar; the training curves are still simulated.

## Search observations

Lion produces a sign update from a single EMA; its state is cheap, but every coordinate gets exactly the same step. The fixed-gradient-stream proxy shows that extrapolating between fast and slow EMAs and using `|z|/(|z|+τ)` as a magnitude-confidence gate reduces jitter on weak-signal coordinates while keeping the directional robustness of sign. `search.py` uses only frozen ASTs, and `search_log.jsonl` records every candidate and its proxy score.

## Update rule

\[
 m^f_t=\beta_f m^f_{t-1}+(1-\beta_f)g_t,\quad
 m^s_t=\beta_s m^s_{t-1}+(1-\beta_s)g_t,
\]
\[
 z_t=m^f_t+\alpha(m^f_t-m^s_t),\quad
 c_t=|z_t|/(|z_t|+\tau),\quad q_t=\tfrac12(1+\operatorname{sign}(m^f_t-m^s_t)),\quad
 \theta_{t+1}=\theta_t-\eta_t(\operatorname{sign}(z_t)c_tq_t+\lambda\theta_t).
\]

This stores only two EMAs of the parameter's shape, so the state is 2×; it carries no learning-rate schedule.

## Differences from existing methods

- Lion has a single EMA and a constant-magnitude sign; RidgeSign maintains an extrapolation of fast and slow EMAs and keeps a continuous confidence magnitude.
- AdamW needs element-wise second moments; RidgeSign stores no second moment, has smaller state, and does not use validation loss.
- AdEMAMix combines fast and slow EMAs and then normalizes adaptively; RidgeSign's second step is the sign × confidence × delta-sign filter found by frozen-primitive search, not second-order preconditioning.

## Expectations and failure conditions

Expected to help in micro-scale training stages where the gradient direction flips often but the main signal stays stable; when all gradient magnitudes are large or the noise is extremely low, confidence is close to 1 and the behavior approaches Lion. Large-scale advantages and cross-dataset generalization must be confirmed by hidden final evaluation.

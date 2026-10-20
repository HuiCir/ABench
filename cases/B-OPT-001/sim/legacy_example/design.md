# Design notes: RankGate (example submission)

> Example method used to demonstrate the ABench flow. Its gain figures are all simulated and have not been validated by real training.

## Motivation

1. Matrix optimizers (Muon, SOAP) have a clear token-efficiency advantage over tuned AdamW at the 0.1B scale, but this advantage shrinks as scale grows (2509.02046).
2. Muon flattens all singular values of the momentum matrix to 1. When the momentum is high-rank this is beneficial whitening; when it is low-rank (e.g. late in training, some attention projections), what gets amplified are noise-dominated small singular directions.
3. Element-wise adaptive methods (Adam/Adafactor) are more robust in such cases; the ablations of 2407.07972 also show that adaptivity is especially critical for a few layers.

## Update rule

For a 2D parameter $W \in \mathbb{R}^{m\times n}$:

$$
\begin{aligned}
M_t &= \mu M_{t-1} + G_t,\quad \tilde M_t = G_t + \mu M_t \\
O_t &= 0.2\sqrt{\max(m,n)}\cdot \mathrm{NS}_5(\tilde M_t) \\
A_t &= \mathrm{RMS\text{-}match}_{0.2}\big(\tilde M_t / \sqrt{\hat V^{\text{row}}_t \hat V^{\text{col}}_t / \overline{\hat V^{\text{row}}_t}}\big) \\
r_t &= \frac{\|\tilde M_t\|_F^2}{\|\tilde M_t\|_2^2 \cdot \min(m,n)},\quad a_t = \sigma\big(\kappa (\ln r_t-\ln\tau)\big) \\
W_{t+1} &= W_t - \eta_t\big(a_t O_t + (1-a_t) A_t + \lambda W_t\big)
\end{aligned}
$$

The spectral norm is estimated by a single power-iteration step on a persistent vector, at an extra cost of two matrix–vector products.

The gate operates on a log scale: L1 self-tests found that the normalized stable rank of the accumulated Nesterov momentum concentrates in 0.01–0.05, so a linear gate (τ=0.15) is almost always 0 and the method degenerates to Adafactor; with a log gate the default τ is 0.02. 1D parameters and embeddings use AdamW.

## Differences from the closest existing methods

| Method | Difference |
|---|---|
| Muon | Muon always orthogonalizes; RankGate interpolates continuously between orthogonalization and factored adaptivity by stable rank |
| Adafactor | RankGate's A branch reuses Adafactor's factored second moment, but applies it to the Nesterov momentum and dominates only at low rank |
| SOAP | SOAP runs Adam in the Shampoo eigenbasis and must maintain two preconditioner matrices; RankGate keeps no eigenbasis, and its state is only 1× the parameter count plus row/column vectors |

## Expectations

- Expected to help: larger models and late training where the momentum is low-rank (corresponding to the long-training setting).
- Expected not to help or uncertain: at large batch sizes the momentum itself is smoother and higher-rank, so the gate approaches pure Muon and the gain may disappear.
- Risk: τ and κ may need different values in different layers; v1 uses shared values.

# Synthesis: reusable conclusions from Lion-style program search

1. The core of Lion is a sign update with low state cost: one EMA gives the update direction and another EMA gives the next state; it has more of an advantage at larger batch sizes, but the fixed magnitude loses signal strength.
2. Multi-setting optimizer benchmarks show that once the reference envelope is tuned, leads in a single public setting shrink, so scale, long training and large batch must be hidden points.
3. The main risk of the search process is not that the candidate space is too small, but that overfitting the fixed stream produces programs that cannot be replayed. This case therefore freezes the AST, candidate count, seed and an independent μ1 confirmation.
4. The gap is to combine Lion's discrete sign with a "signal confidence" primitive while keeping 2× state and the shared schedule; RidgeSign is an auditable candidate in that gap and does not claim that real training has proven it SOTA.

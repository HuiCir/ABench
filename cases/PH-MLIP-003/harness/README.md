# PH-MLIP-003 harness contract

Only the interface is defined at present; no simulated implementation is provided. The formal harness must:

1. read data, models, hardware and seeds from the server manifest;
2. allow only the proxies/budgets declared in `eval_protocol.yaml`;
3. output a deterministic `run.jsonl`, metrics and `artifacts.sha256`;
4. reject unauthorized network, file-system, GPU/CPU resources and undeclared sources of randomness;
5. write structured status for failures, OOM, timeouts and numerical anomalies, without swallowing errors.

Domain-specific inputs, compilers, simulators or evaluators are wired in once blocked_inputs are frozen.

# X-DPO-006 harness contract

Only the interface is defined at present; no simulated implementation is provided. The formal harness must read offline resources, models, hardware and seeds from the server manifest, reject undeclared network/file/process access, output a deterministic `run.jsonl`, `metrics.json` and `artifacts.sha256`, and record OOM, timeouts, failures and numerical anomalies in structured form.

The target of this case is `pareto_frontier`; a single performance value must not replace the full cost, robustness and constraint metrics.

# Anti-gaming contract scaffold

- Run results must be produced by the benchmark operator's runner and include configuration, code, resource and artifact hashes.
- A submission must not read hidden labels, the validation set, evaluator state or external network resources.
- Time, memory, throughput, OOM, timeouts and randomness are measured on the server; self-reported fields from the submission are not trusted on their own.
- Results that are missing, irreproducible or of unclear provenance are marked failed/unknown and are never interpolated into a score.
- Before real launch, case-specific negative examples, leak checks and hidden perturbations must be added.

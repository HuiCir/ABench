# Anti-gaming checklist (C-QNT-003)

| Gaming method | Protection | Stage |
|---|---|---|
| Self-reported scores or fabricated curves | Scoring reads only server-side runs/judge outputs and rejects simulated provenance | L2/L3 |
| Overfitting the public test set | Hidden inputs, hidden configurations and OOD workloads | L3 |
| Exceeding resource caps | The sandbox monitors wall clock, memory, CPU, network and file system | L0-L3 |
| Runtime drift | runtime/container/config hashes registered | L0 |
| Submitting only the best result after filtering out failures | A complete attempt log and an immutable artifact index are required | L2 |
| Falling back to undeclared library/hardware features | Audits of imports, kernels, devices and the call graph | L0/L3 |

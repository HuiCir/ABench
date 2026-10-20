# Anti-gaming contract scaffold

- Patches may modify only allowed source files; changes to tests, configuration, dependency lockfiles and evaluation scripts are rejected.
- The runner executes the fixed visible/hidden tests on an isolated copy and monitors network, subprocesses, file access, timeouts and resource usage.
- Tasks are held out by repository, API, defect family and time/version hash; hard-coded task answers, test probing and external code retrieval are not accepted.
- Passed tests, failure categories, time, memory, GPU-h and randomness are measured server-side; self-reported fields from the submission are not trusted.
- Results that are missing, irreproducible or of unclear provenance are marked failed/unknown and are never interpolated into a score.
- Before real launch, the fixed bundle, hidden regressions, contamination audit and harmlessness/safe-process checks must be added.

# X-CODE-008 harness contract

Only the interface is defined at present; no model, repository, test or simulated implementation is provided. The formal harness must:

1. read the hashes of the fixed model, tokenizer, training data, repositories and seeds from the server manifest;
2. run training/inference in a sandbox with no network, write-only output and limited GPU/CPU/wall clock;
3. output structured patches, generation configuration and `run.jsonl` for each task, without outputting hidden inputs or test contents;
4. apply patches on an isolated copy, reject modifications to test/dependency files, and then have the server run pytest and the hidden regression tests;
5. output a deterministic `metrics.json` and `artifacts.sha256`, recording OOM, timeouts, syntax errors and test-failure categories;
6. reject undeclared network, external file, shared cache and arbitrary subprocess access.

All test results, timing, memory and process audits are produced by the benchmark operator's runner; this directory executes no code and creates no scores.

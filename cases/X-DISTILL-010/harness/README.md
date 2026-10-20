# X-DISTILL-010 harness contract

Only the interface is defined at present; no simulated implementation is provided. The formal harness reads the frozen teacher/student, tokenizer, offline corpus, optional teacher-logit cache and seed from the server manifest, rejects network/undeclared file access, and outputs a deterministic `run.jsonl`, `metrics.json` and `artifacts.sha256`. Time, memory and failure states of the teacher forward/cache, student training and inference are measured server-side.

# X-DATA-007 reference submission

`pipeline.py`, `src/dedup.py`: exact deduplication, MinHash near-duplicate removal (word 5-shingles, 20 bands × 5 rows) and a hashed-n-gram logistic-regression quality filter trained to separate the target examples from pool documents; the best-scoring documents are kept until the token budget is reached. Code only; the evaluator trains the fixed model on the selection. Design report: `report/index.html`; references: `references.yaml`.

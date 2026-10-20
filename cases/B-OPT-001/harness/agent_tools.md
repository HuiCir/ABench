# Tools available to the agent under test (inside the sandbox)

Every call is written to `submission/retrieval_log.jsonl`, so the scoring stage can check sources and contamination.

| Tool | Arguments | Returns | Constraints |
|---|---|---|---|
| `paper_search(query, k=20)` | keywords | list of paper metadata (title, authors, arXiv/DOI, first-release date, venue) | Backed by local mirrors of arXiv + OpenAlex + Semantic Scholar, **filtering out entries with first_public ≥ 2026-07-01**; citation counts are not returned (to keep the agent from chasing reference quality by citations) |
| `paper_read(id, section=None)` | paper identifier | text of the full paper or a given section | Same cutoff rule; code repositories return only the last commit before the cutoff |
| `run_l0(path)` | submission directory | pass/fail and error messages for each L0 check | Not charged to the GPU budget |
| `run_l1(path, hparams, seed, level="mu1")` | submission directory, hyperparameters, tier (mu1 / mu2) | training curve and validation loss | about 0.04 GPU-h at μ1 and about 0.25 GPU-h at μ2, deducted from the 8 GPU-h budget; μ3 and hidden settings are not available |
| `submit(path)` | submission directory | submission receipt | Can be called only once; the session ends after the call |

Log format (one entry per line):

```json
{"t": "2026-07-01T09:12:03Z", "tool": "paper_search", "args": {"query": "orthogonalized momentum optimizer"}, "result_ids": ["2502.16982", "..."]}
```

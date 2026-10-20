# Offline paper search

The agent-side tool `abench_papers.py` is copied into every agent view as `tools/abench_papers.py`. It searches
titles and abstracts of arXiv papers whose **first version is dated before the case's knowledge cutoff**, with no
network access, and logs every call (hash-chained JSON lines) to `$ABENCH_RETRIEVAL_LOG`. The runner
(`tools/sandbox_runner.py`) sets `ABENCH_PAPERS_INDEX`, `ABENCH_CUTOFF` (from `knowledge_cutoff` in `case/oj.yaml`)
and `ABENCH_RETRIEVAL_LOG`, copies the log into the run record as `retrieval_log.jsonl`, and records its hash,
entry count and chain check in `agent_run.json` together with the index provenance.

## Building the index (operator, once per cutoff)

1. Obtain the arXiv metadata snapshot as JSON lines (one object per paper with `id`, `title`, `authors`,
   `abstract`, `categories` and `versions[].created`), for example `arxiv-metadata-oai-snapshot.json` from the
   arXiv dataset that Cornell University publishes on Kaggle (linked from https://info.arxiv.org/help/bulk_data.html).
   arXiv descriptive metadata may be retrieved, stored, transformed and shared under CC0 1.0
   (https://info.arxiv.org/help/api/tou.html); no full texts are used.
2. Build:

   ```
   python3 tools/paper_search/build_paper_index.py --snapshot arxiv-metadata-oai-snapshot.json \
       --cutoff 2026-07-01 --out ~/.abench/papers/papers-2026-07-01.sqlite
   ln -s papers-2026-07-01.sqlite ~/.abench/papers/current.sqlite
   ```

   The default category filter keeps cs.*, stat.*, eess.*, q-bio.*, q-fin.*, econ.*, nlin.* and a few math and
   physics categories used by the tracks; pass `--categories` with no values to keep everything. The builder refuses
   to overwrite an existing index and stops if the input does not have the expected schema.
3. Check: `ABENCH_PAPERS_INDEX=~/.abench/papers/current.sqlite python3 tools/paper_search/abench_papers.py info`.

The index must be built with a cutoff no later than the case cutoff; the tool also applies `ABENCH_CUTOFF` on every
query, so a later index cannot leak newer papers, only waste space.

## Limits

- The log is written by a process the agent controls; under `host_soft` isolation the agent could edit it. The hash
  chain makes edits detectable but not impossible; benchmark runs rely on the wrapper's isolation.
- Only arXiv is covered. Papers published elsewhere (journals, conference proceedings without an arXiv version) are
  not searchable; the agent may still cite them from its own knowledge if they predate the cutoff.

#!/usr/bin/env python3
"""ABench offline paper search (agent side): titles and abstracts of arXiv papers first submitted before the case cutoff.

  python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "linear attention delta rule" [-n 10] [--category cs.LG]
  python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" show 2312.06635
  python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" info

The index is a SQLite FTS5 database built by the operator with build_paper_index.py from the arXiv metadata snapshot
(descriptive metadata, CC0 1.0 under the arXiv API terms of use). It holds only papers whose first version is dated
before the cutoff, and every result is checked against the cutoff again here. There is no network access; full texts
are not available.

Every call is appended to the retrieval log ($ABENCH_RETRIEVAL_LOG, default $ABENCH_WORKSPACE/.abench/retrieval_log.jsonl)
with the query and the returned ids. Each line carries the SHA-256 of the previous line, so edits to the log are
detectable; the runner collects it with the run record. Standard library only (Python 3.9+).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent


def index_path() -> Path:
    p = os.environ.get("ABENCH_PAPERS_INDEX")
    return Path(p) if p else HERE / "papers.sqlite"


def log_path() -> Path | None:
    p = os.environ.get("ABENCH_RETRIEVAL_LOG")
    if p:
        return Path(p)
    ws = os.environ.get("ABENCH_WORKSPACE")
    return Path(ws) / ".abench" / "retrieval_log.jsonl" if ws else None


def append_log(entry: dict) -> None:
    p = log_path()
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = "0" * 64
    if p.is_file():
        lines = p.read_bytes().splitlines()
        if lines:
            prev = hashlib.sha256(lines[-1]).hexdigest()
    entry = {"ts": round(time.time(), 3), **entry, "prev_sha256": prev}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def connect() -> tuple[sqlite3.Connection, dict]:
    p = index_path()
    if not p.is_file():
        raise SystemExit(f"paper index not found: {p} (the operator has not installed it for this run)")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    meta = dict(con.execute("SELECT key, value FROM meta"))
    return con, meta


def cutoff(meta: dict) -> str:
    """The stricter of the index cutoff and $ABENCH_CUTOFF (YYYY-MM-DD; papers dated on or after it are excluded)."""
    c = [meta.get("cutoff", "9999-12-31")]
    if os.environ.get("ABENCH_CUTOFF"):
        c.append(os.environ["ABENCH_CUTOFF"])
    return min(c)


def fts_query(text: str) -> str:
    """Plain words → an FTS5 query: every word as a quoted token (OR-ed), so punctuation cannot break the syntax."""
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not words:
        raise SystemExit("empty query")
    return " OR ".join(f'"{w}"' for w in words[:32])


def do_search(a: argparse.Namespace) -> int:
    con, meta = connect()
    cut = cutoff(meta)
    sql = ("SELECT p.id, p.title, p.authors, p.categories, p.date, snippet(papers_fts, 2, '[', ']', ' … ', 24) "
           "FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
           "WHERE papers_fts MATCH ? AND p.date < ?")
    args: list = [fts_query(a.query), cut]
    if a.category:
        sql += " AND (' ' || p.categories || ' ') LIKE ?"
        args.append(f"% {a.category}%")
    if a.since:
        sql += " AND p.date >= ?"
        args.append(f"{a.since:04d}-01-01")
    sql += " ORDER BY bm25(papers_fts, 4.0, 1.0) LIMIT ?"     # title weighted over abstract
    args.append(max(1, min(int(a.n), 50)))
    rows = con.execute(sql, args).fetchall()
    append_log({"cmd": "search", "query": a.query, "category": a.category, "since": a.since, "n": a.n,
                "cutoff": cut, "results": [r[0] for r in rows]})
    if a.json:
        print(json.dumps([dict(zip(("id", "title", "authors", "categories", "date", "snippet"), r)) for r in rows],
                         ensure_ascii=False, indent=1))
        return 0
    if not rows:
        print("no results")
    for i, (pid, title, authors, cats, date, snip) in enumerate(rows, 1):
        au = authors if len(authors) <= 90 else authors[:87] + "..."
        print(f"{i:2d}. arXiv:{pid}  ({date}, {cats.split()[0] if cats else '-'})\n    {title}\n    {au}")
        print(textwrap.indent(textwrap.fill(" ".join(snip.split()), 110), "    ") + "\n")
    return 0


def do_show(a: argparse.Namespace) -> int:
    con, meta = connect()
    cut = cutoff(meta)
    pid = re.sub(r"^(arxiv:)?", "", a.id.strip(), flags=re.I)
    pid = re.sub(r"v\d+$", "", pid)
    row = con.execute("SELECT id, title, authors, categories, date, abstract FROM papers WHERE id = ? AND date < ?",
                      (pid, cut)).fetchone()
    append_log({"cmd": "show", "id": pid, "cutoff": cut, "results": [row[0]] if row else []})
    if not row:
        print(f"arXiv:{pid}: not in the index (unknown, or first submitted on or after the cutoff {cut})")
        return 1
    if a.json:
        print(json.dumps(dict(zip(("id", "title", "authors", "categories", "date", "abstract"), row)),
                         ensure_ascii=False, indent=1))
        return 0
    pid, title, authors, cats, date, abstract = row
    print(f"arXiv:{pid}  (first version {date}; {cats})\n{title}\n{authors}\n")
    print(textwrap.fill(" ".join(abstract.split()), 110))
    return 0


def do_info(_: argparse.Namespace) -> int:
    con, meta = connect()
    meta["cutoff_applied"] = cutoff(meta)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline arXiv title/abstract search, filtered to the case cutoff.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="ranked full-text search over titles and abstracts (BM25)")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=10, help="number of results (max 50)")
    s.add_argument("--category", help="arXiv category prefix, e.g. cs.LG or stat")
    s.add_argument("--since", type=int, help="only papers from this year on")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=do_search)
    s = sub.add_parser("show", help="title, authors, date and abstract of one paper")
    s.add_argument("id", help="arXiv id, e.g. 2312.06635 or cs/0101001")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=do_show)
    s = sub.add_parser("info", help="index provenance: source snapshot, cutoff, number of papers")
    s.set_defaults(fn=do_info)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

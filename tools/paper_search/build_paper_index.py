#!/usr/bin/env python3
"""Build the offline paper index used by abench_papers.py (operator side; runs once per cutoff, not in the sandbox).

  python3 tools/paper_search/build_paper_index.py --snapshot arxiv-metadata-oai-snapshot.json \\
      --cutoff 2026-07-01 --out ~/.abench/papers/papers-2026-07-01.sqlite [--categories cs. stat. eess. math.OC ...]

Input: the arXiv metadata snapshot, one JSON object per line (the format of the "arXiv Dataset" published by Cornell
University on Kaggle and of the OAI-PMH arXiv metadata: id, title, authors, abstract, categories, versions[] with
RFC 2822 "created" dates). arXiv descriptive metadata may be stored, transformed and shared under CC0 1.0 (arXiv API
terms of use, https://info.arxiv.org/help/api/tou.html); full texts are not included.

A paper is kept if its FIRST version is dated strictly before --cutoff and it has a category matching one of the
--categories prefixes (default: all). The SQLite file has a table `papers`, an FTS5 table `papers_fts` over
title/authors/abstract (porter stemming), and a `meta` table recording the snapshot's SHA-256, the cutoff, the
category filter and the row count. Standard library only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

DEFAULT_CATEGORIES = ["cs.", "stat.", "eess.", "math.OC", "math.NA", "math.ST", "q-bio.", "q-fin.", "physics.comp-ph",
                      "physics.flu-dyn", "physics.data-an", "cond-mat.dis-nn", "nlin.", "econ."]


def first_date(rec: dict) -> str | None:
    vs = rec.get("versions") or []
    for v in vs:
        if v.get("version") in ("v1", None) and v.get("created"):
            try:
                return parsedate_to_datetime(v["created"]).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                return None
    if vs and vs[0].get("created"):
        try:
            return parsedate_to_datetime(vs[0]["created"]).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return None
    return None


def opener(p: Path):
    return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(snapshot: Path, out: Path, cutoff: str, categories: list[str] | None) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; indexes are immutable, write a new file")
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out)
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE papers (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, title TEXT, authors TEXT, abstract TEXT,
                             categories TEXT, date TEXT);
        CREATE VIRTUAL TABLE papers_fts USING fts5(title, authors, abstract, content='papers', content_rowid='rowid',
                                                   tokenize='porter unicode61');
    """)
    kept = seen = undated = 0
    batch = []
    with opener(snapshot) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            r = json.loads(line)
            date = first_date(r)
            if not date or "id" not in r:
                undated += 1
                if seen == 1000 and undated > 500:
                    raise SystemExit("most records lack an id or a versions[].created date: not the expected arXiv "
                                     "metadata snapshot schema (see the module docstring)")
                continue
            if date >= cutoff:
                continue
            cats = " ".join((r.get("categories") or "").split())
            if categories and not any(c.startswith(p) for c in cats.split() for p in categories):
                continue
            batch.append((str(r["id"]).strip(), " ".join((r.get("title") or "").split()),
                          " ".join((r.get("authors") or "").split()), (r.get("abstract") or "").strip(), cats, date))
            if len(batch) >= 5000:
                con.executemany("INSERT OR IGNORE INTO papers (id, title, authors, abstract, categories, date) "
                                "VALUES (?, ?, ?, ?, ?, ?)", batch)
                kept += len(batch)
                batch = []
    if batch:
        con.executemany("INSERT OR IGNORE INTO papers (id, title, authors, abstract, categories, date) "
                        "VALUES (?, ?, ?, ?, ?, ?)", batch)
        kept += len(batch)
    con.execute("INSERT INTO papers_fts(papers_fts) VALUES ('rebuild')")
    n = con.execute("SELECT count(*) FROM papers").fetchone()[0]
    meta = {"source": "arXiv metadata snapshot (CC0 1.0 descriptive metadata, https://info.arxiv.org/help/api/tou.html)",
            "snapshot_file": snapshot.name, "snapshot_sha256": sha256_file(snapshot), "cutoff": cutoff,
            "cutoff_rule": "first version dated strictly before the cutoff",
            "categories": " ".join(categories) if categories else "all", "records_read": str(seen),
            "records_without_date": str(undated), "papers": str(n),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "builder": "build_paper_index.py v1"}
    con.executemany("INSERT INTO meta VALUES (?, ?)", list(meta.items()))
    con.commit()
    con.execute("VACUUM")
    con.close()
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD; papers first submitted on or after it are excluded")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES,
                    help="category prefixes to keep; pass with no values to keep everything")
    a = ap.parse_args()
    meta = build(a.snapshot, a.out.expanduser(), a.cutoff, a.categories or None)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ABench v2 reference integrity checker.

The normal submission validator checks only the local YAML/report relationship.  This
companion checker adds two explicit levels:

* static (default): parse YAML, verify IDs and report ``bib.*`` links, and reject
  obviously malformed arXiv identifiers;
* ``--network``: query the official arXiv Atom endpoint and compare the registered
  title with the canonical title.  It falls back to the public arXiv abstract
  page when the Atom endpoint is rate-limited.  Transient failures are warnings
  by default; ``--strict-network`` turns them into errors for a controlled CI job.

It never edits a submission.  Use it before freezing a case; network checks should
run in a controlled CI job and their JSON output should be retained with the case
snapshot.  The runner intentionally does not invoke this tool at submission time.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

ARXIV_ID = re.compile(r"^(?:[a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
ATOM = "{http://www.w3.org/2005/Atom}"


class BibParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.bib: set[str] = set()
        self.cites: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        ident = a.get("id") or ""
        if tag == "li" and ident.startswith("bib."):
            self.bib.add(ident[4:])
        if tag == "a":
            href = a.get("href") or ""
            if href.startswith("#bib."):
                self.cites.add(href[5:])


def normal_title(value: str) -> str:
    # arXiv may use TeX-ish braces or punctuation; comparing alphanumeric words
    # avoids false positives from curly quotes, punctuation and whitespace.
    return "".join(ch.casefold() for ch in value if ch.isalnum())


def canonical_arxiv(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v.lower().startswith("arxiv:"):
        v = v.split(":", 1)[1].strip()
    return v


def check_file(path: Path, network: bool, strict_network: bool, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "errors": [], "warnings": [], "network": []}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        result["errors"].append(f"invalid YAML: {exc}")
        return result
    if not isinstance(doc, dict) or doc.get("schema") != "abench.references/v2":
        result["errors"].append("schema must be abench.references/v2")
    refs = doc.get("references") if isinstance(doc, dict) else None
    if not isinstance(refs, list) or not refs:
        result["errors"].append("references must be a non-empty list")
        return result
    sub = path.parent
    report = sub / "report" / "index.html"
    parser = BibParser()
    if report.is_file():
        parser.feed(report.read_text(encoding="utf-8", errors="replace"))
    ids: set[str] = set()
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            result["errors"].append(f"references[{i}] is not a mapping")
            continue
        rid = ref.get("id")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", rid):
            result["errors"].append(f"references[{i}].id is missing or invalid")
            continue
        if rid in ids:
            result["errors"].append(f"duplicate id: {rid}")
        ids.add(rid)
        if not ref.get("title"):
            result["errors"].append(f"references[{rid}] has no title")
        raw_arxiv = ref.get("arxiv")
        if raw_arxiv is not None and not isinstance(raw_arxiv, str):
            result["errors"].append(f"references[{rid}] arxiv must be a quoted string")
        arxiv = canonical_arxiv(raw_arxiv)
        if arxiv is not None and not ARXIV_ID.fullmatch(arxiv):
            result["errors"].append(f"references[{rid}] malformed arXiv id: {arxiv}")
        if not any(ref.get(k) for k in ("arxiv", "doi", "url")):
            result["errors"].append(f"references[{rid}] has no arxiv/doi/url")
        if network and arxiv and ARXIV_ID.fullmatch(arxiv):
            item = check_arxiv(ref, arxiv, timeout)
            result["network"].append(item)
    for rid in sorted(ids - parser.bib):
        result["errors"].append(f"{rid} is not present as report bib.{rid}")
    for rid in sorted(parser.bib - ids):
        result["errors"].append(f"report bib.{rid} is not registered")
    for rid in sorted(parser.cites - parser.bib):
        result["errors"].append(f"report citation bib.{rid} has no bibliography item")
    for item in result["network"]:
        if item.get("error"):
            message = f"{item['id']}: {item['error']}"
            (result["errors"] if strict_network else result["warnings"]).append(message)
        elif item.get("title_match") is False:
            result["errors"].append(
                f"{item['id']}: title mismatch; registered={item['registered_title']!r}, "
                f"arXiv={item['canonical_title']!r}"
            )
    return result


class _MetaTitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        a = dict(attrs)
        if (a.get("name") or "").lower() == "citation_title" and a.get("content"):
            self.title = html.unescape(a["content"] or "")


def _abs_title(arxiv: str, timeout: float) -> str:
    url = f"https://arxiv.org/abs/{urllib.parse.quote(arxiv, safe='/') }"
    req = urllib.request.Request(url, headers={"User-Agent": "ABench-reference-checker/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        parser = _MetaTitleParser()
        parser.feed(response.read().decode("utf-8", errors="replace"))
    if not parser.title:
        raise ValueError("arXiv abs page has no citation_title")
    return " ".join(parser.title.split())


def check_arxiv(ref: dict[str, Any], arxiv: str, timeout: float) -> dict[str, Any]:
    out: dict[str, Any] = {"id": ref.get("id"), "arxiv": arxiv}
    query = urllib.parse.urlencode({"id_list": arxiv})
    url = f"https://export.arxiv.org/api/query?{query}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ABench-reference-checker/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            root = ET.fromstring(response.read())
        entry = root.find(f"{ATOM}entry")
        if entry is None:
            out["error"] = "arXiv returned no entry"
            return out
        title_node = entry.find(f"{ATOM}title")
        canonical = " ".join((title_node.text or "").split()) if title_node is not None else ""
        registered = str(ref.get("title") or "")
        out.update({"canonical_title": canonical, "registered_title": registered,
                    "title_match": normal_title(canonical) == normal_title(registered)})
        return out
    except Exception as exc:
        # export.arxiv.org is rate-limited more aggressively than the public abs
        # page.  Fall back to its canonical citation_title meta tag before giving
        # up; this keeps a transient API 429 from bypassing a title mismatch.
        try:
            canonical = _abs_title(arxiv, timeout)
            registered = str(ref.get("title") or "")
            out.update({"source": "arxiv_abs", "canonical_title": canonical,
                        "registered_title": registered,
                        "title_match": normal_title(canonical) == normal_title(registered)})
            return out
        except Exception as fallback_exc:
            out["error"] = f"network lookup failed: {exc}; abs fallback: {fallback_exc}"
            return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check ABench references.yaml integrity")
    ap.add_argument("--path", action="append", required=True,
                    help="references.yaml path (repeat for multiple submissions)")
    ap.add_argument("--network", action="store_true",
                    help="query official export.arxiv.org and verify titles")
    ap.add_argument("--strict-network", action="store_true",
                    help="treat transient network lookup failures as errors (implies --network)")
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()
    files = [Path(x) for x in args.path]
    network = args.network or args.strict_network
    results = [check_file(p, network, args.strict_network, args.timeout) for p in files]
    errors = sum(len(r["errors"]) for r in results)
    print(json.dumps({"valid": errors == 0, "network": network, "strict_network": args.strict_network,
                      "files": results}, ensure_ascii=False, indent=2))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

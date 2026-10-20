#!/usr/bin/env python3
"""Render a design report (arXiv-HTML / LaTeXML markup, design_report_v2) and its references.yaml from a YAML source.

Used for the reference submissions of the standard cases (spec/standard_cases/reports/<ID>.yaml); agents may use it
too, but any report following sandbox/templates/design_report.html is accepted.

Source format:
  case: TC-FT-001
  title: ...
  author_note: ...
  abstract: paragraph text
  sections:                       # S1..S5: Introduction, Related Work, Method, Design Rationale, Scope and Limitations
    - title: Introduction
      paras: [text, ...]
      subsections: [{title, paras, algorithm: [lines]}]
  changes:                        # required: rendered at the end of S1 as <ol id="changes">
    - {text: "what differs from the anchor/baseline and why", code: "model.py: function"}
  references: [{id, title, authors, year, venue?, arxiv?|doi?|url?, used_for, code_location}]
Inline markup in paragraphs: [@id] or [@id1;@id2] citations, `code`, **bold**, *italic*.

  python3 tools/render_design_report.py SRC.yaml OUT_SUBMISSION_DIR
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STYLE_SRC = ROOT / "sandbox" / "templates" / "design_report.html"


def _style() -> str:
    t = STYLE_SRC.read_text(encoding="utf-8")
    m = re.search(r"<style>.*?</style>", t, re.S)
    return m.group(0) if m else "<style></style>"


def _short(ref: dict) -> str:
    names = [a.strip() for a in re.split(r",| and ", ref["authors"]) if a.strip()]
    last = lambda n: n.split()[-1]
    if len(names) == 1:
        s = last(names[0])
    elif len(names) == 2:
        s = f"{last(names[0])} &amp; {last(names[1])}"
    else:
        s = f"{last(names[0])} et al."
    return s


def _inline(text: str, refs: dict) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r'<code class="ltx_text ltx_font_typewriter">\1</code>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r'<span class="ltx_text ltx_font_bold">\1</span>', out)
    out = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r'<em class="ltx_emph ltx_font_italic">\1</em>', out)

    def cite(m):
        ids = [x.strip().lstrip("@") for x in m.group(1).split(";")]
        parts = []
        for i in ids:
            if i not in refs:
                raise SystemExit(f"citation to unknown reference {i!r}")
            r = refs[i]
            parts.append(f'<a href="#bib.{i}" class="ltx_ref">{_short(r)}, {r["year"]}</a>')
        return '<cite class="ltx_cite ltx_citemacro_citep">(' + "; ".join(parts) + ")</cite>"
    return re.sub(r"\[(@[^\]]+)\]", cite, out)


def _paras(paras: list, refs: dict) -> str:
    return "\n".join(f'<div class="ltx_para"><p class="ltx_p">{_inline(p, refs)}</p></div>' for p in paras or [])


def _algorithm(lines: list, n: int, caption: str, refs: dict) -> str:
    rows = []
    for i, ln in enumerate(lines, 1):
        indent = (len(ln) - len(ln.lstrip(" "))) // 2
        body = _inline(ln.strip(), refs)
        rows.append(f'<div class="ltx_listingline"><span class="ltx_tag ltx_tag_listingline">{i}:</span>'
                    + '<span class="ltx_indent"></span>' * indent + body + "</div>")
    return (f'<figure id="alg{n}" class="ltx_float ltx_float_algorithm ltx_framed ltx_framed_top">'
            f'<figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_float"><span class="ltx_text ltx_font_bold">'
            f'Algorithm {n}</span> </span>{_inline(caption, refs)}</figcaption>'
            f'<div class="ltx_listing ltx_framed_topbottom">' + "\n".join(rows) + "</div></figure>")


def _changes(items: list, refs: dict) -> str:
    if not items:
        raise SystemExit("report source lacks `changes` (the required list of changes relative to the anchor)")
    lis = "\n".join(f'<li class="ltx_item">{_inline(c["text"], refs)} — <code class="ltx_verbatim">{html.escape(c["code"])}</code></li>'
                    for c in items)
    return ('<div class="ltx_para"><p class="ltx_p"><span class="ltx_text ltx_font_bold">Changes relative to the anchor.</span> '
            'Each item states what this submission does differently and where it is in the code.</p>\n'
            f'<ol id="changes" class="ltx_enumerate">\n{lis}\n</ol></div>')


def _bib(r: dict) -> str:
    if r.get("arxiv"):
        link = f'arXiv: <a href="https://arxiv.org/abs/{r["arxiv"]}" class="ltx_ref ltx_href">{r["arxiv"]}</a>'
    elif r.get("doi"):
        link = f'doi: <a href="https://doi.org/{r["doi"]}" class="ltx_ref ltx_href">{html.escape(r["doi"])}</a>'
    else:
        link = f'<a href="{html.escape(r["url"])}" class="ltx_ref ltx_href">{html.escape(r["url"].split("://", 1)[-1])}</a>'
    venue = f'In <em class="ltx_emph ltx_font_italic">{html.escape(r["venue"])}</em>, ' if r.get("venue") else ""
    return (f'<li id="bib.{r["id"]}" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">'
            f'{_short(r)} ({r["year"]})</span>\n<span class="ltx_bibblock">{html.escape(r["authors"])}.\n</span>'
            f'<span class="ltx_bibblock">“{html.escape(r["title"])}”.\n</span>'
            f'<span class="ltx_bibblock">{venue}{r["year"]}. {link}\n</span></li>')


def render(src: dict) -> tuple[str, str]:
    refs = {r["id"]: r for r in src["references"]}
    toc = ['<li class="ltx_tocentry ltx_tocentry_abstract"><a href="#abstract1" class="ltx_ref"><span class="ltx_text ltx_ref_title">Abstract</span></a></li>']
    body = []
    alg_n = 0
    for i, sec in enumerate(src["sections"], 1):
        sid = f"S{i}"
        toc.append(f'<li class="ltx_tocentry ltx_tocentry_section"><a href="#{sid}" class="ltx_ref"><span class="ltx_text ltx_ref_title">'
                   f'<span class="ltx_tag ltx_tag_ref">{i} </span>{html.escape(sec["title"])}</span></a></li>')
        parts = [f'<section id="{sid}" class="ltx_section"><h2 class="ltx_title ltx_title_section">'
                 f'<span class="ltx_tag ltx_tag_section">{i}</span>{html.escape(sec["title"])}</h2>', _paras(sec.get("paras"), refs)]
        if i == 1:
            parts.append(_changes(src.get("changes") or [], refs))
        for j, ss in enumerate(sec.get("subsections") or [], 1):
            ssid = f"{sid}.SS{j}"
            toc.append(f'<li class="ltx_tocentry ltx_tocentry_subsection"><a href="#{ssid}" class="ltx_ref"><span class="ltx_text ltx_ref_title">'
                       f'<span class="ltx_tag ltx_tag_ref">{i}.{j} </span>{html.escape(ss["title"])}</span></a></li>')
            parts.append(f'<section id="{ssid}" class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection">'
                         f'<span class="ltx_tag ltx_tag_subsection">{i}.{j}</span>{html.escape(ss["title"])}</h3>')
            parts.append(_paras(ss.get("paras"), refs))
            if ss.get("algorithm"):
                alg_n += 1
                parts.append(_algorithm(ss["algorithm"], alg_n, ss.get("algorithm_caption", ss["title"]), refs))
            parts.append("</section>")
        parts.append("</section>")
        body.append("\n".join(parts))
    toc.append('<li class="ltx_tocentry ltx_tocentry_section"><a href="#bib" class="ltx_ref"><span class="ltx_text ltx_ref_title">References</span></a></li>')
    case = src["case"]
    title = html.escape(src["title"])
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<meta name="generator" content="ABench design-report template v2 (LaTeXML-compatible markup, modelled on arxiv.org/html)">
<meta name="abench-schema" content="abench.design_report/v2">
<meta name="abench-case-id" content="{case}">
{_style()}
</head>
<body>
<header class="abench-html-header"><span class="brand">ABench</span><span class="crumb">Design report · {case}</span></header>
<nav class="ltx_page_navbar"><nav class="ltx_TOC"><ol class="ltx_toclist">
{chr(10).join(toc)}
</ol></nav></nav>
<div class="ltx_page_main"><div id="infobox" class="infobox"><div id="watermark-tr">ABench:{case} reference submission · literature cutoff 2026-07-01</div></div>
<div class="ltx_page_content"><article class="ltx_document ltx_authors_1line">
<h1 class="ltx_title ltx_title_document">{title}</h1>
<div class="ltx_authors"><span class="ltx_creator ltx_role_author"><span class="ltx_personname">ABench reference submission</span><span class="ltx_contact ltx_role_affiliation">{html.escape(src.get("author_note", "Standard reference for task statement standard v2.1"))}</span></span></div>

<div id="abstract1" class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p">{_inline(src["abstract"], refs)}</p></div>

{chr(10).join(body)}

<section id="bib" class="ltx_bibliography"><h2 class="ltx_title ltx_title_bibliography">References</h2><ul class="ltx_biblist">
{chr(10).join(_bib(r) for r in src["references"])}
</ul></section>
</article></div></div>
<footer class="abench-html-footer">ABench design report · layout follows arXiv HTML (LaTeXML markup), offline and self-contained</footer>
</body>
</html>
"""
    reg = {"schema": "abench.references/v2", "references": []}
    for r in src["references"]:
        e = {k: r[k] for k in ("id", "title", "authors", "year") if k in r}
        for k in ("venue", "arxiv", "doi", "url"):
            if r.get(k):
                e[k] = str(r[k])
        e["used_for"] = r["used_for"]
        e["code_location"] = r.get("code_location")
        reg["references"].append(e)
    return doc, yaml.safe_dump(reg, allow_unicode=True, sort_keys=False, width=110)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    doc, reg = render(src)
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "report" / "index.html").write_text(doc, encoding="utf-8")
    (out / "references.yaml").write_text(reg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Report contract (v2)

The agent submits only one **design report** (a **judgment report** for judgment tasks) and one reference registry. The report explains only the algorithm design:
what was done, why it is designed this way, how it differs from existing methods and under which conditions it holds. **No experiments, results, ablations or numerical evaluation of any kind**;
all of that is done by the judge outside the sandbox. The evaluation rules are stated in the task statement; the report does not repeat them.

## Files

```
report/index.html     # arXiv HTML (LaTeXML markup) layout, offline and self-contained
references.yaml       # one to one with the report's bibliography
```

Templates: `sandbox/templates/design_report.html`, `sandbox/templates/judgment_report.html`,
`sandbox/templates/references.yaml`. Copied into `templates/` of the agent view at run time.
Example: `cases/B-OPT-001/example_submission/report/index.html` (RankGate).

## Layout: following arxiv.org/html

The templates follow the structure and styling of arXiv's official HTML rendering (LaTeXML output):

- Page: dark top bar (#1c1a17), fixed table of contents on the left `nav.ltx_page_navbar > nav.ltx_TOC`,
  body `div.ltx_page_main > article.ltx_document` (max width 832px), info strip `#watermark-tr` at the top right;
- Title `h1.ltx_title_document`, authors `div.ltx_authors`, abstract `div.ltx_abstract` with an `h6` heading;
- Sections `section.ltx_section#S<n>`, `ltx_subsection#S<n>.SS<m>`, `ltx_paragraph`; paragraphs `div.ltx_para > p.ltx_p`;
- Equations `table.ltx_equation.ltx_eqn_table`, number on the right `ltx_eqn_eqno`, MathML with `alttext`;
- Algorithms `figure.ltx_float_algorithm`, one `div.ltx_listingline` per line;
- Theorems and proofs `div.ltx_theorem`, `div.ltx_proof`;
- Citations `cite.ltx_cite > a.ltx_ref[href="#bib.<id>"]`; bibliography `section#bib.ltx_bibliography > li.ltx_bibitem#bib.<id>`.

Body text 16px / line height 24px, justified, citation color #1b7cac, the same as arXiv.

## Design report sections

| Anchor | Section | Content |
|---|---|---|
| `abstract1` | Abstract | Problem, method, core mechanism, one paragraph |
| `S1` | Introduction | Limitations of existing methods, the design points of this work |
| `S2` | Background | Key equations of related methods and their relation to this work |
| `S3` | Method | Overall structure diagram, definitions and equations, algorithm box, complexity |
| `S4` | Design rationale | Why it works (propositions/derivations/intuition), differences from existing methods, sources borrowed from |
| `S5` | Scope and limitations | Optional |
| `bib` | References | |

Judgment report: `abstract1`, `S1` the claim under test, `S2` evidence from the literature, `S3` judgment and reasons, `S4` minimal verification plan, `bib`.

## Automatic checks (`tools/validate_submission.py`)

- the required LaTeXML classes and anchors are present;
- headings must not contain: experiment, result, ablation, evaluation, assessment, metric, benchmark, leaderboard
  (case-insensitive, whole words, singular or plural);
- no `<script>`, `<link>`, `<iframe>`, `<object>`, `<embed>`, `<base>` or non-allowlisted `<meta>`,
  and no references to external resources; only html/svg/png/jpg/webp are allowed under `report/`;
- each `references.yaml` entry has `id`, `title`, `year`, `used_for` and one of `arxiv`/`doi`/`url`; `arxiv` must be a quoted string, so that YAML does not turn something like `1412.6980` into a float and lose trailing zeros;
  ids agree in both directions with the report's `bib.<id>`; every `#bib.` citation in the body has an entry.

## `references.yaml`

```yaml
schema: abench.references/v2
references:
  - id: kingma2015adam
    title: "Adam: A Method for Stochastic Optimization"
    authors: "Diederik P. Kingma, Jimmy Ba"
    year: 2015
    venue: ICLR
    arxiv: "1412.6980"
    used_for: "First- and second-moment updates and bias correction for 1D parameters"
    code_location: "optimizer.py: RankGate._adamw"   # null for pure background citations
```

Reference quality (venue tier × citations) and local discovery are computed by the operator outside the sandbox from this file; the agent does not know the weights.

## Verifying that references are real

`tools/validate_submission.py` only ensures that the local YAML, the report's `bib.*` anchors and the fields agree with each other; it does not treat an arXiv number as proof of a real paper. When freezing a case or preparing a release, first run the offline static check:

```bash
python3 tools/validate_references.py \
  --path cases/B-OPT-001/example_submission/references.yaml \
  --path cases/B-OPT-002/example_submission/references.yaml
```

A networked CI can add verification of titles against the official arXiv Atom API:

```bash
python3 tools/validate_references.py --network --strict-network \
  --path cases/<CASE_ID>/example_submission/references.yaml
```

By default `--network` records transient network failures such as timeouts and rate limits as warnings, and falls back to the arXiv abstract page when the Atom API is rate-limited; only `--strict-network` makes a final failure to query an error. A title mismatch always fails. The verification output should be saved with the case snapshot; it must not run inside the agent sandbox, and the external network must not be a dependency of formal scoring.

# Report templates (v2)

| File | Purpose |
|---|---|
| `design_report.html` | Design report template, arXiv HTML (LaTeXML) layout, offline and self-contained |
| `judgment_report.html` | Judgment report template (for report_only tasks such as T7) |
| `references.yaml` | Reference registry template; ids correspond one to one with `bib.<id>` in the report |

The runner copies the corresponding templates to `templates/report_template.html` and
`templates/references_template.yaml` in the agent view. Reports explain only the design rationale and contain no experiments or numerical results.
Rules are in `spec/report_contract.md`; template self-check: `python3 tools/validate_report_profiles.py`.

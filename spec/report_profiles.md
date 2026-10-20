# Report profiles (v2)

Machine-readable rules are in [report_profiles.yaml](report_profiles.yaml), and the contract is described in [report_contract.md](report_contract.md).

| Profile | Applies to | Required | Required anchors |
|---|---|---|---|
| `design_report_v2` | All method cases (architecture, optimizers, post-training, systems, domain pipelines) | `report/index.html`, `references.yaml` | abstract1, S1–S4, bib |
| `judgment_report_v2` | T7 and other `report_only` judgment tasks (currently also U-KAN-009) | `report/index.html`, `references.yaml` | abstract1, S1–S4, bib |

Each case's profile is taken from `submission.report_profile` in `case/oj.yaml` and must agree with `bench_contract.yaml`;
`report_focus` in `bench_contract.yaml` keeps the v1 direction labels (architecture_design, system_efficiency, etc.)
for classification only.

Assignment precedence is written in `report_profiles.yaml`: explicit exceptions (currently `U-KAN-009`) take precedence over the `-T7-` suffix,
and all other cases use `design_report_v2`. This way renaming a case does not silently change its submission format.

The v1 multi-file report package (method_report.md, experiment_report.md, evidence.yaml, report_manifest.json,
architecture_report.html, etc.) is retired, and its templates moved to `_to_delete/legacy_v1/`.

Check: `python3 tools/validate_report_profiles.py`.

# Report profiles (v2)

Machine-readable rules are in [report_profiles.yaml](report_profiles.yaml), and the contract is described in [report_contract.md](report_contract.md).

| Profile | Applies to | Required | Required anchors |
|---|---|---|---|
| `design_report_v2` | All method cases (architecture, optimizers, post-training, systems, domain pipelines) | `report/index.html`, `references.yaml` | abstract1, S1–S4, bib |
| `judgment_report_v2` | `report_only` judgment tasks (none since the T7 cases were retired on 2026-10-22) | `report/index.html`, `references.yaml` | abstract1, S1–S4, bib |

Each case's profile is taken from `submission.report_profile` in `case/oj.yaml`.

Assignment precedence is written in `report_profiles.yaml`: explicit exceptions (currently none) take precedence over the `-T7-` suffix,
and all other cases use `design_report_v2`. This way renaming a case does not silently change its submission format.

Check: `python3 tools/validate_submission.py --submission <dir> --case-id <ID>` (layout, report structure, references).

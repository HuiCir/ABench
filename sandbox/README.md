# ABench sandbox interface

> **The v2 (OJ-style) notes come first; the text below is the v1 legacy description.** Current flow: the agent sees only `task.md`, interface files, report templates and public
> training data; the hidden set is stored encrypted and decrypted only by the judge; the judge reads only the submitted code and checkpoint to compute metrics.
> See `spec/oj_contract.md` and `spec/sandbox_runner.md`. The v1 report-package schema and MPS smoke profile are retired
> (`profiles/mps_smoke_ds_gbm.yaml` and `tools/smoke_ds_gbm.py` moved to `_to_delete/legacy_v1/`).


This directory defines the runner-facing interface for ABench. It is a
declarative contract: a backend may be implemented with a local process,
container, scheduler, or a remote worker as long as it produces the same
manifest and result records. The interface is deliberately separate from a
case's `case/` files so that cases keep their scientific protocol while the
execution backend can evolve.

The first smoke target is `DS-GBM-001` using the low-cost data-science
contract and a pinned public micro fixture. On Apple Silicon the fixture may
be executed with `--device mps`. This run is a non-scoring workflow check:
smoke collection validates the submission package and returns
`smoke_unscored`; it does not decrypt or run the hidden judge.

## Contracts

- `schemas/sandbox_manifest.schema.json` is the input contract for one run.
- `schemas/result_manifest.schema.json` is the only result contract retained
  after cleanup.
- `schemas/submission_manifest.schema.json` constrains the files that may be
  promoted from the writable workspace.
- `tools/validate_submission.py` validates the v2 submission package
  (`report/index.html` plus `references.yaml`) without requiring `jsonschema`
  (YAML checks use the repository's PyYAML runtime). The report profile and
  required anchors are defined in `spec/report_profiles.yaml`.
- The former `mps_smoke_ds_gbm.yaml` manifest is retained under
  `_to_delete/legacy_v1/` for migration reference. The active local runner is
  configured by `cases/DS-GBM-001/case/oj.yaml` and `spec/sandbox_runner.md`.
- `interfaces.md` specifies data, compute, model-store, workspace, harness,
  submission, and cleanup semantics.
- `harnesses.md` specifies the common lifecycle and adapters for direct Python,
  shell/container, and `claude -p` execution.

The reference local adapter is `tools/sandbox_runner.py`; it supports direct
Python, shell-like command, and Claude Code print-mode harnesses. Its concrete
environment-variable mapping and MPS smoke command are documented in
`spec/sandbox_runner.md`. The generic manifest below is intentionally backend
neutral; the local adapter exposes the active paths through
`ABENCH_DATA_DIR`, `ABENCH_DEVICE`, `ABENCH_WORKSPACE`,
`ABENCH_CHECKPOINT_DIR`, and `ABENCH_SUBMISSION_DIR`.

The schemas do not grant access by themselves. A runner must enforce the path,
network, process, wall-clock, memory, and artifact policies before invoking a
submission. Inputs and base models are immutable; the agent gets a writable
workspace for training and temporary checkpoints, while the collector retains
only the attested submission result and metadata.

Every v2 submission is checked against its case's `report_profile`: the
collector rejects missing `report/index.html`, missing references, external
resources, forbidden result/evaluation headings, and self-reported metric
files. The same checks run for pending judges, so a smoke run can validate the
package without producing a score.

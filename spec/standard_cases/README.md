# Standard cases

A **standard case** (`standard: true`) is a template case whose task statement, interface, sandbox guide, judge and
reference submission are complete and follow task statement standard v2.1; the other cases were filled from these
templates (`spec/cases/`). The 16 standard cases are IC-SPD-001 (hand-written, its own timing judge) and the 15 cases
specified here. On 2026-10-23 four standard cases were removed after the quality review (MD-MNIST-003, FN-LOB-002,
BI-DNA-002, LG-HOLD-002), and TC-FT-001 and DS-PIPE-002 became pipeline-calibration cases
(`scope: calibration` in `index.yaml`): kept to calibrate the judge, the runner and the CPU checkpoint path, not scored.

## Evaluation modes (`mode:` in the spec, copied to `oj.yaml`)

The agent always develops, trains and checks its work in the sandbox with the provided compute (`selfcheck.py` runs
the evaluator's own `stages.py` and `metrics.py` on a dev split). What the evaluator does depends on what is measured:

| Mode | Cases | Submitted | The evaluator |
|---|---|---|---|
| `checkpoint` | TC-FT-001, DS-PIPE-002 (calibration), RS-SEQ-001, GN-OGB-002, PH-NO-001, U-LOOP-003, E-ACT-001, X-LORA-004, C-QNT-003 | model code, `checkpoint/`, and the `train.py` that produced it (kept for audit, never run) | loads the checkpoint, runs inference and measurements only; one run (E-ACT-001: three rollout runs of 150 episodes, for stochastic policies) |
| `judge_trains` | IC-SPD-001, PRO-SPD-004 (time to target), B-OPT-001, T-OPT-002 (optimizers), A-LIN-002 (token mixer under a fixed recipe) | code | trains from scratch, because the training program, or the architecture under a fixed recipe, is what is measured |
| `no_training` | K-FA-003, V-SMP-002 | code | runs and measures; nothing is trained |

Checkpoint cases, and cases whose reference is public prior work (`show_baseline: true`: PRO-SPD-004, a port of the
modded-nanogpt GPT-2 Medium record), show the reference code (not its report) to the agent under `interface/baseline/`
as a working starting point. A-LIN-002 trains the GLA baseline with the same fixed recipe and seeds and reports recall
and throughput relative to it; C-QNT-003 times the original bf16 model in a separate judge-only stage (`ref`).

## Shared evaluation rules

- **Ties.** Cases with `RESAMPLE` in `metrics.py` report a bootstrap 95% interval of the primary metric
  (`primary_ci95`); differences inside it are ties. `selfcheck.py` prints the same interval on the dev split.
- **Compute.** The dev budget is metered by the runner (GPU-busy time, CPU time of the agent's processes); a session
  over budget is not scored. Budgets are stated as one GPU for at most the session length.
- **Environment.** Each case runs in one profile of `spec/env/profiles.yaml` (draft); the judge refuses to score
  without a matching lock (`tools/record_environment.py`).
- **Data.** Sources are verified against the host's published hashes at freeze; secret generation seeds are escrowed
  with the hidden-set key. Long texts are stored packed (`textpack.py`: UTF-8 bytes + offsets) in TC-FT-001 and
  X-LORA-004.

## Working procedure required of the agent

Every standard task statement has a `### Workflow` subsection: research throughout the session, coming back to the
literature whenever experiments raise a question (the sandbox is online for literature, code and documentation through
the runner's logging proxy, `spec/egress_policy.yaml`; the offline paper search `tools/paper_search/` answers fast from
arXiv titles and abstracts; both are logged with the run; there is no literature cutoff), synthesise it as Section 2 (Related Work) of the design
report, design at least one change relative to the anchor and list every change with its code location in
`<ol id="changes">` (checked by `validate_submission.py`), develop and verify with `selfcheck.py`, then submit.

## What a spec contains

`<ID>.yaml` holds the agent-facing text (title, anchor, task, data, interface summary, metrics, rules, full interface
notes), the hardware and budget, and the `oj` block (data sources with licences and the date they were verified,
submission layout, judge seeds and baseline). `reports/<ID>.yaml` is the source of the reference submission's design
report.

`python3 tools/build_standard_cases.py` writes, for each spec: `case/agent/task.md`, `harness/interface.md`,
`harness/sandbox_guide.md`, `harness/submission_rules.json`, `case/oj.yaml`, `README.md`, and copies of the shared kit
(`tools/harness_kit/abench_stage.py`, `abench_metrics.py`, `selfcheck.py`). `--check` fails if anything is stale.
`python3 tools/render_design_report.py reports/<ID>.yaml cases/<ID>/example_submission` renders the report.

Hand-written per case: `harness/stages.py` (pipeline: `STAGES`, `PIPELINE`, `PARAMS`, `HELD_OUT_KEYS`, `make_dev`, optional
`held_out_inputs`), `harness/metrics.py` (`score`, `GATES`, `PRIMARY`, optional `relative`), case helpers, the baseline
under `harness/baselines/` where the judge runs one, `tools/prepare/<id>.py` and `example_submission/`.

## Stage protocol (tools/harness_kit/abench_stage.py)

One stage = one fresh process with empty HOME/TMPDIR/compiler caches and no network proxy:
`setup` (untimed) → byte R → `timed` (imports the submission) → byte T → the parent copies held-out inputs into the
run directory and sends G → `finish` (capped) → byte P → outputs saved. The parent times with its own clock, requires
exactly R, T, P, enforces the finish cap and kills the process group afterwards. Each stage has its own directory and
only files declared in `carry` move between stages. Labels and reference values never enter a child.

## Lifecycle

1. Operator downloads the sources listed by `tools/prepare/run.py --case ID --list-sources`.
2. `tools/prepare/run.py --case ID --raw-dir RAW --key-file KEY` freezes the public files, encrypts the held-out split,
   copies frozen models and writes `case/data_lock.yaml` (hashes), then regenerates `oj.yaml`.
3. A calibration run of the reference submission with `tools/oj_eval.py`; thresholds in `metrics.py` are provisional
   until then.
4. `tools/prepare/run.py --case ID --activate` sets `judge.activation: active`.

No standard case is active yet: all wait for step 1–4 on the operator's hardware. Synthetic data
(`--synthetic N`, with `ABENCH_ALLOW_SYNTHETIC=1`) exists only for plumbing tests and can never be activated.

## Verification status

`tests/test_standard_cases.py` runs ten cases (those whose reference submission needs only NumPy/SciPy/scikit-learn)
end to end on synthetic data, building each checkpoint with the reference `train.py --quick` first, and the judge path
for TC-FT-001 (including `primary_ci95`). The nine cases that need PyTorch, a GPU or a simulator (A-LIN-002, B-OPT-001,
C-QNT-003, E-ACT-001, K-FA-003, PRO-SPD-004, T-OPT-002, V-SMP-002, X-LORA-004) have not been executed: none of these
was available where they were built. Their README states which parts were checked in NumPy transcriptions.

## Cases filled from the templates

`spec/cases/<ID>.yaml` holds 45 further cases built from these templates: 36 filled from the track index,
7 bucket-E cases rebuilt on public data, and FN-LTSF-005 and U-KAN-009, rebuilt as design cases when the T7 judgment
cases were retired (with X-CODE-008 and FN-TS-001). The status of every case is in `verify.csv`.
They use the same builder, stage protocol, prepare modules and checks. Shared evaluator modules for a family of cases
live in `tools/harness_kit/` (`microlm/`, `vision/`, `smol/`, `kernels/`, `lmpost/`, `ltsf/`, `libero/`,
`pro/`) and are copied into each case's harness through the spec's `kit_extra`. Like the standard cases, none is
active yet; each spec's `reference_verification` states what has and has not been run.

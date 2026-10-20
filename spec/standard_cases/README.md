# Standard cases

Every track of `index.yaml` has at least one **standard case** (`standard: true`): a case whose task statement,
interface, sandbox guide, judge and reference submission are complete and follow task statement standard v2.1. The
20 standard cases are IC-SPD-001 (hand-written, its own timing judge) and the 19 cases specified here.

## Evaluation modes (`mode:` in the spec, copied to `oj.yaml`)

The agent always develops, trains and checks its work in the sandbox with the provided compute (`selfcheck.py` runs
the evaluator's own `stages.py` and `metrics.py` on a dev split). What the evaluator does depends on what is measured:

| Mode | Cases | Submitted | The evaluator |
|---|---|---|---|
| `checkpoint` | TC-FT-001, DS-PIPE-002, RS-SEQ-001, GN-OGB-002, FN-LOB-002, MD-MNIST-003, BI-DNA-002, PH-NO-001, LG-HOLD-002, U-LOOP-003, E-ACT-001, X-LORA-004, A-LIN-002, C-QNT-003 | model code, `checkpoint/`, and the `train.py` that produced it (kept for audit, never run) | loads the checkpoint, runs inference and measurements only; one run (E-ACT-001: three rollout runs, for stochastic policies) |
| `judge_trains` | IC-SPD-001, PRO-SPD-004 (time to target), B-OPT-001, T-OPT-002 (optimizers) | code | trains from scratch, because the training program itself is what is measured |
| `no_training` | K-FA-003, V-SMP-002 | code | runs and measures; nothing is trained |

Checkpoint cases show the reference code (model code and `train.py`, not its report) to the agent under
`interface/baseline/` as a working starting point. A-LIN-002 times a freshly initialised GLA model in the same process
for its throughput ratio; C-QNT-003 times the original bf16 model in a separate judge-only stage (`ref`).

## Working procedure required of the agent

Every standard task statement has a `### Workflow` subsection: retrieve related work with the offline paper search
(`tools/paper_search/`, logged and collected with the run), synthesise it as Section 2 (Related Work) of the design
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
for TC-FT-001. The nine cases that need PyTorch, a GPU or a simulator (A-LIN-002, B-OPT-001,
C-QNT-003, E-ACT-001, K-FA-003, PRO-SPD-004, T-OPT-002, V-SMP-002, X-LORA-004) have not been executed: none of these
was available where they were built. Their README states which parts were checked in NumPy transcriptions.

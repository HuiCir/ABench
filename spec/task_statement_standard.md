# Task statement standard v2.1 (task.md)

The task statement is the agent's only source for the task. It must let an agent with no background know what to do, which data to use, what to submit,
**how it will be judged**, and how to work in the sandbox. Reference implementation: `cases/IC-SPD-001/case/agent/task.md`
(hand-written); the other standard cases are generated from `spec/standard_cases/<ID>.yaml` by `tools/build_standard_cases.py`,
which fixes the section order and the shared text of Evaluation, Sandbox Guide, Deliverables and Rules.

## 1. What is public and what is not

The OJ principle is "the rules are public, the test data is not".

| Public (written in the statement) | Not public (judge side only) |
|---|---|
| Task goal and constraints | The test data itself and its labels |
| Data format and the scope of usable data | Evaluation seeds and the concrete values of hidden settings |
| Definition, direction and aggregation of metrics (how many seeds, mean or median) | Code, configuration and results of reference methods |
| Thresholds and validity rules (when valid, when invalid) | Weights and coefficients of the final score components |
| Timing and resource-metering rules, evaluation hardware model | Concrete implementation of anti-gaming detection |
| Evaluation procedure (who runs where, what is read, how it is computed) | Judge-side material (held-out data, reference values, seeds) |
| Sandbox hardware, software, directories, environment variables, time limits and budgets | The hidden-set key and where it is stored |

Words such as "metric", "threshold" and "evaluation" are allowed in the statement, but must be listed explicitly in `agent_visible.leak_allow` of `oj.yaml`.
Terms such as "reference pool", "overall score formula" and "precomputed" are never allowed (see `NEVER_ALLOW` in `tools/leak_check.py`).

## 2. Required sections

| Section | Must contain |
|---|---|
| `## Task` | The goal (measurable), what is submitted (code, or code plus a checkpoint trained in the sandbox and the `train.py` that produced it) and whether the evaluator trains, usable and unusable resources, the open design space; a `### Workflow` subsection: research throughout the session (online literature, code and documentation through the sandbox proxy, plus the offline paper search; no literature cutoff), synthesise (report Section 2, Related Work), design (changes relative to the anchor in `<ol id="changes">` with code locations), develop and verify with the self-check, submit |
| `## Data` | Path, keys, shape, dtype and meaning of each file; a statement that test data is not in the sandbox |
| `## Interface` | Entry file and function signatures (or a pointer to the notes in `interface/`), and how the judge calls them |
| `## Metrics` | A metric table: name, definition, direction, aggregation, threshold; timing or resource-metering rules; validity rules; composition of the final score (without weights) |
| `## Evaluation` | Where evaluation runs, which files it reads, how many seeds, how test data is provided, whether the agent can see the results |
| `## Sandbox Guide` | Hardware, software, network, session limit (`ABENCH_DEADLINE`), development budget, directories and environment variables, self-check commands; may link to a full guide in `interface/` |
| `## Deliverables` | Directory tree, required and optional files, size and type limits, report and reference requirements (related work, the changes list) |
| `## Rules` | What is forbidden and the consequences of violations |

## 3. Companion files

- `oj.yaml`: `task_statement: v2.1`, `agent_visible.interface` (including at least the interface notes), `leak_allow`,
  `agent_budget`, a complete `judge` configuration. Cases whose judge is not yet ready set `judge.activation: awaiting_data`.
- `harness/`: interface notes, sandbox guide, and a self-check script shared with the judge (timed tasks must share the same timing code).
  Standard cases built on the shared kit expose `stages.py` (the pipeline), `metrics.py` (the metric code), `abench_stage.py`,
  `abench_metrics.py`, `selfcheck.py` and `submission_rules.json`, so the agent can read exactly how it will be judged.
- Data preparation from official, verified sources: `tools/prepare/<case>.py` run through `tools/prepare/run.py` (standard cases;
  hashes go to `case/data_lock.yaml` and are merged into `oj.yaml`), or `tools/prepare_<case>.py` (IC-SPD-001, DS-GBM-001).

## 4. Checks

```bash
python3 tools/validate_task_statement.py            # conformance of all cases
python3 tools/validate_task_statement.py --case IC-SPD-001 --strict
```

Every statement is generated from its source `spec/standard_cases/<ID>.yaml` or `spec/cases/<ID>.yaml` by `tools/build_standard_cases.py` (IC-SPD-001's is handcrafted); `tools/validate_task_statement.py --strict` must report it as conforming before a case is activated.

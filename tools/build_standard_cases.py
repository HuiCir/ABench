#!/usr/bin/env python3
"""Build the agent-facing files of the ABench standard cases from spec/standard_cases/<ID>.yaml.

For each spec this writes, inside cases/<ID>/:
  case/agent/task.md          task statement in the v2.1 section order
  harness/interface.md        interface reference (from the spec's interface_md)
  harness/sandbox_guide.md    sandbox operating guide (hardware, directories, workflow, self-check)
  harness/submission_rules.json  layout rules used by selfcheck.py (mirrors oj.yaml submission)
  harness/abench_stage.py, harness/selfcheck.py, harness/abench_metrics.py  copies of tools/harness_kit/
and merges the spec's oj block into case/oj.yaml (standard: true, task_statement: v2.1, judge kind harness).

The case-specific harness/stages.py and harness/metrics.py are written by hand and are not touched here.
  python3 tools/build_standard_cases.py            # all specs
  python3 tools/build_standard_cases.py --case X   # one case
  python3 tools/build_standard_cases.py --check    # fail if any generated file is out of date
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "spec" / "standard_cases"
KIT = ROOT / "tools" / "harness_kit"
KIT_FILES = ("abench_stage.py", "selfcheck.py", "abench_metrics.py")
CUTOFF = "2026-07-01"

HARDWARE = {
    "gpu": {"label": "One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model",
            "device": "cuda", "software": "Python 3, PyTorch (CUDA, Triton), NumPy"},
    "cpu": {"label": "8 CPU cores, no GPU; the evaluation uses the same CPU allotment",
            "device": "cpu", "software": "Python 3, NumPy, SciPy, scikit-learn"},
    "pro": {"label": "4 × H100 80GB in one node (NVLink); the evaluation uses the same node type",
            "device": "cuda", "software": "Python 3, PyTorch (CUDA, Triton, torch.distributed / torchrun), NumPy"},
}


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def render_task(s: dict) -> str:
    hw = HARDWARE[s["hardware"]["kind"]]
    a = s["anchor"]
    kind = s["oj"]["submission"]["kind"]
    parts = [f"# {s['id']}: {s['title']}\n",
             f"Knowledge cutoff: **{CUTOFF}**. You may only rely on material that was public and verifiable before that date.\n",
             f"Anchor: {a['authors']}, *{a['title']}* (arXiv:{a['arxiv']})."
             + (f" Baseline used for comparison: {agent_baseline(s)}." if s.get("baseline") else "") + "\n"]
    parts.append(_section("Task", s["task"].rstrip() + "\n\n" + render_workflow(s)))
    parts.append(_section("Data", s["data"]))
    parts.append(_section("Interface", s["interface"] + "\n\nThe full reference is `interface/interface.md`. "
                          "`abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own "
                          "pipeline and metric code; `selfcheck.py` runs them on a dev split."))
    parts.append(_section("Metrics", s["metrics"]))
    n = s["n_seeds"]
    reps = f"once for each of {n} seeds unknown to you" if n > 1 else "once"
    mode = s.get("mode", "judge_trains")
    if mode == "checkpoint":
        run = (f"loads your submitted checkpoint with your model code and runs inference through the stages in "
               f"`interface/stages.py`, {reps}, on the evaluation hardware; it never retrains your model and never runs "
               f"`train.py`")
    elif mode == "no_training":
        run = (f"runs your code through the stages in `interface/stages.py`, {reps}, on the evaluation hardware; "
               f"nothing is trained")
    else:
        run = (f"runs your training program from scratch through the stages in `interface/stages.py`, {reps}, on the "
               f"evaluation hardware, because the training itself is what this case measures")
    steps = [
        "When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.",
        f"The evaluator runs outside the sandbox: it checks the submission format, then {run}.",
        "Held-out inputs are copied into the run directory only after the timed phase has ended, and never with "
        "their labels or reference values; those stay in the evaluator process, which computes the metrics with "
        "`interface/metrics.py`.",
    ]
    if s.get("baseline_run"):
        steps.append(s["baseline_run"])
    steps.append("You receive no evaluation results during your session.")
    parts.append(_section("Evaluation", "\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1))))
    h = s["hardware"]
    guide = [f"The full guide is `interface/sandbox_guide.md`. Key points:\n",
             f"- {hw['label']}. No network and no package installation; available software: {h.get('software', hw['software'])}.",
             f"- The session lasts at most {h['session_hours']} hours; the deadline is in `$ABENCH_DEADLINE`. "
             f"Compute during development is capped at {h['dev_budget']}.",
             "- Data is read-only under `$ABENCH_DATA_DIR`"
             + (", frozen pretrained files under `$ABENCH_MODELS_DIR`" if s["oj"]["data"].get("models_dir") else "")
             + ". Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.",
             "- Paper search (offline, cutoff-filtered, logged): " + PAPERS + ".",
             "- Self-check: `python3 \"$ABENCH_VIEW_DIR/interface/selfcheck.py\" --quick` (interface, small slice), "
             "then without `--quick` on the full dev split" + (" with your checkpoint." if s.get("mode") == "checkpoint"
                                                                 else ".")]
    parts.append(_section("Sandbox Guide", "\n".join(guide)))
    parts.append(_section("Deliverables", "```\n" + s["deliverables"]["tree"].rstrip() + "\n```\n\n"
                          + s["deliverables"].get("notes", "").strip() + "\n"
                          "- Write the design report following the structure of `templates/report_template.html`: related "
                          "work and the gap you address, the changes relative to the anchor (`<ol id=\"changes\">`, each "
                          "with its code location), method definition, algorithm, complexity and design rationale. The "
                          "report explains the design only; it does not describe experiments or numeric results.\n"
                          "- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states "
                          "what was borrowed and where it is in the code; ids match the report's references one to one."))
    common = ["Do not read or write files outside the submission directory, the run directory and the paths the "
              "stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.",
              "Do not use extra processes or background threads to move work outside the timed phase, and do not leave "
              "processes running after a stage returns.",
              "Submission is one-shot. A submission that breaks the rules is invalid."]
    rules = s["rules"].strip() + "\n" + "\n".join(f"- {r}" for r in common)
    parts.append(_section("Rules", rules))
    return "\n".join(parts)


def agent_baseline(s: dict) -> str:
    """The baseline as the agent should read it: example_submission/ is not in the agent view; checkpoint cases show
    its code under interface/baseline/."""
    where = "`interface/baseline/`" if s.get("mode") == "checkpoint" else "the evaluator's reference submission, not shown"
    return s["baseline"].replace("`example_submission/`", where)


PAPERS = '`python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`'


def render_workflow(s: dict) -> str:
    """The working procedure every standard case asks for (retrieve → synthesise → design → develop → submit)."""
    mode = s.get("mode", "judge_trains")
    base = "the anchor and the baseline" if s.get("baseline") else "the anchor"
    develop = {
        "checkpoint": "Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the "
                      "sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. "
                      "The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a "
                      "starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.",
        "judge_trains": "Implement it and run the evaluator's own training pipeline on the dev split with `selfcheck.py`; use "
                        "the results to iterate as your budget allows. The evaluator then runs your submission from scratch "
                        "under the same pipeline.",
        "no_training": "Implement it, measure it on the dev setting with `selfcheck.py`, and iterate as your budget allows.",
    }[mode]
    steps = [
        f"**Retrieve.** Study the anchor and find related work published before the cutoff with the offline paper search "
        f"tool: {PAPERS} and `... show <arXiv id>` (titles and abstracts of arXiv papers up to the cutoff). Every call "
        f"is logged automatically; the log is kept with your run.",
        "**Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will "
        "address. This is Section 2 (Related Work) of the design report.",
        f"**Design.** Propose an improvement over {base}: at least one change you can state precisely and point to in your "
        "code. List every change, with its code location, in the report's \"Changes relative to the anchor\" list "
        "(`<ol id=\"changes\">`); a report without it fails the format check.",
        f"**Develop and verify.** {develop}",
        "**Submit** the deliverables below. Submission is one-shot.",
    ]
    return "### Workflow\n\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(steps, 1))


def render_guide(s: dict) -> str:
    hw = HARDWARE[s["hardware"]["kind"]]
    h = s["hardware"]
    has_models = bool(s["oj"]["data"].get("models_dir"))
    rows = [("`ABENCH_VIEW_DIR`", "Root of the task statement, interface, templates", "read-only"),
            ("`ABENCH_DATA_DIR`", "Public data of this case (see the Data section of the task)", "read-only")]
    if has_models:
        rows.append(("`ABENCH_MODELS_DIR`", "Frozen pretrained files allowed in this case", "read-only"))
    rows += [("`ABENCH_WORKSPACE`", "Your workspace: code drafts, logs, temporary files", "writable, deleted after the session"),
             ("`ABENCH_CHECKPOINT_DIR`", "Intermediate training results (never submitted)", "writable, deleted after the session"),
             ("`ABENCH_SUBMISSION_DIR`", "Final submission directory, collected when the session ends", "writable"),
             ("`ABENCH_DEVICE`", f"Device name; `{hw['device']}` in this sandbox", "—"),
             ("`ABENCH_DEADLINE`", "Session deadline (Unix seconds)", "—")]
    table = "| Variable | Path and purpose | Access |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    train_line = ("\npython3 \"$ABENCH_SUBMISSION_DIR/train.py\" --data \"$ABENCH_DATA_DIR\" --out \"$ABENCH_SUBMISSION_DIR/checkpoint\""
                  "   # your training program (the evaluator only loads the checkpoint)" if s.get("mode") == "checkpoint" else "")
    check_hw = ("nvidia-smi\npython3 -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())\""
                if hw["device"] == "cuda" else "nproc\npython3 -c \"import numpy, sklearn; print(numpy.__version__, sklearn.__version__)\"")
    return f"""# Sandbox guide ({s['id']})

## 1. Where you are

| | |
|---|---|
| Hardware | {hw['label']} |
| Network | None. You cannot download data or models or install packages |
| Software | {h.get('software', hw['software'])}; trust the versions reported in the sandbox |
| Session length | At most {h['session_hours']} hours wall clock; the deadline is in `$ABENCH_DEADLINE` (Unix seconds), and all processes are killed at that time |
| Compute | At most {h['dev_budget']} during development; the sandbox records usage, and exceeding it is a violation |

Check the environment first:

```bash
{check_hw}
python3 -c "import os,time; print(round((int(os.environ['ABENCH_DEADLINE'])-time.time())/60), 'min left')"
```

## 2. Directories and environment variables

{table}

`HOME` and `TMPDIR` both point to temporary directories inside the workspace. Paths outside the workspace are not writable and should not be read.

## 3. Recommended workflow

```bash
cd "$ABENCH_WORKSPACE"
ls -la "$ABENCH_DATA_DIR"{' "$ABENCH_MODELS_DIR"' if has_models else ''}
# ... develop, train and test in the workspace ...
python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<topic of the anchor>"   # related work before the cutoff
python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" show <arXiv id>                  # title, authors, date, abstract
mkdir -p "$ABENCH_SUBMISSION_DIR"      # copy the deliverables listed in the task statement here{train_line}
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick      # interface check on a small slice, 1 seed
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --seeds 2    # full dev split, 2 seeds
```

`selfcheck.py` runs your submission through the evaluator's own pipeline (`abench_stage.py`, `stages.py`) and metric
code (`metrics.py`) on a dev split carved out of the public data: every stage is a fresh process with empty caches,
timed by the parent process, and held-out inputs appear only after the timed phase. Dev metrics are an estimate of
the held-out metrics, not a promise.

## 4. Case notes

{s['hardware'].get('notes', 'None.').strip()}
"""


def submission_rules(s: dict) -> dict:
    sub = s["oj"]["submission"]
    req = [sub["entry"], "report/index.html", "references.yaml"] + list(sub.get("required_files", []))
    data_dirs, data_max = [], {}
    if sub["kind"] == "code_plus_checkpoint":
        req.append(sub["checkpoint_dir"])
        data_dirs.append(sub["checkpoint_dir"])
        if sub.get("checkpoint_max_bytes"):
            data_max[sub["checkpoint_dir"]] = sub["checkpoint_max_bytes"]
    return {"required": req, "allowed_top_level": sub["allowed_top_level"],
            "code_suffixes": sub.get("code_suffixes", [".py", ".yaml", ".yml", ".md", ".txt"]),
            "code_max_bytes": sub.get("code_max_bytes", 200_000), "data_dirs": data_dirs, "data_max_bytes": data_max}


def baseline_sources(case_dir: Path, s: dict) -> dict[Path, Path]:
    """Checkpoint cases: the reference submission's code (not its report or checkpoint) is shown to the agent under
    interface/baseline/, as the starting point to improve on."""
    if s.get("mode") != "checkpoint":
        return {}
    ex = case_dir / "example_submission"
    out = {}
    for p in sorted(ex.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[case_dir / "harness" / "baseline" / p.relative_to(ex)] = p
    return out


def baseline_files(case_dir: Path, s: dict) -> list[str]:
    return [str(d.relative_to(case_dir)) for d in baseline_sources(case_dir, s)]


def merge_oj(case_dir: Path, s: dict) -> str:
    p = case_dir / "case" / "oj.yaml"
    oj = {"schema": "abench.oj/v2", "case_id": s["id"], "standard": True, "task_statement": "v2.1",
          "knowledge_cutoff": CUTOFF, "mode": s.get("mode", "judge_trains")}
    oj["agent_visible"] = {
        "task": "case/agent/task.md",
        "interface": ["harness/interface.md", "harness/sandbox_guide.md", "harness/stages.py", "harness/metrics.py",
                      "harness/abench_stage.py", "harness/abench_metrics.py", "harness/selfcheck.py",
                      "harness/submission_rules.json"] + s.get("extra_interface", []) + baseline_files(case_dir, s),
        "leak_allow": ["hidden", "evaluation", "metric", "threshold", "score", "evaluate", "held-out"]
                      + s.get("leak_allow", []),
    }
    oj["agent_budget"] = s["oj"]["agent_budget"]
    oj["data"] = s["oj"]["data"]
    sub = dict(s["oj"]["submission"])
    sub.setdefault("report_profile", "design_report_v2")
    oj["submission"] = sub
    judge = {"kind": "harness", "activation": "awaiting_data", "stages": "harness/stages.py",
             "metrics": "harness/metrics.py"}
    judge.update(s["oj"]["judge"])
    oj["judge"] = judge
    # merge the operator's frozen hashes (written by tools/prepare/run.py)
    lock_p = case_dir / "case" / "data_lock.yaml"
    if lock_p.is_file():
        lock = yaml.safe_load(lock_p.read_text(encoding="utf-8")) or {}
        d = dict(oj["data"])
        for k in ("public_dir", "public", "hidden", "models_dir", "models", "sources_sha256"):
            if k in lock:
                d[k] = lock[k]
        d["synthetic"] = bool(lock.get("synthetic"))
        oj["data"] = d
        if lock.get("activation") == "active" and not lock.get("synthetic"):
            oj["judge"]["activation"] = "active"
    head = (f"# {s['id']} OJ contract (standard case, generated by tools/build_standard_cases.py from "
            f"spec/standard_cases/{s['id']}.yaml; edit the spec, not this file)\n")
    return head + yaml.safe_dump(oj, allow_unicode=True, sort_keys=False, width=110)


def render_readme(s: dict) -> str:
    oj = s["oj"]
    a = s["anchor"]
    j = oj["judge"]
    srcs = "\n".join(f"- {x['name']}: {x['url']} — license: {x.get('license', '?')}"
                      + (f" (verified {x['verified'].split(':')[0]})" if x.get("verified") else "")
                      for x in oj["data"].get("sources", []))
    baseline = f"\n- Baseline run by the judge alongside every submission: `{j['baseline']}`." if j.get("baseline") else ""
    extra = ("\n- `SCORING_DEMO.md` keeps the earlier ABench-level scoring design and its simulated demo (`sim/`, `outputs/`)."
             if (ROOT / "cases" / s["id"] / "SCORING_DEMO.md").is_file() else "")
    return f"""# ABench standard case {s['id']}: {s['title']}

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: {a['authors']}, *{a['title']}*
(arXiv:{a['arxiv']}).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): {s.get('reference_verification', 'not yet run')}
- No number in the standard-case files is a result: there are no training curves or reference scores.{baseline}{extra}

## Files

- `spec/standard_cases/{s['id']}.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case {s['id']}`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/{s['id'].lower().replace('-', '_')}.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

{srcs}

## Activation (operator)

```bash
python3 tools/prepare/run.py --case {s['id']} --list-sources                 # what to download
python3 tools/prepare/run.py --case {s['id']} --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case {s['id']} --submission cases/{s['id']}/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case {s['id']} --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
"""


def build(spec_path: Path, check: bool) -> list[str]:
    s = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    cdir = ROOT / "cases" / s["id"]
    outs = {cdir / "case/agent/task.md": render_task(s),
            cdir / "harness/interface.md": s["interface_md"].rstrip() + "\n",
            cdir / "harness/sandbox_guide.md": render_guide(s),
            cdir / "harness/submission_rules.json": json.dumps(submission_rules(s), indent=2) + "\n",
            cdir / "case/oj.yaml": merge_oj(cdir, s),
            cdir / "README.md": render_readme(s)}
    for f in KIT_FILES:
        outs[cdir / "harness" / f] = (KIT / f).read_text(encoding="utf-8")
    for dst, src in baseline_sources(cdir, s).items():
        outs[dst] = src.read_text(encoding="utf-8")
    stale = []
    for p, text in outs.items():
        cur = p.read_text(encoding="utf-8") if p.is_file() else None
        if cur != text:
            stale.append(str(p.relative_to(ROOT)))
            if not check:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
    for need in ("harness/stages.py", "harness/metrics.py"):
        if not (cdir / need).is_file():
            stale.append(f"cases/{s['id']}/{need} (missing, hand-written)")
    return stale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    specs = [SPECS / f"{a.case}.yaml"] if a.case else sorted(SPECS.glob("*.yaml"))
    bad = 0
    for sp in specs:
        stale = build(sp, a.check)
        if stale:
            bad += 1
            print(f"{sp.stem}: {'out of date' if a.check else 'written'}: {', '.join(stale)}")
    print(f"standard cases: {len(specs)}" + (f", out of date: {bad}" if a.check else ""))
    return 1 if a.check and bad else 0


if __name__ == "__main__":
    sys.exit(main())

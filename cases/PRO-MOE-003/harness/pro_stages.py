"""Audit and sampled-resume stages shared by the Pro Set cases that deliver a trained checkpoint (evaluator code,
agent-visible). See pro_audit.py for how registered runs work.

The session directory (environment variable ABENCH_PRO_SESSION, set by the sandbox runner for the evaluator and for
selfcheck.py) holds audit/ (chain.jsonl and one directory of metrics, stdout and telemetry per run, written only by
the supervisor) and checkpoints/<run>/<step dir>/ (every registered checkpoint, kept until judging).

Stage "audit" (evaluator code only):
  the chain is intact and has no gap of more than 35 minutes between registrations of a run; the submitted
  checkpoint/ equals model/ of a registered checkpoint; the run that produced it started from the allowlisted
  weights (or resumed from a registered checkpoint that did); the submitted program files (.py, .sh, .yaml, .yml,
  .json, .toml outside checkpoint/ and report/) are those the run was started with (pro_audit.code_hash).
Stage "resume" (runs the submission's own training command):
  a registered checkpoint of the delivering run, picked with the secret number resume_u, is hashed again and
  resumed with ABENCH_RESUME_FROM for at most RESUME_SECONDS; the logged metric of the resume must match the
  original log (pro_audit.trajectory_match).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import pro_audit as A


def session() -> Path:
    s = os.environ.get("ABENCH_PRO_SESSION")
    if not s:
        raise SystemExit("ABENCH_PRO_SESSION is not set: the audited session directory is required")
    return Path(s)


def audit(submission: Path, start_weights: Path | None) -> dict:
    """start_weights: the allowlisted starting model, or None for training from random initialisation (the run's
    start entry must then record no starting weights)."""
    sess = session()
    ents = A.Chain(sess / "audit" / "chain.jsonl").entries()
    errs = A.verify_chain(ents)
    t = A.trace(ents, A.weights_hashes(submission / "checkpoint"),
                start_hash=A.tree_hash(A.weights_hashes(start_weights)) if start_weights else None)
    code_ok = 0
    if t.get("ok"):
        start = next(e for e in ents if e["kind"] == "start" and e["run"] == t["run"])
        code_ok = int(A.code_hash(submission) == start["code"])
    runs = sorted({e["run"] for e in ents if e["kind"] == "start"})
    info = {"chain_errors": errs, "trace": t.get("why", ""), "run": t.get("run"), "runs": runs,
            "checkpoint": (t.get("checkpoint") or {}).get("dir")}
    return {"chain_ok": np.asarray([int(not errs)]), "trace_ok": np.asarray([int(bool(t.get("ok")))]),
            "code_ok": np.asarray([code_ok]), "n_runs": np.asarray([len(runs)]),
            "audit_json": np.asarray([json.dumps(info)])}


def resume(submission: Path, work: Path, u: float, metric: str, rel: float, max_seconds: float = A.RESUME_SECONDS) -> dict:
    sess = session()
    ents = A.Chain(sess / "audit" / "chain.jsonl").entries()
    t = A.trace(ents, A.weights_hashes(submission / "checkpoint"), "")
    run = (t.get("checkpoint") or {}).get("run")
    if run is None:
        return {"resume_ok": np.asarray([0]), "resume_json": np.asarray([json.dumps({"why": "no traced run"})])}
    orig = A.read_metrics(sess / "audit", run)
    pt = A.pick_resume_point(ents, run, orig, u)
    if pt is None:
        return {"resume_ok": np.asarray([0]), "resume_json": np.asarray([json.dumps({"why": "no resumable checkpoint"})])}
    src = sess / "checkpoints" / run / pt["dir"]
    if A.file_hashes(src) != pt["files"]:
        return {"resume_ok": np.asarray([0]), "resume_json": np.asarray([json.dumps({"why": "checkpoint changed"})])}
    start = next(e for e in ents if e["kind"] == "start" and e["run"] == run)
    code = A.supervise("resume", list(start["cmd"]), work / "audit", work / "ckpt", submission, None,
                       resume_from=pt["hash"], resume_dir=str(src), max_seconds=max_seconds)
    res = A.read_metrics(work / "audit", "resume")
    m = A.trajectory_match([r for r in orig if r["step"] >= int(pt["step"])], res, metric, rel=rel)
    m.update({"from": pt["dir"], "returncode": code})
    return {"resume_ok": np.asarray([int(m["ok"])]), "resume_json": np.asarray([json.dumps(m)])}


def scores(outputs: dict) -> dict:
    a, r = outputs.get("audit", {}), outputs.get("resume", {})
    return {"chain_ok": float(np.asarray(a.get("chain_ok", [0]))[0]), "trace_ok": float(np.asarray(a.get("trace_ok", [0]))[0]),
            "code_ok": float(np.asarray(a.get("code_ok", [0]))[0]), "resume_ok": float(np.asarray(r.get("resume_ok", [0]))[0])}


GATES = [{"metric": "chain_ok", "min": 1.0}, {"metric": "trace_ok", "min": 1.0}, {"metric": "code_ok", "min": 1.0},
         {"metric": "resume_ok", "min": 1.0}]

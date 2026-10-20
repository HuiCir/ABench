"""OJ-style automatic evaluation: read only the submitted model code and checkpoint, and verify on the hidden set.

Flow
  1. The submission first passes the format check (tools/validate_submission.py).
  2. Decrypt the hidden set into the judge's own temporary directory (permissions 700) and split it into inputs and labels.
  3. Copy the submission into another temporary directory and start the child with a minimal environment, a timeout and resource limits
     (tools/oj_predict_child.py or a case runner): the child gets only the copy of the submission and the unlabeled inputs.
  4. The parent computes metrics with the labels held in memory and writes evaluation.json. All temporary directories are deleted.

Results are written only to the operator's result directory; the agent stage has long ended, so results never reach the agent's view.

Usage (normally called by sandbox_runner):
  python3 tools/oj_eval.py --case DS-GBM-001 --submission <dir> --hidden-key-file <key> --out <evaluation.json>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import hidden_store  # noqa: E402
from validate_submission import validate_submission_dir  # noqa: E402

JUDGE_VERSION = "abench-oj-judge-v2.0"


def _dependency_attestation() -> dict[str, Any]:
    """Describe the numerical runtime used by the judge child.

    A benchmark deployment should point this at a frozen venv/container
    runtime with ``ABENCH_JUDGE_PYTHONPATH``.  The operator user-site fallback
    remains available for local plumbing, but is explicitly non-scoreable.
    """
    expected_tools = os.environ.get("ABENCH_JUDGE_TOOLS_SHA256", "").lower()
    actual_tools = tree_hash(TOOLS)
    tools_attested = bool(expected_tools) and expected_tools == actual_tools
    raw = os.environ.get("ABENCH_JUDGE_PYTHONPATH", "")
    if raw:
        paths = [str(Path(p).expanduser().resolve()) for p in raw.split(os.pathsep) if p]
        missing = [p for p in paths if not Path(p).is_dir()]
        try:
            user = str(Path(site.getusersitepackages()).resolve())
        except Exception:
            user = ""
        path_trusted = bool(paths) and not missing and all(p != user and not p.startswith(user + os.sep) for p in paths)
        trusted = path_trusted and tools_attested
        return {"source": "explicit_path", "paths": paths, "trusted": trusted,
                "tools_sha256": actual_tools, "tools_sha256_expected": expected_tools or None,
                "tools_attested": tools_attested, **({"missing": missing} if missing else {})}
    try:
        user = Path(site.getusersitepackages()).resolve()
    except Exception:
        user = None
    if user and user.is_dir():
        return {"source": "operator_user_site", "paths": [str(user)], "trusted": False,
                "tools_sha256": actual_tools, "tools_sha256_expected": expected_tools or None,
                "tools_attested": tools_attested}
    return {"source": "interpreter_default", "paths": [], "trusted": False,
            "tools_sha256": actual_tools, "tools_sha256_expected": expected_tools or None,
            "tools_attested": tools_attested}


def _dependency_path() -> str | None:
    attestation = _dependency_attestation()
    paths = attestation.get("paths") or []
    return os.pathsep.join(paths) if paths else None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*")
                    if x.is_file() and not x.is_symlink() and "__pycache__" not in x.parts and x.suffix != ".pyc"):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def load_oj(case: str) -> dict[str, Any]:
    p = ROOT / "cases" / case / "case" / "oj.yaml"
    if not p.is_file():
        raise SystemExit(f"{case}: case/oj.yaml not found")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ metrics
def accuracy(y: np.ndarray, P: np.ndarray) -> float:
    return float((P.argmax(1) == y).mean())


def log_loss(y: np.ndarray, P: np.ndarray, eps: float = 1e-15) -> float:
    Q = np.clip(P, eps, 1.0)
    Q = Q / Q.sum(1, keepdims=True)
    return float(-np.log(Q[np.arange(len(y)), y]).mean())


def _auc_binary(y01: np.ndarray, s: np.ndarray) -> float:
    pos, neg = y01 == 1, y01 == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # ties: tied values get the average rank
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def roc_auc(y: np.ndarray, P: np.ndarray) -> float:
    k = P.shape[1]
    if k == 2:
        return _auc_binary((y == 1).astype(int), P[:, 1])
    vals = [_auc_binary((y == c).astype(int), P[:, c]) for c in range(k)]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


METRICS = {"accuracy": accuracy, "log_loss": log_loss, "roc_auc": roc_auc}


# ------------------------------------------------------------------ child process
def _limits(cpu_seconds: int | None, memory_mb: int | None):
    def fn():
        try:
            import resource
            if cpu_seconds:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            if memory_mb and sys.platform != "darwin":   # RLIMIT_AS has no effect on macOS
                b = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
        except Exception:
            pass
        os.setsid()
    return fn


def run_child(sub_copy: Path, x_path: Path, out_path: Path, oj: dict, workdir: Path) -> dict[str, Any]:
    j = oj["judge"]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(workdir), "TMPDIR": str(workdir),
           "LANG": "C.UTF-8", "OMP_NUM_THREADS": str(j.get("threads", 1)), "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "no_proxy": "*", "http_proxy": "http://127.0.0.1:9",
           "https_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"}
    dep_path = _dependency_path()
    if dep_path:
        env["PYTHONPATH"] = dep_path
    # Do not use ``-E`` here: it would discard the narrowly scoped
    # PYTHONPATH above and hide the bundled numerical runtime when HOME is
    # replaced by the judge's temporary directory.
    cmd = [sys.executable, str(TOOLS / "oj_predict_child.py"), "--submission", str(sub_copy),
           "--entry", oj["submission"]["entry"], "--checkpoint-dir", oj["submission"].get("checkpoint_dir", "checkpoint"),
           "--x", str(x_path), "--out", str(out_path)]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, timeout=j.get("timeout_sec", 300),
                           preexec_fn=_limits(j.get("cpu_seconds"), j.get("memory_mb")) if os.name == "posix" else None)
        status = "ok" if r.returncode == 0 else "runtime_error"
        out, err, code = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        status, out, err, code = "time_limit_exceeded", e.stdout or b"", e.stderr or b"", None
    return {"status": status, "exit_code": code, "wall_seconds": round(time.perf_counter() - t0, 4),
            "stdout_tail": out[-2000:].decode("utf-8", "replace"), "stderr_tail": err[-4000:].decode("utf-8", "replace")}


# ------------------------------------------------------------------ main flow
def _load_case_runner(case: str, rel: str):
    """Load the case's own timing parent (the same code the self-check in the agent view uses)."""
    import importlib.util
    path = (ROOT / "cases" / case / rel).resolve()
    if not str(path).startswith(str((ROOT / "cases" / case).resolve()) + os.sep) or not path.is_file():
        raise SystemExit(f"judge runner not found inside case: {rel}")
    spec = importlib.util.spec_from_file_location(f"abench_runner_{case.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def evaluate_timed(case: str, submission: Path, key_file: Path, oj: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """timed_train_classification: train from scratch once per hidden seed, timed by the parent, and compute accuracy on the hidden test set.

    The submission is code only; the judge starts a fresh process per seed (speedrun_runner.run_once), the training set comes from the public data,
    test images are given to the child without labels, and labels stay in this process's memory only.
    """
    j = oj["judge"]
    runner, runner_path = _load_case_runner(case, j["runner"])
    rec["judge_harness_sha256"] = {p.name: sha256_bytes(p.read_bytes())
                                   for p in (runner_path, Path(runner.HARNESS))}
    hid, pub = oj["data"]["hidden"], oj["data"]["public"][0]
    plain = hidden_store.decrypt_bytes(hidden_store.read_key(key_file), (ROOT / hid["path"]).read_bytes())
    if sha256_bytes(plain) != hid["sha256_plain"]:
        raise SystemExit("hidden split plaintext hash mismatch")
    with np.load(io.BytesIO(plain), allow_pickle=False) as z:
        X, y, src = np.asarray(z["x"]), np.asarray(z["y"]).astype(np.int64), np.asarray(z["src"])
    del plain
    pub_bytes = (ROOT / pub["path"]).read_bytes()
    if sha256_bytes(pub_bytes) != pub["sha256"]:
        raise SystemExit("public train hash mismatch")
    subsets = j.get("subsets", {"cifar10": 0, "cifar10_1": 1})
    tmp = Path(tempfile.mkdtemp(prefix="abench-judge-"))
    os.chmod(tmp, 0o700)
    try:
        sub_copy = tmp / "submission"
        shutil.copytree(submission, sub_copy)
        with np.load(io.BytesIO(pub_bytes), allow_pickle=False) as z:
            np.savez(tmp / "train.npz", x=z["x"], y=z["y"])           # all 50k training images, without the dev-split information
        np.save(tmp / "x_test.npy", X, allow_pickle=False)
        dep = _dependency_path()
        runs = []
        for i, seed in enumerate(j["seeds"]):
            r = runner.run_once(sub_copy, tmp / "train.npz", tmp / "x_test.npy", int(seed), tmp / f"run{i}",
                                entry=oj["submission"]["entry"], device=j.get("device", "cuda"),
                                timeout=float(j.get("timeout_sec", 600)), predict_cap=float(j.get("predict_cap_sec", 5)),
                                tail_slack=float(j.get("tail_slack_sec", 3)), threads=int(j.get("threads", 8)),
                                extra_env={"PYTHONPATH": dep} if dep else None)
            if r["status"] == "ok":
                P = np.load(r.pop("pred_path"), allow_pickle=False)
                if P.shape != (len(y), int(j.get("n_classes", 10))) or not np.isfinite(P).all():
                    r["status"] = "wrong_output_format"
                    r["detail"] = f"expected finite ({len(y)}, {j.get('n_classes', 10)}), got {list(P.shape)}"
                else:
                    pred = P.argmax(1)
                    r["accuracy"] = {k: round(float((pred[src == v] == y[src == v]).mean()), 6) for k, v in subsets.items()}
            r.pop("pred_path", None)
            runs.append(r)
            if r["status"] != "ok":
                break                                                   # if any seed fails, the whole submission is invalid
        rec["runs"] = runs
        rec["n_hidden"] = {k: int((src == v).sum()) for k, v in subsets.items()}
        bad = [r for r in runs if r["status"] != "ok"]
        if bad:
            rec.update(verdict=bad[0]["status"], metrics=None)
            return rec
        mean = lambda xs: round(float(np.mean(xs)), 6)
        ts = [r["train_seconds"] for r in runs]
        m = {"train_seconds_mean": mean(ts), "train_seconds_median": round(float(np.median(ts)), 4),
             "train_seconds_max": max(ts)}
        for k in subsets:
            m[f"accuracy_{k}_mean"] = mean([r["accuracy"][k] for r in runs])
        gates = {g["metric"]: {"threshold": g["min"], "value": m[g["metric"]], "pass": m[g["metric"]] >= g["min"]}
                 for g in j.get("gates", [])}
        m["valid"] = all(g["pass"] for g in gates.values())
        rec.update(verdict="evaluated", metrics=m, gates=gates, n_seeds=len(runs))
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verify_public(case: str, oj: dict[str, Any]) -> Path | None:
    """Check the frozen public files (sha256 per file) and return the public data directory."""
    data = oj.get("data") or {}
    pdir = data.get("public_dir")
    if not pdir:
        return None
    root = (ROOT / pdir).resolve()
    for f in data.get("public") or []:
        p = ROOT / f["path"]
        if not p.is_file() or sha256_bytes(p.read_bytes()) != f["sha256"]:
            raise SystemExit(f"public data hash mismatch: {f['path']}")
    return root


def _verify_models(oj: dict[str, Any]) -> Path | None:
    data = oj.get("data") or {}
    mdir = data.get("models_dir")
    if not mdir:
        return None
    root = (ROOT / mdir).resolve()
    for m in data.get("models") or []:
        p = ROOT / m["path"]
        if not p.is_dir() or tree_hash(p) != m["tree_sha256"]:
            raise SystemExit(f"frozen model tree hash mismatch: {m['path']}")
    return root


def _aggregate(per_seed: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for m in per_seed for k in m})
    return {k: round(float(np.mean([m[k] for m in per_seed if k in m])), 6) for k in keys}


def _gates(gates: list[dict], m: dict[str, float]) -> dict[str, Any]:
    out = {}
    for g in gates:
        v = m.get(g["metric"])
        ok = v is not None and np.isfinite(v) and ("min" not in g or v >= g["min"]) and ("max" not in g or v <= g["max"])
        out[g["metric"]] = {"min": g.get("min"), "max": g.get("max"), "value": v, "pass": bool(ok)}
    return out


def evaluate_harness(case: str, submission: Path, key_file: Path, oj: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """harness: run the case pipeline (harness/stages.py) through the shared stage runner, once per hidden seed.

    The held-out inputs are copied into a run directory only after the timed phase has ended; labels and reference
    values stay in this process and are scored by the case's harness/metrics.py. If the case declares a baseline
    submission, it is run first under identical conditions and relative metrics are added.
    """
    j = oj["judge"]
    cdir = ROOT / "cases" / case
    kit = cdir / "harness" / "abench_stage.py"
    stages_path = cdir / j.get("stages", "harness/stages.py")
    metrics_path = cdir / j.get("metrics", "harness/metrics.py")
    for p in (kit, stages_path, metrics_path):
        if not p.resolve().is_file() or not str(p.resolve()).startswith(str(cdir.resolve()) + os.sep):
            raise SystemExit(f"judge file not found inside case: {p}")
    sys.path.insert(0, str(kit.parent))
    import importlib
    abench_stage = importlib.import_module("abench_stage") if "abench_stage" not in sys.modules else sys.modules["abench_stage"]
    stages = abench_stage.load_module(stages_path, f"abench_stages_{case.replace('-', '_')}")
    metrics = abench_stage.load_module(metrics_path, f"abench_metrics_{case.replace('-', '_')}")
    rec["judge_harness_sha256"] = {p.name: sha256_bytes(p.read_bytes()) for p in
                                   (kit, stages_path, metrics_path, kit.parent / "abench_metrics.py")}
    public = _verify_public(case, oj)
    models = _verify_models(oj)
    hid = oj["data"]["hidden"]
    plain = hidden_store.decrypt_bytes(hidden_store.read_key(key_file), (ROOT / hid["path"]).read_bytes())
    if sha256_bytes(plain) != hid["sha256_plain"]:
        raise SystemExit("hidden split plaintext hash mismatch")
    with np.load(io.BytesIO(plain), allow_pickle=False) as z:
        hidden = {k: np.asarray(z[k]) for k in z.files}
    del plain
    in_keys = list(getattr(stages, "HELD_OUT_KEYS", []))
    missing = [k for k in in_keys if k not in hidden]
    if missing:
        raise SystemExit(f"hidden split lacks input keys {missing}")
    reference = dict(hidden)                                     # stays in this process only
    make_inputs = getattr(stages, "held_out_inputs", None)     # optional: build model inputs in this process
    held_inputs = make_inputs(hidden) if make_inputs else {k: hidden[k] for k in in_keys}
    tmp = Path(tempfile.mkdtemp(prefix="abench-judge-"))
    os.chmod(tmp, 0o700)
    dep = _dependency_path()
    extra_env = {"PYTHONPATH": dep} if dep else None
    params = getattr(stages, "PARAMS", {})

    def run_all(sub: Path, tag: str) -> tuple[list[dict], list[dict], str | None]:
        sub_copy = tmp / f"{tag}_submission"
        shutil.copytree(sub, sub_copy)
        held = tmp / "held_out_inputs.npz"
        if held_inputs and not held.exists():
            np.savez(held, **held_inputs)
        runs, per_seed = [], []
        for i, seed in enumerate(j["seeds"]):
            res = abench_stage.run_pipeline(stages, stages_path, sub_copy, tmp / f"{tag}_run{i}", public=public,
                                            models=models, held_out=held if held_inputs else None, seed=int(seed),
                                            device=j.get("device", "cuda"), extra_env=extra_env)
            row = {"seed_index": i, "status": res["status"], "seconds": res["seconds"],
                   "stages": [{k: v for k, v in s.items() if k not in ("meta",)} | {"meta": s.get("meta")}
                              for s in res["stages"]]}
            if res["status"] == "ok":
                try:
                    m = metrics.score(res["outputs"], reference, params, res["seconds"])
                    per_seed.append(m)
                    row["metrics"] = m
                except metrics.OutputError as e:
                    row["status"], row["detail"] = "wrong_output_format", str(e)
            runs.append(row)
            shutil.rmtree(tmp / f"{tag}_run{i}", ignore_errors=True)
            if row["status"] != "ok":
                return runs, per_seed, row["status"]
        return runs, per_seed, None

    try:
        base_agg = None
        if j.get("baseline"):
            bdir = (cdir / j["baseline"]).resolve()
            b_runs, b_seed, b_bad = run_all(bdir, "baseline")
            rec["baseline"] = {"path": j["baseline"], "tree_sha256": tree_hash(bdir), "runs": b_runs}
            if b_bad:
                raise SystemExit(f"baseline failed ({b_bad}); the judge environment is not valid for this case")
            base_agg = _aggregate(b_seed)
            rec["baseline"]["metrics"] = base_agg
        runs, per_seed, bad = run_all(submission, "submission")
        rec["runs"] = runs
        rec["n_hidden"] = {k: int(len(v)) for k, v in hidden.items() if hasattr(v, "__len__")}
        if bad:
            rec.update(verdict=bad, metrics=None)
            return rec
        m = _aggregate(per_seed)
        if base_agg is not None and hasattr(metrics, "relative"):
            m.update({k: round(float(v), 6) for k, v in metrics.relative(m, base_agg).items()})
        gates = _gates(list(getattr(metrics, "GATES", [])), m)
        m["valid"] = all(g["pass"] for g in gates.values())
        rec.update(verdict="evaluated", metrics=m, gates=gates, primary=getattr(metrics, "PRIMARY", None),
                   n_seeds=len(per_seed))
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(case: str, submission: Path, key_file: Path | None, run_mode: str = "benchmark") -> dict[str, Any]:
    oj = load_oj(case)
    j = oj.get("judge", {})
    dependency = _dependency_attestation()
    rec: dict[str, Any] = {"schema_version": "2.0", "case_id": case, "judge_version": JUDGE_VERSION,
                           "provenance": "judge_real", "run_mode": run_mode,
                           # The parent evaluator and predictor currently run
                           # under the operator account.  Keep this explicit
                           # so a benchmark record cannot be mistaken for a
                           # fully container-isolated judge.
                           "judge_isolation": "host_soft",
                           "scorable": run_mode == "benchmark",
                           "judge_dependency": dependency,
                           "submission_tree_sha256": tree_hash(submission),
                           "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if run_mode == "benchmark" and not dependency.get("trusted"):
        rec["scorable"] = False
        rec["scoreability_reason"] = (
            "judge requires a frozen ABENCH_JUDGE_PYTHONPATH outside the operator user-site "
            "and a matching ABENCH_JUDGE_TOOLS_SHA256"
        )
    errors = validate_submission_dir(submission, case)
    if errors:
        rec.update(verdict="submission_invalid", errors=errors, metrics=None)
        return rec
    if run_mode == "smoke":
        # A smoke run verifies the view/collection/report contract only.  Do not
        # decrypt hidden data or execute the judge: even a real-looking metric
        # from a smoke run must never be mistaken for a benchmark result.
        rec.update(verdict="smoke_unscored", metrics=None,
                   note="smoke runs are contract checks only and are never scored")
        return rec
    if run_mode != "benchmark":
        rec.update(verdict="submission_invalid", metrics=None,
                   errors=[f"unsupported run_mode={run_mode!r}; only benchmark or smoke is allowed"])
        return rec
    if j.get("kind") in (None, "pending"):
        rec.update(verdict="judge_pending", metrics=None,
                   note="this case's judge is not frozen yet; only the format check was done")
        return rec
    if j.get("activation") == "awaiting_data":
        data = oj.get("data") or {}
        hashed = bool(re.fullmatch(r"[0-9a-f]{64}", str((data.get("hidden") or {}).get("sha256_plain", ""))))
        frozen = (not data.get("synthetic")) and hashed
        # plumbing check of the whole launch path on synthetic data: explicit opt-in, recorded, never scorable
        synthetic_ok = bool(data.get("synthetic")) and hashed and os.environ.get("ABENCH_ALLOW_SYNTHETIC") == "1"
        if not (frozen or synthetic_ok):
            rec.update(verdict="judge_pending", metrics=None,
                       note="this case's data has not been frozen by the operator yet (judge.activation: awaiting_data); "
                            "only the format check was done")
            return rec
        # real data frozen but the case is not activated yet: a calibration run, never scorable
        rec["scorable"] = False
        rec["calibration"] = True
        rec["scoreability_reason"] = "calibration run: case not activated yet (judge.activation: awaiting_data)"
        if synthetic_ok:
            rec["synthetic_data"] = True
            rec["scoreability_reason"] = "plumbing run on SYNTHETIC data (ABENCH_ALLOW_SYNTHETIC=1); metrics are meaningless"
    if j["kind"] == "harness":
        if key_file is None:
            raise SystemExit("--hidden-key-file is required to evaluate on the hidden split")
        return evaluate_harness(case, submission, key_file, oj, rec)
    if j["kind"] == "timed_train_classification":
        if key_file is None:
            raise SystemExit("--hidden-key-file is required to evaluate on the hidden split")
        return evaluate_timed(case, submission, key_file, oj, rec)
    if j["kind"] != "tabular_classification":
        raise SystemExit(f"unsupported judge kind {j['kind']!r}")
    if key_file is None:
        raise SystemExit("--hidden-key-file is required to evaluate on the hidden split")

    hid = oj["data"]["hidden"]
    blob = (ROOT / hid["path"]).read_bytes()
    plain = hidden_store.decrypt_bytes(hidden_store.read_key(key_file), blob)
    if sha256_bytes(plain) != hid["sha256_plain"]:
        raise SystemExit("hidden split plaintext hash mismatch")
    with np.load(io.BytesIO(plain), allow_pickle=False) as z:
        X, y = np.asarray(z["x"]), np.asarray(z["y"]).astype(np.int64)
    del plain

    tmp = Path(tempfile.mkdtemp(prefix="abench-judge-"))
    os.chmod(tmp, 0o700)
    try:
        sub_copy = tmp / "submission"
        shutil.copytree(submission, sub_copy)
        x_path, out_path = tmp / "x.npy", tmp / "pred.npy"
        np.save(x_path, X, allow_pickle=False)
        child = run_child(sub_copy, x_path, out_path, oj, tmp)
        rec["execution"] = child
        if child["status"] != "ok":
            rec.update(verdict=child["status"], metrics=None)
            return rec
        P = np.load(out_path, allow_pickle=False)
        k = int(j.get("n_classes", int(y.max()) + 1))
        if P.shape != (len(y), k) or not np.isfinite(P).all() or (P < -1e-9).any():
            rec.update(verdict="wrong_output_format", metrics=None,
                       detail=f"expected finite non-negative ({len(y)}, {k}), got {list(P.shape)}")
            return rec
        rec["metrics"] = {m: round(METRICS[m](y, P), 6) for m in j.get("metrics", ["accuracy"])}
        rec["n_hidden"] = int(len(y))
        rec["verdict"] = "evaluated"
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        del X, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--hidden-key-file")
    ap.add_argument("--run-mode", choices=["benchmark", "smoke"], default="benchmark")
    ap.add_argument("--out")
    a = ap.parse_args()
    rec = evaluate(a.case, Path(a.submission).resolve(), Path(a.hidden_key_file) if a.hidden_key_file else None, a.run_mode)
    text = json.dumps(rec, ensure_ascii=False, indent=2)
    if a.out:
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f"evaluation record already exists and is immutable: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if rec.get("verdict") in ("evaluated", "judge_pending", "smoke_unscored") else 1


if __name__ == "__main__":
    sys.exit(main())

"""Standard cases: generated files are up to date, every case is complete, reference submissions pass the format
check, and the cases whose reference submission needs only NumPy/scikit-learn run end to end on synthetic data
through the self-check path and (for one case) through the judge itself. No repository file is modified."""
from __future__ import annotations

import ast
import importlib
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "harness_kit"))
import abench_stage  # noqa: E402
from validate_submission import validate_submission_dir  # noqa: E402

SPECS = sorted((ROOT / "spec" / "standard_cases").glob("*.yaml"))
CASES = [p.stem for p in SPECS]
NUMPY_CASES = {"TC-FT-001": 1500, "DS-PIPE-002": 600, "RS-SEQ-001": 300, "GN-OGB-002": 1500, "FN-LOB-002": 12000,
               "MD-MNIST-003": 600, "BI-DNA-002": 400, "PH-NO-001": 120, "LG-HOLD-002": 800, "U-LOOP-003": 200}


def names_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            out.update(a.asname or a.name for a in node.names)
    return out


def prepare_module(case: str):
    return importlib.import_module("prepare." + case.lower().replace("-", "_"))


def submission_kind(case: str) -> str:
    oj = yaml.safe_load((ROOT / "cases" / case / "case" / "oj.yaml").read_text(encoding="utf-8"))
    return oj["submission"]["kind"]


def copy_example(case: str, dst: Path) -> Path:
    shutil.copytree(ROOT / "cases" / case / "example_submission", dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def train_checkpoint(case: str, sub: Path, public: Path) -> None:
    """Checkpoint cases: run the reference train.py (quick mode) on the given public data, as an agent would."""
    if submission_kind(case) != "code_plus_checkpoint":
        return
    with tempfile.TemporaryDirectory() as view:            # what the sandbox exposes as $ABENCH_VIEW_DIR/interface
        shutil.copytree(ROOT / "cases" / case / "harness", Path(view) / "interface")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "ABENCH_VIEW_DIR": view}
        r = subprocess.run([sys.executable, "train.py", "--data", str(public or view), "--out", str(sub / "checkpoint"),
                            "--quick"], cwd=sub, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{case} train.py failed:\n{r.stderr[-3000:]}"
    shutil.rmtree(sub / "__pycache__", ignore_errors=True)


def materialise(fz, public: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    for name, arrays in fz.public.items():
        np.savez(public / name, **arrays)
    for name, src in fz.raw_files.items():
        shutil.copyfile(src, public / name)


class StandardCaseFiles(unittest.TestCase):
    def test_generated_files_up_to_date(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_standard_cases.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_at_least_one_standard_case_per_track(self):
        idx = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
        standard = set(CASES) | {"IC-SPD-001"}
        for track in idx["tracks"]:
            ids = {c["id"] for c in track["cases"]}
            self.assertTrue(ids & standard, f"track {track['id']} has no standard case")

    def test_case_modules_complete(self):
        for c in CASES:
            h = ROOT / "cases" / c / "harness"
            self.assertTrue({"STAGES", "PIPELINE", "PARAMS", "HELD_OUT_KEYS", "make_dev"} <= names_in(h / "stages.py"), c)
            self.assertTrue({"score", "GATES", "PRIMARY", "OutputError"} <= names_in(h / "metrics.py"), c)
            oj = yaml.safe_load((ROOT / "cases" / c / "case" / "oj.yaml").read_text(encoding="utf-8"))
            self.assertTrue(oj.get("standard"))
            self.assertEqual(oj["judge"]["kind"], "harness")
            if oj["judge"].get("baseline"):
                self.assertTrue((ROOT / "cases" / c / oj["judge"]["baseline"]).is_dir(), c)
            mod = prepare_module(c)
            self.assertTrue(mod.SOURCES and callable(mod.build) and callable(mod.synthetic), c)

    def test_reference_submissions_pass_format_check(self):
        for c in CASES + ["IC-SPD-001"]:
            sub = ROOT / "cases" / c / "example_submission"
            with tempfile.TemporaryDirectory() as d:          # copy without caches created by other tests
                dst = Path(d) / "s"
                shutil.copytree(sub, dst, ignore=shutil.ignore_patterns("__pycache__"))
                if c != "IC-SPD-001" and submission_kind(c) == "code_plus_checkpoint":
                    self.assertFalse((sub / "checkpoint").exists(), f"{c}: the reference checkpoint is built by train.py, not shipped")
                    self.assertTrue((sub / "train.py").is_file(), c)
                    (dst / "checkpoint").mkdir()                    # layout check only; contents are tested end to end
                    (dst / "checkpoint" / "weights.bin").write_bytes(b"\0" * 16)
                self.assertEqual(validate_submission_dir(dst, c), [], c)

    def test_baseline_code_shown_to_checkpoint_cases(self):
        for c in CASES:
            if submission_kind(c) != "code_plus_checkpoint":
                continue
            oj = yaml.safe_load((ROOT / "cases" / c / "case" / "oj.yaml").read_text(encoding="utf-8"))
            vis = oj["agent_visible"]["interface"]
            self.assertIn(f"harness/baseline/{oj['submission']['entry']}", vis, c)
            self.assertIn("harness/baseline/train.py", vis, c)


class NumpyCasesEndToEnd(unittest.TestCase):
    """Synthetic data → the case's own dev split → the shared stage runner → the case's metrics."""

    def test_selfcheck_path(self):
        for case, n in NUMPY_CASES.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                d = Path(d)
                fz = prepare_module(case).synthetic(n)
                materialise(fz, d / "public")
                h = ROOT / "cases" / case / "harness"
                stages = abench_stage.load_module(h / "stages.py", f"st_{case.replace('-', '_')}")
                metrics = abench_stage.load_module(h / "metrics.py", f"mt_{case.replace('-', '_')}")
                dev = stages.make_dev(d / "public", d / "dev", quick=True)
                sub = copy_example(case, d / "sub")
                train_checkpoint(case, sub, dev["public"])
                res = abench_stage.run_pipeline(stages, h / "stages.py", sub, d / "run", public=dev["public"],
                                                models=None, held_out=dev["held_out"], seed=7, device="cpu",
                                                scale=dev.get("time_scale", 1.0))
                self.assertEqual(res["status"], "ok", res["stages"][-1].get("stderr_tail"))
                m = metrics.score(res["outputs"], dev["reference"], stages.PARAMS, res["seconds"])
                self.assertIn(metrics.PRIMARY["metric"], m)
                self.assertTrue(np.isfinite(m[metrics.PRIMARY["metric"]]))

    def test_judge_path(self):
        """oj_eval.evaluate_harness on synthetic frozen data with a throw-away key (paths redirected to a temp dir)."""
        import hidden_store
        import oj_eval
        from prepare.common import npz_bytes
        case = "TC-FT-001"
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fz = prepare_module(case).synthetic(1200)
            materialise(fz, d / "public")
            key = d / "key.hex"
            key.write_text(secrets.token_hex(32))
            plain = npz_bytes(**fz.hidden)
            (d / "hidden.enc").write_bytes(hidden_store.encrypt_bytes(hidden_store.read_key(key), plain))
            oj = oj_eval.load_oj(case)
            oj["data"]["public_dir"] = str(d / "public")
            oj["data"]["public"] = [{"path": str(d / "public" / "train.npz"),
                                     "sha256": oj_eval.sha256_bytes((d / "public" / "train.npz").read_bytes())}]
            oj["data"]["hidden"] = {"path": str(d / "hidden.enc"), "sha256_plain": oj_eval.sha256_bytes(plain)}
            oj["judge"]["device"] = "cpu"
            oj["judge"]["seeds"] = [1, 2]          # two runs of the same checkpoint: the protocol, not the training
            sub = copy_example(case, d / "sub")
            train_checkpoint(case, sub, d / "public")
            rec = oj_eval.evaluate_harness(case, sub, key, oj, {})
            self.assertEqual(rec["verdict"], "evaluated", rec)
            self.assertTrue(rec["metrics"]["valid"])
            self.assertEqual(rec["n_seeds"], 2)


if __name__ == "__main__":
    unittest.main()

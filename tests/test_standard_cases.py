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
FILLED_SPECS = sorted((ROOT / "spec" / "cases").glob("*.yaml"))     # built from the standard templates
STANDARD = [p.stem for p in SPECS]
CASES = STANDARD + [p.stem for p in FILLED_SPECS]
NUMPY_CASES = {"TC-FT-001": 1500, "DS-PIPE-002": 600, "RS-SEQ-001": 300, "GN-OGB-002": 1500,
               "PH-NO-001": 120, "U-LOOP-003": 200,
               # cases filled from the templates (spec/cases/)
               "DS-GBM-001": 400, "LG-LONG-001": 300, "PH-SYM-005": 4, "RS-GRAPH-002": 300, "PH-MLIP-003": 300,
               "X-ROBUST-009": 600, "FN-LTSF-005": 6}


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


MODEL_U_LOOP2 = '''
import numpy as np
import reason_tasks as T


def load(checkpoint_dir, device):
    return {}


def n_params(model):
    return 0.0


def init(model, task, x, budget_flops):
    return {"task": task, "x": x, "it": 0}


def step(model, state, active):
    state["it"] += 1
    return state


def readout(model, state):
    n = len(state["x"])
    logits = np.zeros((n, T.N_CLASSES[state["task"]]))
    logits[:, 0] = 1.0
    return logits, np.full(n, state["it"] >= 2)
'''

MODEL_BI_PROT = '''
import numpy as np


def load(checkpoint_dir, models_dir, device):
    return {}


def predict(model, assay):
    to = np.array([m[-1] for m in assay["few_shot_mutants"]])
    y = np.asarray(assay["few_shot_scores"])
    mean = {a: float(y[to == a].mean()) for a in set(to)}
    return np.array([mean.get(m[-1], float(y.mean())) for m in assay["mutants"]])
'''

PIPE_FN_QA = '''
import re
def load(checkpoint_dir, models_dir, device):
    return None
def answer(state, items):                      # stand-in: subtract the first two numbers of the second table row
    out = []
    for it in items:
        nums = [c for c in (it["table"][1][1:] if len(it["table"]) > 1 else []) if re.search(r"[0-9]", c)]
        out.append({"program": f"subtract({nums[0]}, {nums[1]})"} if len(nums) > 1 else {"answer": 0})
    return out
'''

PIPE_LG_RSN = '''
def load(checkpoint_dir, models_dir, device):
    return None
def predict(state, task, rows):                # stand-in: the label named in the text, else the first label
    out = []
    for r in rows:
        text = " ".join(str(v) for v in r.values()).lower()
        out.append(next((l for l in task["labels"] if " " + l.lower() + " " in " " + text + " "), task["labels"][0]))
    return out
'''

AGENT_X_PPO = '''
import numpy as np


class Linear:                      # random search over linear policies, a few hundred steps per task
    def __init__(self, w):
        self.w = w

    def act(self, obs):
        return np.clip(np.asarray(obs) @ self.w, -1, 1)


def train(make_env, spec, seed):
    rng = np.random.default_rng(seed)
    env = make_env(4)
    best, best_r = np.zeros((spec["obs_dim"], 1)), -np.inf
    for _ in range(3):
        w = rng.normal(size=(spec["obs_dim"], 1))
        obs, tot = env.reset(), 0.0
        for _ in range(50):
            obs, r, term, trunc = env.step(np.clip(obs @ w, -1, 1))
            tot += float(r.sum())
        if tot > best_r:
            best, best_r = w, tot
    return Linear(best)
'''


class StandardCaseFiles(unittest.TestCase):
    def test_generated_files_up_to_date(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_standard_cases.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_every_track_has_a_specified_case(self):
        """Every track keeps at least one case built to task statement v2.1 (a template under spec/standard_cases/
        or a case filled from the templates under spec/cases/). Since the 2026-10-23 review some tracks no longer
        have a template of their own."""
        idx = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
        specified = set(CASES) | {"IC-SPD-001"}
        for track in idx["tracks"]:
            ids = {c["id"] for c in track["cases"]}
            self.assertTrue(ids & specified, f"track {track['id']} has no specified case")

    def test_case_modules_complete(self):
        for c in CASES:
            h = ROOT / "cases" / c / "harness"
            self.assertTrue({"STAGES", "PIPELINE", "PARAMS", "HELD_OUT_KEYS", "make_dev"} <= names_in(h / "stages.py"), c)
            self.assertTrue({"score", "GATES", "PRIMARY", "OutputError"} <= names_in(h / "metrics.py"), c)
            oj = yaml.safe_load((ROOT / "cases" / c / "case" / "oj.yaml").read_text(encoding="utf-8"))
            self.assertEqual(bool(oj.get("standard")), c in STANDARD, c)
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
        only = set(filter(None, os.environ.get("ABENCH_TEST_CASES", "").split(",")))   # optional subset for quick runs
        for case, n in NUMPY_CASES.items():
            if only and case not in only:
                continue
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                d = Path(d)
                fz = prepare_module(case).synthetic(n)
                materialise(fz, d / "public")
                h = ROOT / "cases" / case / "harness"
                sys.path.insert(0, str(h))                   # case helper modules (textpack.py, ...), as in the child
                stages = abench_stage.load_module(h / "stages.py", f"st_{case.replace('-', '_')}")
                metrics = abench_stage.load_module(h / "metrics.py", f"mt_{case.replace('-', '_')}")
                dev = stages.make_dev(d / "public", d / "dev", quick=True)
                sub = copy_example(case, d / "sub")
                train_checkpoint(case, sub, dev["public"])
                res = abench_stage.run_pipeline(stages, h / "stages.py", sub, d / "run", public=dev["public"],
                                                models=None, held_out=dev["held_out"], seed=7, device="cpu",
                                                scale=dev.get("time_scale", 1.0), held_pre=dev.get("held_pre"))
                self.assertEqual(res["status"], "ok", res["stages"][-1].get("stderr_tail"))
                m = metrics.score(res["outputs"], dev["reference"], stages.PARAMS, res["seconds"])
                self.assertIn(metrics.PRIMARY["metric"], m)
                self.assertTrue(np.isfinite(m[metrics.PRIMARY["metric"]]))
                # every checkpoint case declares how its held-out examples are resampled for the 95% interval
                if submission_kind(case) == "code_plus_checkpoint":
                    self.assertTrue(getattr(metrics, "RESAMPLE", None), case)
                if not getattr(metrics, "RESAMPLE", None):
                    continue
                import abench_metrics
                ci = abench_metrics.bootstrap_ci(metrics.score, res["outputs"], dev["reference"], stages.PARAMS,
                                                 res["seconds"], metrics.RESAMPLE, metrics.PRIMARY["metric"], n=60)
                self.assertIsNotNone(ci, case)
                v = m[metrics.PRIMARY["metric"]]
                self.assertLessEqual(ci["low"], ci["high"])
                self.assertTrue(ci["low"] - 1e-9 <= v <= ci["high"] + 1e-9 or abs(ci["high"] - ci["low"]) < 1e-9
                                or ci["low"] - 0.1 <= v <= ci["high"] + 0.1, (case, v, ci))

    def test_selfcheck_path_numpy_agent(self):
        """Cases whose reference submission needs PyTorch: the pipeline runs with a small NumPy stand-in agent."""
        agents = {"X-PPO-001": ("agent.py", AGENT_X_PPO), "U-LOOP-002": ("model.py", MODEL_U_LOOP2),
                  "BI-PROT-001": ("model.py", MODEL_BI_PROT), "FN-QA-004": ("pipeline.py", PIPE_FN_QA),
                  "LG-RSN-003": ("pipeline.py", PIPE_LG_RSN)}
        only = set(filter(None, os.environ.get("ABENCH_TEST_CASES", "").split(",")))
        for case, (entry, code) in agents.items():
            if only and case not in only:
                continue
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                d = Path(d)
                h = ROOT / "cases" / case / "harness"
                sys.path.insert(0, str(h))
                stages = abench_stage.load_module(h / "stages.py", f"st_{case.replace('-', '_')}")
                metrics = abench_stage.load_module(h / "metrics.py", f"mt_{case.replace('-', '_')}")
                public = None
                if case in ("BI-PROT-001", "FN-QA-004", "LG-RSN-003"):                 # cases with public data: synthetic stand-in
                    public = d / "public"
                    materialise(prepare_module(case).synthetic(100), public)
                dev = stages.make_dev(public, d / "dev", quick=True)
                sub = d / "sub"
                (sub / "checkpoint").mkdir(parents=True)
                (sub / entry).write_text(code)
                res = abench_stage.run_pipeline(stages, h / "stages.py", sub, d / "run", public=dev["public"],
                                                models=None, held_out=dev["held_out"], seed=7, device="cpu",
                                                scale=dev.get("time_scale", 1.0), held_pre=dev.get("held_pre"))
                self.assertEqual(res["status"], "ok", res["stages"][-1].get("stderr_tail"))
                m = metrics.score(res["outputs"], dev["reference"], stages.PARAMS, res["seconds"])
                self.assertTrue(np.isfinite(m[metrics.PRIMARY["metric"]]), m)
                if case == "BI-PROT-001":
                    self.assertGreater(m["mean_spearman"], 0.3, m)
                if case == "LG-RSN-003":
                    self.assertGreater(m["mean_balanced_accuracy"], 0.95, m)
                if case == "FN-QA-004":                  # half the synthetic questions are one-step revenue changes
                    self.assertTrue(0.3 < m["execution_accuracy"] < 0.7, m)
                if case == "X-PPO-001":
                    self.assertLessEqual(m["max_samples_ratio"], 0.1 + 1e-9)

    def judge(self, case: str, n: int) -> dict:
        """oj_eval.evaluate_harness on synthetic frozen data with a throw-away key (paths redirected to a temp dir)."""
        import hidden_store
        import oj_eval
        from prepare.common import npz_bytes
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fz = prepare_module(case).synthetic(n)
            materialise(fz, d / "public")
            key = d / "key.hex"
            key.write_text(secrets.token_hex(32))
            plain = npz_bytes(**fz.hidden)
            (d / "hidden.enc").write_bytes(hidden_store.encrypt_bytes(hidden_store.read_key(key), plain))
            oj = oj_eval.load_oj(case)
            oj["data"]["public_dir"] = str(d / "public")
            oj["data"]["public"] = [{"path": str(p), "sha256": oj_eval.sha256_bytes(p.read_bytes())}
                                    for p in sorted((d / "public").iterdir())]
            oj["data"]["hidden"] = {"path": str(d / "hidden.enc"), "sha256_plain": oj_eval.sha256_bytes(plain)}
            oj["judge"]["device"] = "cpu"
            oj["judge"]["seeds"] = [1, 2]
            sub = copy_example(case, d / "sub")
            train_checkpoint(case, sub, d / "public")
            return oj_eval.evaluate_harness(case, sub, key, oj, {})

    def test_judge_path(self):
        """A checkpoint case (two runs of the same checkpoint: the protocol, not the training)."""
        rec = self.judge("TC-FT-001", 1200)
        self.assertEqual(rec["verdict"], "evaluated", rec)
        self.assertTrue(rec["metrics"]["valid"])
        self.assertEqual(rec["n_seeds"], 2)
        self.assertIsNotNone(rec.get("primary_ci95"))

    def test_judge_path_trains(self):
        """A judge_trains case with pre-timed held-out inputs and a baseline trained alongside (DS-GBM-001)."""
        rec = self.judge("DS-GBM-001", 400)
        self.assertEqual(rec["verdict"], "evaluated", rec)
        self.assertTrue(rec["metrics"]["valid"], rec["metrics"])
        self.assertEqual(rec["metrics"]["n_datasets"], 20)
        self.assertIn("mean_score_delta_vs_hgb", rec["metrics"])
        self.assertIsNotNone(rec.get("primary_ci95"))


class KernelCaseMetrics(unittest.TestCase):
    """Metric arithmetic of the measurement-only kernel cases on synthetic stage outputs (no GPU needed)."""

    def _metrics(self, case):
        import abench_stage
        h = ROOT / "cases" / case / "harness"
        sys.path.insert(0, str(h))
        try:
            return abench_stage.load_module(h / "metrics.py", f"mtk_{case.replace('-', '_')}")
        finally:
            sys.path.remove(str(h))

    def test_kernelbench_metrics(self):
        for case in ("K-KB-001", "K-FUS-002"):
            m = self._metrics(case)
            ref = np.array([[2.0, 1.0, 10.0], [4.0, np.nan, 10.0], [1.0, 1.0, 10.0]])        # eager, compile, memory
            rows = np.array([[1, 0.5, 1, 0.5, 5.0], [1, 0.2, 1, 8.0, 20.0], [0, np.inf, 0, np.inf, np.nan]])
            out = {"reference": {"rows": ref}, "bench": {"rows": rows, "forbidden_api": np.array([0]),
                                                         "tampered": np.array([0])}}
            r = m.score(out, {}, {}, {})
            # ratios vs the stronger baseline: 1/0.5 = 2, 4/8 = 0.5 (compile failed: eager), failed → 0.1
            self.assertAlmostEqual(r["speedup_geomean"], (2 * 0.5 * 0.1) ** (1 / 3), places=9)
            self.assertAlmostEqual(r["fast_1"], 1 / 3)
            self.assertAlmostEqual(r["fraction_correct"], 2 / 3)
            self.assertEqual(r["compile_failed"], 1.0)

    def test_kernelbench_overrides_cover_selected_files(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from prepare import kernelbench
        self.assertEqual(len(kernelbench._problems("level1")), 30)
        self.assertEqual(len(kernelbench._problems("level2")), 20)
        for case, lv in (("K-KB-001", "level1"), ("K-FUS-002", "level2")):
            names = {n for _, n, _ in kernelbench._problems(lv)}
            files = {p.stem for p in (ROOT / "cases" / case / "example_submission" / "kernels").glob("*.py")}
            self.assertEqual(names, files, case)


class LayerCaseMetrics(unittest.TestCase):
    """U-KAN-009: error-ratio arithmetic, gates and the bootstrap groups on synthetic stage outputs (no GPU needed)."""

    def test_error_ratio(self):
        h = ROOT / "cases" / "U-KAN-009" / "harness"
        sys.path.insert(0, str(h))
        try:
            m = abench_stage.load_module(h / "metrics.py", "mt_ukan")
        finally:
            sys.path.remove(str(h))
        rng = np.random.default_rng(0)
        ref, fo = {}, {}
        for n in ("f00", "f01"):
            y = rng.normal(size=500)
            ref[f"{n}_y_test"] = y
            fo[f"{n}_pred_sub"] = y + 0.05 * rng.normal(size=500)            # about half the MLP's error
            fo[f"{n}_pred_mlp"] = y + 0.10 * rng.normal(size=500)
            for who, ms in (("sub", 2.0), ("mlp", 1.0)):
                fo.update({f"{n}_{who}_width": np.array([30]), f"{n}_{who}_params": np.array([2990]),
                           f"{n}_{who}_lr": np.array([3e-3]), f"{n}_{who}_step_ms": np.array([ms]),
                           f"{n}_{who}_diverged": np.array([0.0])})
        y = rng.integers(0, 10, 1000)
        ref["img_y_test"] = y
        sub, mlp = y.copy(), y.copy()
        sub[:100] = (sub[:100] + 1) % 10                                    # 10% errors
        mlp[:100] = (mlp[:100] + 1) % 10
        io = {"img_pred_sub": sub, "img_pred_mlp": mlp}
        for who in ("sub", "mlp"):
            io.update({f"img_{who}_width": np.array([120]), f"img_{who}_params": np.array([99000]),
                       f"img_{who}_lr": np.array([1e-3]), f"img_{who}_step_ms": np.array([1.0]),
                       f"img_{who}_diverged": np.array([0.0])})
        out = {"functions": fo, "image": io}
        r = m.score(out, ref, {}, {})
        f = np.exp(np.mean([np.log(r["f00_ratio"]), np.log(r["f01_ratio"])]))
        self.assertAlmostEqual(r["function_error_ratio"], f)
        self.assertAlmostEqual(r["image_error_ratio"], 1.0)
        self.assertAlmostEqual(r["error_ratio"], np.sqrt(f), places=12)
        self.assertTrue(0.4 < f < 0.6, f)
        self.assertEqual((r["max_step_time_ratio"], r["budget_violations"]), (2.0, 0.0))
        fo["f01_sub_params"] = np.array([3001])
        self.assertEqual(m.score(out, ref, {}, {})["budget_violations"], 1.0)
        self.assertEqual(len(m.RESAMPLE(ref)), 3)


if __name__ == "__main__":
    unittest.main()

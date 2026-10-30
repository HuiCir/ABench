"""IC-SPD-001 timing-protocol regression tests: call speedrun_runner.run_once directly, without modifying repository files."""
from __future__ import annotations

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("speedrun_runner", ROOT / "cases/IC-SPD-001/harness/speedrun_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

HONEST = textwrap.dedent("""
    import numpy as np
    def train(data, seed, device):
        x = data["x"].reshape(len(data["x"]), -1).astype(np.float32)
        return np.stack([x[data["y"] == c].mean(0) for c in range(10)])
    def predict(m, x, device):
        x = x.reshape(len(x), -1).astype(np.float32)
        return -((x[:, None, :] - m[None]) ** 2).sum(-1)
""")


class SpeedrunTiming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        y = np.arange(200) % 10
        x = np.clip(rng.normal(25 * y[:, None, None, None], 5, (200, 32, 32, 3)), 0, 255).astype(np.uint8)
        np.savez(d / "train.npz", x=x, y=y.astype(np.int64))
        np.save(d / "x.npy", x[:50])
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def run_sub(self, code: str, seed: int = 1, run: str = "r"):
        sub = self.d / f"sub_{run}"
        sub.mkdir(exist_ok=True)
        (sub / "method.py").write_text(code, encoding="utf-8")
        return runner.run_once(sub, self.d / "train.npz", self.d / "x.npy", seed, self.d / run, device="cpu",
                               timeout=60, predict_cap=2.0, tail_slack=1.0)

    def test_migrated_metric_preserves_cross_seed_median_and_max(self):
        metric_spec = importlib.util.spec_from_file_location("speedrun_metrics", ROOT / "cases/IC-SPD-001/harness/metrics.py")
        metric = importlib.util.module_from_spec(metric_spec)
        metric_spec.loader.exec_module(metric)
        rows = [{"train_seconds_mean": t, "train_seconds_median": t, "train_seconds_max": t,
                 "accuracy_cifar10_mean": 0.95, "accuracy_cifar10_1_mean": 0.81} for t in (1.,2.,100.)]
        result = metric.aggregate(rows)
        self.assertEqual(result["train_seconds_median"], 2.)
        self.assertEqual(result["train_seconds_max"], 100.)
        self.assertEqual(result["train_seconds_mean"], 34.333333)
        self.assertEqual(metric.GATES, [{"metric":"accuracy_cifar10_mean", "min":0.94},
                                        {"metric":"accuracy_cifar10_1_mean", "min":0.80}])

    def test_migrated_gate_preserves_legacy_threshold_field(self):
        from tools.oj_eval import _gates
        gate={"metric":"accuracy_cifar10_mean","min":0.94}
        for value,passed in ((0.94,True),(0.939,False)):
            result=_gates([gate],{"accuracy_cifar10_mean":value})[gate["metric"]]
            self.assertEqual(result['threshold'],0.94)
            self.assertEqual(result['min'],0.94)
            self.assertIsNone(result['max'])
            self.assertEqual(result['pass'],passed)

    def test_test_images_are_given_before_training_and_predict_is_capped_from_go(self):
        """As with --x-test, the unlabeled test images exist from the start; P comes within predict_cap of G."""
        code = HONEST.replace("def train(data, seed, device):",
                              "def train(data, seed, device):\n    import os\n"
                              "    assert os.path.isfile(os.path.join('pre', 'held_pre.npz'))")
        r = self.run_sub(code)
        self.assertEqual(r["status"], "ok", r.get("stderr_tail"))
        self.assertEqual(np.load(r["pred_path"]).shape, (50, 10))
        self.assertLess(r["predict_seconds"], 2.0)
        self.assertNotIn("post_inputs", {k for k, v in r["phase_seconds"].items() if v > 0.5})

    def test_early_handshake_is_violation(self):
        code = HONEST.replace("def train(data, seed, device):",
                              "def train(data, seed, device):\n    import os, sys\n"
                              "    os.write(int(sys.argv[sys.argv.index('--fd') + 1]), b'T')")
        self.assertEqual(self.run_sub(code)["status"], "protocol_violation")

    def test_background_work_after_train_is_caught(self):
        code = HONEST.replace("    return np.stack", "    import threading, time\n"
                              "    threading.Thread(target=lambda: time.sleep(6)).start()\n    return np.stack")
        self.assertEqual(self.run_sub(code)["status"], "predict_time_exceeded")

    def test_import_time_is_counted(self):
        r = self.run_sub("import time\ntime.sleep(1.0)\n" + HONEST)
        self.assertEqual(r["status"], "ok")
        self.assertGreaterEqual(r["train_seconds"], 1.0)

    def test_caches_are_fresh_per_run(self):
        code = HONEST.replace("def train(data, seed, device):", textwrap.dedent("""\
            def train(data, seed, device):
                import os
                for k in ("HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                    m = os.path.join(os.environ[k], "marker")
                    assert not os.path.exists(m), "cache reused"
                    open(m, "w").close()"""))
        self.assertEqual(self.run_sub(code, run="a")["status"], "ok")
        self.assertEqual(self.run_sub(code, run="b")["status"], "ok")


if __name__ == "__main__":
    unittest.main()

"""Regression tests of the shared stage protocol (tools/harness_kit/abench_stage.py), using a throw-away case.

They check that the timed phase cannot see the held-out inputs, that protocol bytes written by a submission or work
left running after the stage are caught, that the finish phase is capped, and that stages only receive declared files.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "tools" / "harness_kit"
sys.path.insert(0, str(KIT))
import abench_stage  # noqa: E402

STAGES = textwrap.dedent('''
    import numpy as np
    from abench_stage import Stage
    PARAMS = {}
    HELD_OUT_KEYS = ["x"]
    PIPELINE = [{"stage": "a", "timeout": 30, "finish_cap": 3, "tail_slack": 2, "threads": 1, "held_out": False},
                {"stage": "b", "timeout": 30, "finish_cap": 3, "tail_slack": 2, "threads": 1, "held_out": True,
                 "carry": {"a": ["keep.txt"]}}]
    def a_timed(ctx, s):
        mod = ctx.import_submission("sub.py")
        return mod.train(ctx)
    def a_finish(ctx, s):
        return {"seen_post": np.asarray([int(bool(s))])}
    def b_timed(ctx, s):
        return sorted(p.name for p in ctx.work.iterdir() if not p.name.startswith("_"))
    def b_finish(ctx, listing):
        mod = ctx.import_submission("sub.py")
        with np.load(ctx.post_dir / "held_out.npz") as z:
            return {"y": mod.predict(z["x"]), "n_files": np.asarray([len(listing)])}
    STAGES = {"a": Stage(timed=a_timed, finish=a_finish), "b": Stage(timed=b_timed, finish=b_finish)}
''')

HONEST = textwrap.dedent('''
    import numpy as np
    def train(ctx):
        (ctx.work / "keep.txt").write_text("ok")
        (ctx.work / "drop.txt").write_text("not carried")
        return (ctx.work / "post").exists()        # held-out inputs must not exist during the timed phase
    def predict(x):
        return x * 2
''')


class StageProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        (self.d / "case").mkdir()
        (self.d / "case" / "stages.py").write_text(STAGES)
        np.savez(self.d / "held.npz", x=np.arange(5))

    def tearDown(self):
        self.tmp.cleanup()

    def run_sub(self, code: str):
        sub = self.d / "sub"
        sub.mkdir(exist_ok=True)
        (sub / "sub.py").write_text(code)
        stages = abench_stage.load_module(self.d / "case" / "stages.py", "t_stages_" + str(abs(hash(code))))
        return abench_stage.run_pipeline(stages, self.d / "case" / "stages.py", sub, self.d / "work", public=None,
                                         models=None, held_out=self.d / "held.npz", seed=1, device="cpu")

    def test_honest_pipeline(self):
        r = self.run_sub(HONEST)
        self.assertEqual(r["status"], "ok", r["stages"][-1].get("stderr_tail"))
        self.assertEqual(int(r["outputs"]["a"]["seen_post"][0]), 0)          # nothing held out during training
        self.assertEqual(r["outputs"]["b"]["y"].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(int(r["outputs"]["b"]["n_files"][0]), 1)            # only the declared keep.txt was carried
        self.assertFalse((self.d / "work" / "b" / "post").exists())         # held-out copy removed after the run

    def test_forged_handshake_is_violation(self):
        code = HONEST.replace("def train(ctx):", "def train(ctx):\n    import os, sys\n"
                              "    os.write(int(sys.argv[sys.argv.index('--fd') + 1]), b'T')")
        r = self.run_sub(code)
        self.assertEqual(r["status"], "protocol_violation")

    def test_slow_finish_is_capped(self):
        code = HONEST.replace("    return x * 2", "    import time; time.sleep(10)\n    return x * 2")
        r = self.run_sub(code)
        self.assertEqual(r["status"], "finish_time_exceeded")

    def test_missing_artifact(self):
        r = self.run_sub(HONEST.replace('(ctx.work / "keep.txt").write_text("ok")', "pass"))
        self.assertEqual(r["status"], "missing_artifact")


if __name__ == "__main__":
    unittest.main()

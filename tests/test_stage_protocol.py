"""Regression tests of the shared stage protocol (tools/harness_kit/abench_stage.py), using a throw-away case.

They check that the timed phase cannot see the held-out inputs, that protocol bytes written by a submission or work
left running after the stage are caught, that the finish phase is capped, and that stages only receive declared files.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
import zipfile
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
        import os
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            assert os.environ[name] == '1', name
        assert 'OPENBLAS_NUM_THREADS' not in os.environ and 'NUMEXPR_NUM_THREADS' not in os.environ
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

    def run_sub(self, code: str, stages_source: str = STAGES):
        sub = self.d / "sub"
        sub.mkdir(exist_ok=True)
        (sub / "sub.py").write_text(code)
        (self.d / "case" / "stages.py").write_text(stages_source)
        stages = abench_stage.load_module(self.d / "case" / "stages.py",
                                          "t_stages_" + str(abs(hash((stages_source, code)))))
        return abench_stage.run_pipeline(stages, self.d / "case" / "stages.py", sub, self.d / "work", public=None,
                                         models=None, held_out=self.d / "held.npz", seed=1, device="cpu")

    def test_honest_pipeline(self):
        r = self.run_sub(HONEST)
        self.assertEqual(r["status"], "ok", r["stages"][-1].get("stderr_tail"))
        self.assertEqual(int(r["outputs"]["a"]["seen_post"][0]), 0)          # nothing held out during training
        self.assertEqual(r["outputs"]["b"]["y"].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(int(r["outputs"]["b"]["n_files"][0]), 1)            # only the declared keep.txt was carried
        self.assertFalse((self.d / "work" / "b" / "post").exists())         # held-out copy removed after the run

    def test_stdout_is_log_and_child_timing_is_recorded(self):
        code = HONEST.replace("def train(ctx):", "def train(ctx):\n    print('RTPT this is ordinary output', flush=True)\n    import time; time.sleep(0.04)")
        r = self.run_sub(code)
        self.assertEqual(r["status"], "ok", r)
        stage = r["stages"][0]
        self.assertIn("ordinary output", stage["stdout_tail"])
        self.assertGreaterEqual(stage["child_timing"]["timed_seconds"], 0.04)
        self.assertLess(abs(stage["timed_seconds"] - stage["child_timing"]["timed_seconds"]), 0.02)
        self.assertEqual(set(stage["phase_seconds"]), {"start", "post_inputs", "run", "collect"})

    def test_forged_handshake_is_violation(self):
        code = HONEST.replace("def train(ctx):", "def train(ctx):\n    import os, sys\n"
                              "    os.write(int(sys.argv[sys.argv.index('--fd') + 1]), b'T')")
        r = self.run_sub(code)
        self.assertEqual(r["status"], "protocol_violation")

    def test_forged_complete_handshake_cannot_skip_go(self):
        code = HONEST.replace("def train(ctx):", "def train(ctx):\n    import os, sys\n"
                              "    (ctx.work / 'keep.txt').write_text('ok')\n"
                              "    np.savez(ctx.work / (ctx.stage + '.out.npz'), seen_post=np.asarray([0]))\n"
                              "    os.write(int(sys.argv[sys.argv.index('--fd') + 1]), b'TP')\n"
                              "    os._exit(0)")
        self.assertEqual(self.run_sub(code)["status"], "protocol_violation")

    def test_slow_finish_is_capped(self):
        code = HONEST.replace("    return x * 2", "    import time; time.sleep(10)\n    return x * 2")
        r = self.run_sub(code)
        self.assertEqual(r["status"], "finish_time_exceeded")

    def test_writing_outputs_after_p_counts_against_finish_cap_unless_the_stage_allows_slack(self):
        """P and the outputs within finish_cap (3 s) of G; with slack_after_p the outputs may use tail_slack (2 s) more."""
        slow = ("    class Slow:\n"
                "        def detach(self):\n"
                "            import time; time.sleep({})\n"
                "            return self\n"
                "        def cpu(self):\n"
                "            return self\n"
                "        def numpy(self):\n"
                "            return x * 2\n"
                "    return Slow()")
        self.assertEqual(self.run_sub(HONEST.replace("    return x * 2", slow.format(3.5)))["status"],
                         "finish_time_exceeded")
        slack = STAGES.replace('"held_out": True,', '"held_out": True, "slack_after_p": True,')
        r = self.run_sub(HONEST.replace("    return x * 2", slow.format(3.5)), slack)
        self.assertEqual(r["status"], "ok", r["stages"][-1])
        self.assertEqual(r["outputs"]["b"]["y"].tolist(), [0, 2, 4, 6, 8])
        r = self.run_sub(HONEST.replace("    return x * 2", slow.format(6)), slack)
        self.assertEqual(r["status"], "finish_time_exceeded")

    def test_only_the_tail_of_a_large_log_is_read(self):
        log = self.d / "_stdout.txt"
        log.write_bytes(b"x" * 5_000_000 + b"last line")
        self.assertEqual(abench_stage._read_regular(log, tail=9), b"last line")
        self.assertEqual(len(abench_stage._read_regular(log)), 5_000_009)

    def test_missing_artifact(self):
        r = self.run_sub(HONEST.replace('(ctx.work / "keep.txt").write_text("ok")', "pass"))
        self.assertEqual(r["status"], "missing_artifact")


class NpzOutputLimits(unittest.TestCase):
    def test_compressed_output_is_bounded_before_expansion(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "output.npz"
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('large.npy', b'x' * 32768)
            self.assertLess(path.stat().st_size, 1024)
            with self.assertRaisesRegex(ValueError, 'uncompressed byte limit'):
                abench_stage._check_npz(path, max_bytes=1024)

    def test_member_count_and_valid_numpy_archive(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "output.npz"
            np.savez(path, a=np.arange(5), b=np.arange(3))
            abench_stage._check_npz(path, max_bytes=1024, max_members=2)
            with self.assertRaisesRegex(ValueError, 'member limit'):
                abench_stage._check_npz(path, max_members=1)

    def test_npy_header_cannot_declare_more_than_its_payload(self):
        # an 8 TiB float64 array declared in a few hundred bytes: rejected before NumPy allocates it
        header = {'descr': '<f8', 'fortran_order': False, 'shape': (1024, 1024, 1024, 1024)}
        for name in ('huge.npy', 'no_suffix'):                 # NumPy parses any member with the NPY magic
            with tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "output.npz"
                with zipfile.ZipFile(path, 'w') as archive:
                    with archive.open(name, 'w') as member:
                        np.lib.format.write_array_header_1_0(member, header)
                        member.write(b'\0' * 64)
                with self.assertRaisesRegex(ValueError, 'declares more data'):
                    abench_stage._check_npz(path)

    def test_header_bound_accepts_real_arrays_and_rejects_objects(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "output.npz"
            np.savez_compressed(path, a=np.zeros((7, 3), dtype=np.float32), b=np.array(['x', 'yz']),
                                c=np.zeros((0, 5)), d=np.asfortranarray(np.ones((4, 2))))
            abench_stage._check_npz(path)
            with zipfile.ZipFile(path, 'a') as archive:
                with archive.open('e.npy', 'w') as member:
                    np.lib.format.write_array_header_1_0(member, {'descr': '|O', 'fortran_order': False, 'shape': (1,)})
            with self.assertRaisesRegex(ValueError, 'dtype object'):
                abench_stage._check_npz(path)


if __name__ == "__main__":
    unittest.main()

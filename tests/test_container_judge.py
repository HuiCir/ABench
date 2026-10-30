"""Opt-in real Docker stage tests: ABENCH_TEST_IMAGE=<cpu-sci reference> unittest discover ..."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools/harness_kit'))
import abench_stage
from judge_container import DockerStageLauncher


@unittest.skipUnless(os.environ.get('ABENCH_TEST_IMAGE') and shutil.which('docker'),
                     'set ABENCH_TEST_IMAGE to a locally available CPU profile image')
class DockerJudgeTest(unittest.TestCase):
    def test_stdio_delayed_inputs_readonly_submission_and_host_equivalence(self):
        source = textwrap.dedent('''
            import numpy as np
            from abench_stage import Stage
            HELD_OUT_KEYS = ['x']
            PIPELINE = [{'stage':'predict','timeout':30,'finish_cap':10,'threads':1,'held_out':True}]
            def timed(ctx, state):
                assert not ctx.post_dir.exists(), 'held-out inputs leaked before T'
                print('stage stdout stays in the log', flush=True)
                return None
            def finish(ctx, state):
                with np.load(ctx.post_dir / 'held_out.npz', allow_pickle=False) as z:
                    return {'prediction': z['x'] * 2}
            STAGES = {'predict':Stage(timed=timed,finish=finish)}
        ''')
        resources = dict(gpus=0, cpus=1, memory='1g', shm_size='64m', pids=128,
                         budget_tier='budget')
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / 'harness').mkdir()
            stages_path = root / 'harness/stages.py'
            stages_path.write_text(source)
            # Docker receives the exact same protocol code as the local selfcheck.
            shutil.copyfile(ROOT / 'tools/harness_kit/abench_stage.py', root / 'harness/abench_stage.py')
            sub = root / 'submission'
            sub.mkdir()
            (sub / 'code.py').write_text('pass\n')
            np.savez(root / 'held.npz', x=np.arange(6))
            module = abench_stage.load_module(stages_path, 'docker_test_stages')
            common = dict(public=None, models=None, held_out=root / 'held.npz', seed=1, device='cpu')
            local = abench_stage.run_pipeline(module, stages_path, sub, root / 'local', **common)
            launcher = DockerStageLauncher(os.environ['ABENCH_TEST_IMAGE'], resources)
            container = abench_stage.run_pipeline(module, stages_path, sub, root / 'docker', launcher=launcher, **common)
            launcher.close_inputs()
            self.assertEqual(container['status'], 'ok', container)
            self.assertEqual(len(launcher.input_record['cleanup']['removed']['volumes']), 1)
            self.assertEqual(launcher.input_record['cleanup']['remaining'], {'containers': [], 'volumes': []})
            np.testing.assert_array_equal(local['outputs']['predict']['prediction'], container['outputs']['predict']['prediction'])
            rec = container['stages'][0]
            self.assertTrue(rec['sandbox']['inspect']['verified'])
            # infrastructure wall times of every Docker step, recorded beside (never inside) the timed metrics
            self.assertEqual(set(rec['sandbox']['seconds']), {'inputs', 'data_caches', 'create', 'upload', 'verify',
                                                              'attach', 'post', 'download', 'remove'})
            self.assertEqual(set(rec['phase_seconds']), {'start', 'post_inputs', 'run', 'collect'})
            self.assertIn('stage stdout', rec['stdout_tail'])
            self.assertIn('child_timing', rec)
            self.assertFalse((root / 'docker/predict/post').exists())
            self.assertNotEqual(local['stages'][0].get('sandbox'), rec['sandbox'])

"""Process-group shutdown and regular-file handling, shared by local smoke and Docker orchestration."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/harness_kit"))
import abench_stage
import sandbox_runner


class AgentViewValidation(unittest.TestCase):
    def test_static_check_scans_interfaces_and_never_exempts_private_terms(self):
        import validate_task_statement as validator

        cid = 'LG-RSN-003'
        source = ROOT / 'cases' / cid
        oj = yaml.safe_load((source / 'case/oj.yaml').read_text())
        oj['agent_visible']['interface'] = ['harness/public.py']
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            case = root / 'cases' / cid
            (case / 'case/agent').mkdir(parents=True)
            (case / 'harness').mkdir()
            shutil.copyfile(source / 'case/agent/task.md', case / 'case/agent/task.md')
            interface = case / 'harness/public.py'
            interface.write_text('"""Public label scoring."""\n')
            (case / 'case/oj.yaml').write_text(yaml.safe_dump(oj))
            with patch.object(validator, 'ROOT', root):
                self.assertEqual(validator.check_case(cid)['issues'], [])
                # A visible interface must be checked even when the task statement is clean.
                interface.write_text('"""Private reference recipe."""\n')
                oj['agent_visible']['leak_allow'].append('reference recipe')
                (case / 'case/oj.yaml').write_text(yaml.safe_dump(oj))
                result = validator.check_case(cid)
                self.assertEqual(result['state'], 'draft')
                self.assertTrue(any('harness/public.py' in issue and 'reference recipe' in issue
                                    for issue in result['issues']))


class AgentProcess(unittest.TestCase):
    def test_timeout_kills_the_whole_process_group_and_streams_logs(self):
        marker = "abench-test-" + uuid.uuid4().hex[:10]
        d = Path(tempfile.mkdtemp())
        try:
            # a background child that would outlive its parent if only the parent were killed
            script = ("import subprocess, sys, time\n"
                      f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)', '{marker}'])\n"
                      "print('started', flush=True)\n"
                      "time.sleep(300)\n")
            t = time.time()
            status, code = sandbox_runner.run_agent([sys.executable, "-c", script, marker], d, None, b"prompt",
                                                    timeout=2, out_path=d / "o.log", err_path=d / "e.log", grace=1)
            self.assertEqual(status, "timeout")
            self.assertIsNone(code)
            self.assertLess(time.time() - t, 30)
            self.assertIn(b"started", (d / "o.log").read_bytes())
            time.sleep(0.3)
            left = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()
            self.assertEqual(left, [], "a process of the agent's group survived the deadline")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_prompt_reaches_stdin(self):
        d = Path(tempfile.mkdtemp())
        try:
            status, code = sandbox_runner.run_agent(
                [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read()[::-1])"], d, None, b"abc" * 50000,
                timeout=30, out_path=d / "o.log", err_path=d / "e.log")
            self.assertEqual((status, code), ("succeeded", 0))
            self.assertEqual((d / "o.log").read_bytes(), b"cba" * 50000)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class ChildFiles(unittest.TestCase):
    def test_read_regular_refuses_symlinks(self):
        d = Path(tempfile.mkdtemp())
        try:
            (d / "secret").write_bytes(b"s")
            os.symlink(d / "secret", d / "link")
            self.assertIsNone(abench_stage._read_regular(d / "link"))
            self.assertEqual(abench_stage._read_regular(d / "secret"), b"s")
            self.assertIsNone(abench_stage._read_regular(d / "missing"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_fresh_dir_replaces_a_planted_symlink(self):
        d = Path(tempfile.mkdtemp())
        try:
            (d / "elsewhere").mkdir()
            os.symlink(d / "elsewhere", d / "post")
            abench_stage._fresh_dir(d / "post")
            self.assertFalse((d / "post").is_symlink())
            self.assertTrue((d / "post").is_dir())
        finally:
            shutil.rmtree(d, ignore_errors=True)

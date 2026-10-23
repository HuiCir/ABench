"""Isolation of the agent stage (tools/sandbox_runner.py) and of the judge's children (tools/oj_eval.py,
tools/harness_kit/abench_stage.py).

The tests marked root-only create nothing: they need the users abench-agent and abench-judge
(useradd --system --no-create-home NAME) and a root shell, and are skipped otherwise.
"""
from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "harness_kit"))

import abench_stage  # noqa: E402
import sandbox_runner  # noqa: E402
from wrapper_policy import inspect_wrapper  # noqa: E402


def _user(name: str) -> bool:
    try:
        pwd.getpwnam(name)
        return True
    except KeyError:
        return False


ROOT_ONLY = os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0 \
    and _user("abench-agent") and _user("abench-judge") and shutil.which("setpriv") and shutil.which("unshare")


class WrapperPolicyNetns(unittest.TestCase):
    def test_exact_netns_wrapper_accepted(self):
        for w in ("unshare --net -- sudo -u abench-agent --", "unshare -n -- /usr/bin/sudo -u abench-agent --"):
            a = inspect_wrapper(w)
            self.assertTrue(a["accepted"] and a["scoreable"], (w, a))
            self.assertEqual(a["kind"], "user_netns")
            self.assertTrue(a["network"].startswith("none"))

    def test_variants_rejected(self):
        for w in ("unshare --net --mount -- sudo -u abench-agent --", "unshare --net -- sudo -u root --",
                  "unshare --net -- sudo -u abench-agent -- bash -c", "unshare --net -- /tmp/sudo -u abench-agent --",
                  "unshare --net -- sudo -E -u abench-agent --", "unshare -- sudo -u abench-agent --"):
            self.assertFalse(inspect_wrapper(w)["accepted"], w)

    def test_sudo_wrapper_records_host_network(self):
        self.assertEqual(inspect_wrapper("sudo -u abench-agent --")["network"], "host")


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


STAGES = '''
import os, socket
import numpy as np
from abench_stage import Stage

PIPELINE = [{"stage": "probe", "timeout": 60, "finish_cap": 30, "held_out": True}]
HELD_OUT_KEYS = ["x"]


def _readable(p):
    try:
        if os.path.isdir(p):
            os.listdir(p)
        else:
            open(p, "rb").read(1)
        return 1
    except OSError:
        return 0


def timed(ctx, state):
    probes = [p for p in os.environ.get("PROBE_PATHS", "").split(os.pathsep) if p]
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        net = 1
    except OSError:
        net = 0
    planted = 0
    try:                                   # try to redirect the parent's copy of the held-out inputs
        os.symlink("/root", str(ctx.work / "post"))
        planted = 1
    except OSError:
        pass
    return {"uid": os.getuid(), "net": net, "probes": [_readable(p) for p in probes], "planted": planted}


def finish(ctx, state):
    with np.load(ctx.post_dir / "held_out.npz") as z:
        n = len(z["x"])
    return {"uid": np.array([state["uid"]]), "net": np.array([state["net"]]), "n": np.array([n]),
            "probes": np.array(state["probes"] or [0]), "planted": np.array([state["planted"]])}


STAGES = {"probe": Stage(timed=timed, finish=finish)}
'''


@unittest.skipUnless(ROOT_ONLY, "needs root and the users abench-agent / abench-judge")
class JudgeChildIsolation(unittest.TestCase):
    def test_stage_child_runs_as_judge_user_without_network(self):
        import numpy as np
        import oj_eval
        tmp = Path(tempfile.mkdtemp(prefix="abench-judge-test-"))
        try:
            os.chmod(tmp, 0o711)
            h = tmp / "harness"
            h.mkdir()
            shutil.copyfile(ROOT / "tools" / "harness_kit" / "abench_stage.py", h / "abench_stage.py")
            (h / "stages.py").write_text(STAGES)
            oj_eval._share(h)
            sub = tmp / "sub"
            sub.mkdir()
            oj_eval._share(sub)
            held = tmp / "held_out_inputs.npz"
            np.savez(held, x=np.arange(7))
            held.chmod(0o600)
            kdir = tmp / "keys"                    # like ~/.abench: the key's directory is closed to others
            kdir.mkdir(mode=0o700)
            secret = kdir / "secret.key"
            secret.write_text("k")
            secret.chmod(0o600)
            os.environ["ABENCH_JUDGE_CHILD_USER"] = "abench-judge"
            try:
                iso = oj_eval.judge_isolation(secret, {"data": {}})
            finally:
                del os.environ["ABENCH_JUDGE_CHILD_USER"]
            self.assertEqual(iso["network_probe"], "offline")
            spec = __import__("importlib.util").util.spec_from_file_location("abench_stage_iso_test", h / "abench_stage.py")
            kit = __import__("importlib.util").util.module_from_spec(spec)
            sys.modules["abench_stage_iso_test"] = kit
            spec.loader.exec_module(kit)
            stages = kit.load_module(h / "stages.py", "iso_test_stages")
            res = kit.run_pipeline(stages, h / "stages.py", sub, tmp / "run0", public=None, models=None, held_out=held,
                                   seed=0, device="cpu", isolation=iso,
                                   extra_env={"PROBE_PATHS": os.pathsep.join([str(held), str(secret), str(ROOT), "/root"])})
            self.assertEqual(res["status"], "ok", json.dumps(res["stages"], default=str)[-2000:])
            out = res["outputs"]["probe"]
            self.assertEqual(int(out["uid"][0]), pwd.getpwnam("abench-judge").pw_uid)
            self.assertEqual(int(out["net"][0]), 0)
            self.assertEqual(int(out["n"][0]), 7, "held-out inputs must reach the child after the timed phase")
            self.assertEqual(out["probes"].tolist(), [0, 0, 0, 0], "the child reached a judge-side path")
            self.assertTrue(held.exists() and not any((tmp / "run0" / "probe" / "post").glob("*")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_preflight_refuses_a_readable_key(self):
        import oj_eval
        d = Path(tempfile.mkdtemp())
        try:
            os.chmod(d, 0o755)
            key = d / "k"
            key.write_text("x")
            key.chmod(0o644)
            os.environ["ABENCH_JUDGE_CHILD_USER"] = "abench-judge"
            try:
                with self.assertRaises(SystemExit):
                    oj_eval.judge_isolation(key, {"data": {}})
            finally:
                del os.environ["ABENCH_JUDGE_CHILD_USER"]
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(ROOT_ONLY, "needs root and the users abench-agent / abench-judge")
class AgentUserPreflight(unittest.TestCase):
    def test_preflight_and_cleanup(self):
        d = Path(tempfile.mkdtemp(prefix="abench-pf-"))
        try:
            os.chmod(d, 0o711)
            paths = {"workspace": d / "ws", "view": d / "view"}
            paths["workspace"].mkdir()
            pw = pwd.getpwnam("abench-agent")
            os.chown(paths["workspace"], pw.pw_uid, pw.pw_gid)
            paths["view"].mkdir()
            (paths["view"] / "task.md").write_text("t")
            os.chmod(paths["view"], 0o755)
            (paths["view"] / "task.md").chmod(0o444)
            rec = sandbox_runner.isolation_preflight("abench-agent", paths, None, {"data": {}})
            self.assertTrue(rec["workspace_writable"] and rec["view_readable"], rec)
            self.assertFalse(any(c["reachable"] for c in rec["forbidden_paths"]), rec)
            # a stray process and a stray file of the agent user are removed after the run
            marker = "/tmp/abench-stray-" + uuid.uuid4().hex[:8]
            subprocess.run(["sudo", "-u", "abench-agent", "--", "sh", "-c",
                            f"touch {marker}; (sleep 300 >/dev/null 2>&1 &)"], check=True)
            rec = sandbox_runner.cleanup_user("abench-agent", d)
            self.assertGreaterEqual(rec["stray_processes_killed"], 1)
            self.assertEqual(rec["processes_after"], 0)
            self.assertIn(marker, rec["stray_files_removed"])
            self.assertFalse(os.path.exists(marker))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

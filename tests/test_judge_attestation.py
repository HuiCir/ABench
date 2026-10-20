import os
import tempfile
import unittest
from unittest.mock import patch

from tools.oj_eval import TOOLS, _dependency_attestation, tree_hash


class JudgeAttestationTest(unittest.TestCase):
    def test_explicit_runtime_requires_tools_hash(self):
        with tempfile.TemporaryDirectory() as runtime:
            env = {
                "ABENCH_JUDGE_PYTHONPATH": runtime,
                "ABENCH_JUDGE_TOOLS_SHA256": tree_hash(TOOLS),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertTrue(_dependency_attestation()["trusted"])

    def test_wrong_tools_hash_is_not_scoreable(self):
        with tempfile.TemporaryDirectory() as runtime:
            env = {
                "ABENCH_JUDGE_PYTHONPATH": runtime,
                "ABENCH_JUDGE_TOOLS_SHA256": "0" * 64,
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertFalse(_dependency_attestation()["trusted"])


if __name__ == "__main__":
    unittest.main()

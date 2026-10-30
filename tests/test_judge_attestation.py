import os
import unittest
from unittest.mock import patch
from tools.oj_eval import TOOLS, _dependency_attestation, tree_hash


class JudgeAttestationTest(unittest.TestCase):
    def test_locked_image_requires_matching_host_tools(self):
        image='registry.example/cpu@sha256:'+'a'*64
        with patch.dict(os.environ, {'ABENCH_JUDGE_TOOLS_SHA256':tree_hash(TOOLS)}):
            self.assertTrue(_dependency_attestation(image,locked=True)['trusted'])
            self.assertFalse(_dependency_attestation(image,locked=False)['trusted'])
            self.assertFalse(_dependency_attestation()['trusted'])

    def test_wrong_tools_hash_is_not_scoreable(self):
        with patch.dict(os.environ, {'ABENCH_JUDGE_TOOLS_SHA256':'0'*64}):
            self.assertFalse(_dependency_attestation('registry.example/cpu@sha256:'+'a'*64,locked=True)['trusted'])

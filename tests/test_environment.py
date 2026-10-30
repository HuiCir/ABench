"""Image software identity must not depend on whether a GPU was injected during the build check."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from record_environment import _torch

class EnvironmentTest(unittest.TestCase):
    def test_nccl_recorded_without_visible_gpu(self):
        fake=SimpleNamespace(__version__='2.8.0',version=SimpleNamespace(cuda='12.8',git_version='test'),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:91002)),
            cuda=SimpleNamespace(is_available=lambda:False,nccl=SimpleNamespace(version=lambda:(2,27,3))))
        with patch.dict(sys.modules,{'torch':fake}):
            self.assertEqual(_torch()['nccl'],'2.27.3')

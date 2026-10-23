"""verify.csv must list every case of index.yaml and agree with the repository (tools/verify_registry.py --check)."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VerifyRegistry(unittest.TestCase):
    def test_check(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "verify_registry.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:])


if __name__ == "__main__":
    unittest.main()

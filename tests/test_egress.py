"""Hostname egress policy and hash-chained TCP proxy for host smoke."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import egress_proxy as E  # noqa: E402


def _oj(*urls):
    return {"data": {"sources": [{"name": "x", "url": u} for u in urls]}}


class Policy(unittest.TestCase):
    def test_literature_allowed_hubs_denied(self):
        pol = E.policy_for_case(_oj("https://data.openml.org/datasets/1.pq"), "command")
        self.assertTrue(E.decide(pol, "arxiv.org", 443)[0])
        self.assertTrue(E.decide(pol, "export.arxiv.org", 443)[0])
        self.assertTrue(E.decide(pol, "github.com", 443)[0])
        self.assertFalse(E.decide(pol, "huggingface.co", 443)[0])
        self.assertFalse(E.decide(pol, "cdn-lfs.huggingface.co", 443)[0])
        self.assertFalse(E.decide(pol, "data.openml.org", 443)[0], "the case's own data source")
        self.assertFalse(E.decide(pol, "example.com", 443)[0], "not on the allow list")
        self.assertFalse(E.decide(pol, "arxiv.org", 22)[0], "port")
        self.assertFalse(E.decide(pol, "evilarxiv.org", 443)[0], "suffix match needs a dot")

    def test_github_source_denies_the_family(self):
        pol = E.policy_for_case(_oj("https://github.com/clinc/oos-eval"), "command")
        for h in ("github.com", "api.github.com", "raw.githubusercontent.com", "codeload.github.com"):
            self.assertFalse(E.decide(pol, h, 443)[0], h)

    def test_harness_and_case_overrides(self):
        oj = {"data": {}, "agent_network": {"allow": ["docs.example.org"], "deny": ["pypi.org"]}}
        pol = E.policy_for_case(oj, "claude")
        self.assertTrue(E.decide(pol, "api.anthropic.com", 443)[0])
        self.assertFalse(E.decide(E.policy_for_case({"data": {}}, "command"), "api.anthropic.com", 443)[0])
        self.assertTrue(E.decide(pol, "docs.example.org", 443)[0])
        self.assertFalse(E.decide(pol, "pypi.org", 443)[0])

    def test_every_case_has_a_policy(self):
        import yaml
        for oj_path in sorted((ROOT / "cases").glob("*/case/oj.yaml")):
            oj = yaml.safe_load(oj_path.read_text(encoding="utf-8"))
            pol = E.policy_for_case(oj, "claude")
            lit = E.load_policy()["literature"]
            for h in pol["case_source_hosts"]:
                if any(E._match(h, pat) for pat in lit):
                    self.assertTrue(E.decide(pol, h, 443)[0], f"paper host {h} must stay reachable")
                    continue
                self.assertFalse(E.decide(pol, h, 443)[0], f"{oj_path.parent.parent.name}: source {h} reachable")


class FakeUpstream:
    """A CONNECT proxy that answers 200 and then echoes: lets the test reach an 'allowed' host without a network."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(c,), daemon=True).start()

    def _one(self, c):
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += c.recv(4096)
        c.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
        while True:
            d = c.recv(4096)
            if not d:
                break
            c.sendall(d.upper())
        c.close()


class Proxy(unittest.TestCase):
    def _connect(self, port, host):
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        s.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = s.recv(4096)
            if not chunk:
                break
            head += chunk
        return s, head

    def test_allowed_denied_and_log_chain(self):
        d = Path(tempfile.mkdtemp())
        up = FakeUpstream()
        try:
            pol = E.policy_for_case({"data": {}}, "command")
            px = E.EgressProxy(pol, d / "egress.jsonl", upstream=up.url).start()
            s, head = self._connect(px.port, "arxiv.org")
            self.assertIn(b" 200 ", head)
            s.sendall(b"hello")
            self.assertEqual(s.recv(5), b"HELLO")
            s.close()
            s, head = self._connect(px.port, "huggingface.co")
            self.assertIn(b" 403 ", head)
            self.assertIn(b"egress policy", s.recv(4096) or head)
            s.close()
            import time
            time.sleep(0.3)
            summary = px.stop()
            self.assertEqual((summary["allowed"], summary["denied"]), (1, 1))
            lines = (d / "egress.jsonl").read_bytes().splitlines()
            prev = "0" * 64
            for line in lines:
                self.assertEqual(json.loads(line)["prev_sha256"], prev)
                prev = hashlib.sha256(line).hexdigest()
            recs = [json.loads(x) for x in lines]
            self.assertEqual({r["host"]: r["allowed"] for r in recs}, {"arxiv.org": True, "huggingface.co": False})
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_direct_mode_refuses_private_targets(self):
        self.assertIsNone(E._public_address("localhost", 443))
        self.assertIsNone(E._public_address("127.0.0.1", 443))


if __name__ == "__main__":
    unittest.main()

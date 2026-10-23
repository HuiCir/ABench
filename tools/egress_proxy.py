#!/usr/bin/env python3
"""Egress control for the agent stage: a logging forward proxy plus a per-user firewall (operator side).

The agent's session is online (spec/egress_policy.yaml): it may look up literature, code and documentation at any
point. The runner starts `EgressProxy` on 127.0.0.1 as the operator, points the agent's HTTP(S)_PROXY at it and,
with `Firewall`, rejects every other connection of the agent user (iptables/ip6tables owner match), so all traffic
passes the proxy. The proxy

  - accepts `CONNECT host:port` (HTTPS) and absolute-URI HTTP requests, for the ports in the policy;
  - allows or denies by host name (TLS is not inspected);
  - resolves names itself (the agent needs no DNS) and refuses hosts that resolve to loopback or private addresses;
  - optionally chains to an upstream proxy (ABENCH_EGRESS_UPSTREAM=http://host:port, for operator networks that
    themselves egress through a proxy);
  - writes one hash-chained JSON line per connection (time, method, host, port, decision, bytes each way, seconds)
    to the run record; the agent cannot write it. Connections that download more than LARGE_DOWNLOAD bytes are
    listed in the summary for the operator's review (data mirrored on an allowed host, e.g. GitHub, is the
    residual risk of host-level filtering).

Policy matching and the case deny list are in `policy_for_case`; `python3 tools/egress_proxy.py --case ID` prints
the effective policy of a case.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import select
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "spec" / "egress_policy.yaml"
LARGE_DOWNLOAD = 20 * 1024 * 1024        # connections that bring in more than this are listed for review
# hosts that serve the same repositories: denying one denies the family
FAMILIES = {
    "github.com": ["github.com", ".github.com", ".githubusercontent.com"],
    "huggingface.co": ["huggingface.co", ".huggingface.co", "hf.co", ".hf.co"],
}


# ------------------------------------------------------------------ policy
def _match(host: str, pattern: str) -> bool:
    host, pattern = host.lower().rstrip("."), pattern.lower()
    return host.endswith(pattern) if pattern.startswith(".") else host == pattern


def load_policy() -> dict[str, Any]:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "abench.egress/v1":
        raise RuntimeError(f"invalid egress policy: {POLICY_PATH}")
    return raw


def _hosts_of(obj: Any) -> list[str]:
    """Hosts of every http(s) URL in a nested oj.yaml data structure."""
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _hosts_of(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _hosts_of(v)
    elif isinstance(obj, str):
        for tok in obj.replace(",", " ").split():
            if tok.startswith(("http://", "https://")):
                h = urlsplit(tok).hostname
                if h:
                    out.append(h.lower())
    return out


def policy_for_case(oj: dict[str, Any], harness: str) -> dict[str, Any]:
    """Effective allow/deny lists of one case and harness."""
    p = load_policy()
    data = oj.get("data") or {}
    src_hosts = sorted(set(_hosts_of(data.get("sources")) + _hosts_of(data.get("models"))
                           + _hosts_of(data.get("judge_models"))))
    literature = list(p.get("literature") or [])
    case_deny: list[str] = []
    for h in src_hosts:
        if any(_match(h, pat) for pat in literature):
            continue                                   # a cited paper, not a data download
        fam = next((v for k, v in FAMILIES.items() if _match(h, k) or _match(h, "." + k)), None)
        case_deny += fam if fam else [h]
    net = oj.get("agent_network") or {}
    case_deny += list(net.get("deny") or [])
    allow = literature + list(p.get("allow") or []) + list((p.get("harness_allow") or {}).get(harness) or []) \
        + list(net.get("allow") or [])
    return {"policy_version": p["schema"], "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            "ports": [int(x) for x in p.get("ports") or [443]], "allow": allow,
            "deny": list(p.get("deny") or []), "case_deny": sorted(set(case_deny)), "case_source_hosts": src_hosts}


def decide(policy: dict[str, Any], host: str, port: int) -> tuple[bool, str]:
    if port not in policy["ports"]:
        return False, f"port {port} not allowed"
    for pat in policy["case_deny"]:
        if _match(host, pat):
            return False, f"case data/model source ({pat})"
    for pat in policy["deny"]:
        if _match(host, pat):
            return False, f"data or model hub ({pat})"
    for pat in policy["allow"]:
        if _match(host, pat):
            return True, f"allowed ({pat})"
    return False, "not on the allow list"


# ------------------------------------------------------------------ proxy
class _Log:
    def __init__(self, path: Path):
        self.path, self.prev, self.lock = path, "0" * 64, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def write(self, rec: dict[str, Any]) -> None:
        with self.lock:
            rec["prev_sha256"] = self.prev
            line = json.dumps(rec, sort_keys=True).encode()
            with open(self.path, "ab") as f:
                f.write(line + b"\n")
            self.prev = hashlib.sha256(line).hexdigest()


def _public_address(host: str, port: int) -> tuple[str, int] | None:
    """First resolved address that is globally routable (no loopback, private, link-local, ... targets)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return None
    for fam, _, _, _, addr in infos:
        ip = ipaddress.ip_address(addr[0])
        if ip.is_global:
            return addr[0], port
    return None


class EgressProxy:
    def __init__(self, policy: dict[str, Any], log_path: Path, upstream: str | None = None,
                 idle_timeout: float = 300.0):
        self.policy, self.log, self.idle = policy, _Log(log_path), idle_timeout
        self.upstream = None
        if upstream:
            u = urlsplit(upstream)
            self.upstream = (u.hostname, u.port or 80)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(256)
        self.port = self.sock.getsockname()[1]
        self.stats = {"connections": 0, "allowed": 0, "denied": 0, "bytes_down": 0, "bytes_up": 0,
                      "hosts": {}, "large_downloads": []}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._serve, daemon=True)

    # -- lifecycle
    def start(self) -> "EgressProxy":
        self._t.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        self._t.join(timeout=5)
        raw = self.log.path.read_bytes()
        hosts = sorted(self.stats["hosts"].items(), key=lambda kv: -kv[1]["n"])
        return {"connections": self.stats["connections"], "allowed": self.stats["allowed"],
                "denied": self.stats["denied"], "bytes_down": self.stats["bytes_down"],
                "bytes_up": self.stats["bytes_up"], "hosts": dict(hosts[:200]),
                "large_downloads": self.stats["large_downloads"][:100],
                "log_sha256": hashlib.sha256(raw).hexdigest(), "log_entries": raw.count(b"\n")}

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()

    # -- one client connection
    def _read_head(self, c: socket.socket) -> bytes | None:
        buf = b""
        c.settimeout(30)
        while b"\r\n\r\n" not in buf:
            chunk = c.recv(65536)
            if not chunk:
                return None
            buf += chunk
            if len(buf) > 65536:
                return None
        return buf

    def _record(self, method: str, host: str, port: int, ok: bool, why: str, up: int = 0, down: int = 0,
                seconds: float = 0.0, error: str | None = None) -> None:
        st = self.stats
        st["connections"] += 1
        st["allowed" if ok else "denied"] += 1
        st["bytes_up"] += up
        st["bytes_down"] += down
        h = st["hosts"].setdefault(host, {"n": 0, "allowed": ok, "why": why, "bytes_down": 0})
        h["n"] += 1
        h["bytes_down"] += down
        if down >= LARGE_DOWNLOAD:                 # for the operator's audit: data could hide in an allowed host
            st["large_downloads"].append({"host": host, "bytes_down": down, "t": round(time.time(), 1)})
        rec = {"t": round(time.time(), 3), "method": method, "host": host, "port": port, "allowed": ok,
               "why": why, "bytes_up": up, "bytes_down": down, "seconds": round(seconds, 3)}
        if error:
            rec["error"] = error[:200]
        self.log.write(rec)

    def _connect(self, host: str, port: int) -> socket.socket:
        if self.upstream:
            s = socket.create_connection(self.upstream, timeout=30)
            s.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = s.recv(4096)
                if not chunk:
                    raise OSError("upstream proxy closed the connection")
                head += chunk
            status = head.split(b"\r\n", 1)[0]
            if b" 200" not in status:
                s.close()
                raise OSError(f"upstream proxy refused: {status.decode('latin-1')[:80]}")
            return s
        addr = _public_address(host, port)
        if addr is None:
            raise OSError("host does not resolve to a public address")
        return socket.create_connection(addr, timeout=30)

    def _pipe(self, a: socket.socket, b: socket.socket, first_up: bytes = b"") -> tuple[int, int]:
        up = down = 0
        if first_up:
            b.sendall(first_up)
            up += len(first_up)
        a.settimeout(None)
        b.settimeout(None)
        socks = [a, b]
        while not self._stop.is_set():
            r, _, _ = select.select(socks, [], [], self.idle)
            if not r:
                break
            done = False
            for s in r:
                try:
                    data = s.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    done = True
                    break
                (b if s is a else a).sendall(data)
                if s is a:
                    up += len(data)
                else:
                    down += len(data)
            if done:
                break
        return up, down

    def _handle(self, c: socket.socket) -> None:
        t0 = time.time()
        host, port, method = "?", 0, "?"
        try:
            head = self._read_head(c)
            if not head:
                return
            line, _, rest = head.partition(b"\r\n")
            parts = line.decode("latin-1").split()
            if len(parts) != 3:
                c.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                return
            method, target, version = parts
            if method == "CONNECT":
                host, _, p = target.rpartition(":")
                host, port = host.strip("[]").lower(), int(p or 443)
                body = head.split(b"\r\n\r\n", 1)[1]
            else:
                u = urlsplit(target)
                if u.scheme != "http" or not u.hostname:
                    c.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\nabsolute http:// URI or CONNECT required\r\n")
                    return
                host, port = u.hostname.lower(), u.port or 80
                path = u.path or "/"
                if u.query:
                    path += "?" + u.query
                body = f"{method} {path} {version}\r\n".encode() + rest
            ok, why = decide(self.policy, host, port)
            if not ok:
                msg = f"blocked by the ABench sandbox egress policy: {host}: {why}\r\n".encode()
                c.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nContent-Length: "
                          + str(len(msg)).encode() + b"\r\nConnection: close\r\n\r\n" + msg)
                self._record(method, host, port, False, why, seconds=time.time() - t0)
                return
            try:
                s = self._connect(host, port)
            except OSError as e:
                c.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
                self._record(method, host, port, True, why, seconds=time.time() - t0, error=str(e))
                return
            with s:
                if method == "CONNECT":
                    c.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    up, down = self._pipe(c, s, body)
                else:
                    up, down = self._pipe(c, s, body)
            self._record(method, host, port, True, why, up, down, time.time() - t0)
        except (OSError, ValueError) as e:
            try:
                self._record(method, host, port, False, "proxy error", seconds=time.time() - t0, error=str(e))
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                c.close()
            except OSError:
                pass


# ------------------------------------------------------------------ firewall
def _listening_foreign_ports(uid: int) -> list[int]:
    """TCP ports in LISTEN state that belong to other users (from /proc/net/tcp{,6}); the agent must not reach the
    loopback ones (another user's service, or an operator proxy that would bypass the policy)."""
    ports = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            cols = line.split()
            if len(cols) < 8 or cols[3] != "0A":            # 0A = LISTEN
                continue
            if int(cols[7]) == uid:
                continue
            ports.add(int(cols[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


class Firewall:
    """iptables/ip6tables rules for one agent uid: TCP to the proxy port and to the agent's own loopback servers is
    allowed; TCP to loopback ports other users listen on when the run starts, and every other connection (any
    protocol, including DNS) is rejected. Services that other users start on loopback during the run are not
    covered."""

    def __init__(self, uid: int, proxy_port: int):
        self.uid, self.proxy_port = uid, proxy_port
        self.chain = f"ABENCH_EGRESS_{uid}"
        self.installed: list[str] = []
        self.foreign_ports: list[int] = []

    @staticmethod
    def available() -> tuple[bool, str]:
        import shutil
        if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
            return False, "needs root"
        if not Path("/proc/net/tcp").exists():
            return False, "/proc/net/tcp not found"
        for tool in ("iptables", "ip6tables"):
            if not shutil.which(tool):
                return False, f"{tool} not found"
        r = subprocess.run(["iptables", "-S", "OUTPUT"], capture_output=True, text=True)
        if r.returncode != 0:
            return False, "iptables unusable: " + r.stderr.strip()[:200]
        return True, "iptables owner match"

    def _run(self, tool: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run([tool, "-w", *args], capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{tool} {' '.join(args)}: {r.stderr.strip()}")
        return r

    def install(self) -> dict[str, Any]:
        self.foreign_ports = [p for p in _listening_foreign_ports(self.uid) if p != self.proxy_port]
        for tool, lo in (("iptables", "127.0.0.0/8"), ("ip6tables", "::1/128")):
            self._run(tool, "-N", self.chain, check=False)
            self._run(tool, "-F", self.chain)
            if tool == "iptables":
                self._run(tool, "-A", self.chain, "-p", "tcp", "-d", "127.0.0.1", "--dport", str(self.proxy_port),
                          "-j", "ACCEPT")
            for p in self.foreign_ports:
                self._run(tool, "-A", self.chain, "-p", "tcp", "-d", lo, "--dport", str(p), "-j", "REJECT")
            self._run(tool, "-A", self.chain, "-p", "tcp", "-o", "lo", "-d", lo, "-j", "ACCEPT")   # own servers
            self._run(tool, "-A", self.chain, "-j", "REJECT")
            self._run(tool, "-I", "OUTPUT", "1", "-m", "owner", "--uid-owner", str(self.uid), "-j", self.chain)
            self.installed.append(tool)
        return {"enforced": True, "method": "iptables/ip6tables owner match", "chain": self.chain,
                "proxy_port": self.proxy_port, "rejected_foreign_loopback_ports": self.foreign_ports}

    def remove(self) -> None:
        for tool in self.installed:
            while self._run(tool, "-D", "OUTPUT", "-m", "owner", "--uid-owner", str(self.uid), "-j", self.chain,
                            check=False).returncode == 0:
                pass
            self._run(tool, "-F", self.chain, check=False)
            self._run(tool, "-X", self.chain, check=False)
        self.installed = []


def main() -> int:
    ap = argparse.ArgumentParser(description="Print the effective egress policy of a case")
    ap.add_argument("--case", required=True)
    ap.add_argument("--harness", default="claude")
    ap.add_argument("--check", nargs="*", default=[], help="host names to test against the policy")
    a = ap.parse_args()
    oj = yaml.safe_load((ROOT / "cases" / a.case / "case" / "oj.yaml").read_text(encoding="utf-8"))
    pol = policy_for_case(oj, a.harness)
    print(json.dumps({k: pol[k] for k in ("case_source_hosts", "case_deny")}, indent=1))
    for h in a.check:
        print(h, decide(pol, h, 443))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

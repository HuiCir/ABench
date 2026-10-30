#!/usr/bin/env python3
"""Hostname policy and hash-chained egress proxy (Unix socket sidecar or host smoke TCP)."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import select
import socket
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


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
    import yaml
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


def policy_for_case(oj: dict[str, Any], harness: str, endpoint: dict | None = None) -> dict[str, Any]:
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
    if endpoint:
        official = list((p.get("harness_allow") or {}).get(harness) or [])
        allow = [h for h in allow if h not in official] + [endpoint["host"]]
    return {"model_endpoint": endpoint, "policy_version": p["schema"], "policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            "ports": [int(x) for x in p.get("ports") or [443]], "allow": allow,
            "deny": list(p.get("deny") or []), "case_deny": sorted(set(case_deny)), "case_source_hosts": src_hosts}


def decide(policy: dict[str, Any], host: str, port: int) -> tuple[bool, str]:
    ep = policy.get("model_endpoint")
    model_target = bool(ep and host.lower().rstrip(".") == ep["host"] and port == ep["port"])
    if ep and host.lower().rstrip(".") == ep["host"] and not model_target:
        return False, "model endpoint port mismatch"
    if port not in policy["ports"] and not model_target:
        return False, f"port {port} not allowed"
    for pat in policy["case_deny"]:
        if _match(host, pat):
            return False, f"case data/model source ({pat})"
    for pat in policy["deny"]:
        if _match(host, pat):
            return False, f"data or model hub ({pat})"
    if model_target:
        return True, "model_api"
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
                 idle_timeout: float = 300.0, unix_socket: str | Path | None = None):
        self.policy, self.log, self.idle = policy, _Log(log_path), idle_timeout
        self.upstream = None
        if upstream:
            u = urlsplit(upstream)
            self.upstream = (u.hostname, u.port or 80)
        self.unix_socket = Path(unix_socket) if unix_socket else None
        if self.unix_socket:
            self.unix_socket.parent.mkdir(parents=True, exist_ok=True)
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.bind(str(self.unix_socket))
            self.unix_socket.chmod(0o666)
            self.port = None
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", 0))
            self.port = self.sock.getsockname()[1]
        self.sock.settimeout(0.5)
        self.sock.listen(256)
        self.stats = {"connections": 0, "allowed": 0, "denied": 0, "bytes_down": 0, "bytes_up": 0,
                      "hosts": {}, "large_downloads": []}
        self._stop = threading.Event()
        self._connections = set()
        self._workers = []
        self._connection_lock = threading.Lock()
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
        with self._connection_lock:
            clients = list(self._connections)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + 5
        for worker in self._workers:
            worker.join(timeout=max(0, deadline-time.monotonic()))
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
            except socket.timeout:
                continue
            except OSError:
                return
            worker = threading.Thread(target=self._handle, args=(c,), daemon=True)
            with self._connection_lock:
                self._connections.add(c)
                self._workers.append(worker)
            worker.start()

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
        ep = self.policy.get("model_endpoint") or {}
        private_allowed = ep.get("allow_private") and host == ep.get("host") and port == ep.get("port")
        # Literal/private local targets are forbidden even when chaining. The trusted
        # upstream resolves public host names (local DNS may be a proxy fake-IP range).
        private_literal = host == "localhost" or host.endswith(".localhost")
        try:
            private_literal |= not ipaddress.ip_address(host).is_global
        except ValueError:
            pass
        if private_literal and not private_allowed:
            raise OSError("private target is not an explicitly allowed model endpoint")
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
        addr = (host, port) if private_allowed else _public_address(host, port)
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
            with self._connection_lock:
                self._connections.discard(c)
            try:
                c.close()
            except OSError:
                pass


def summarize_log(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    prev, ok = "0" * 64, True
    stats = {"connections": 0, "allowed": 0, "denied": 0, "bytes_down": 0, "bytes_up": 0,
             "hosts": {}, "large_downloads": []}
    for line in raw.splitlines():
        try:
            e = json.loads(line)
            ok &= e.get("prev_sha256") == prev
            prev = hashlib.sha256(line).hexdigest()
            stats["connections"] += 1
            stats["allowed" if e["allowed"] else "denied"] += 1
            for k in ("bytes_down", "bytes_up"):
                stats[k] += e.get(k, 0)
            h = stats["hosts"].setdefault(e["host"], {"n": 0, "allowed": e["allowed"], "bytes_down": 0})
            h["n"] += 1
            h["bytes_down"] += e.get("bytes_down", 0)
            if e.get("bytes_down", 0) >= LARGE_DOWNLOAD:
                stats["large_downloads"].append(e)
        except (ValueError, KeyError, TypeError):
            ok = False
    return {**stats, "chain_ok": bool(ok), "log_sha256": hashlib.sha256(raw).hexdigest(),
            "log_entries": len(raw.splitlines())}


def main() -> int:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        ap = argparse.ArgumentParser(description="Unix socket sidecar proxy")
        ap.add_argument("--policy", required=True)
        ap.add_argument("--socket", required=True)
        ap.add_argument("--log", required=True)
        a = ap.parse_args(sys.argv[2:])
        px = EgressProxy(json.loads(Path(a.policy).read_text()), Path(a.log),
                         upstream=os.environ.get("ABENCH_EGRESS_UPSTREAM"), unix_socket=a.socket).start()
        import signal
        done = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: done.set())
        done.wait()
        px.stop()
        return 0
    import yaml
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

"""Serving measurements shared by PRO-SRV-005 and PRO-SPEC-006 (evaluator code, agent-visible).

Server protocol (what the submission's serve.py must implement): it is started as
    python3 serve.py --model MODEL_DIR --port PORT
and must answer GET /health with status 200 once it can serve, and POST /generate with the JSON body
    {"input_ids": [int, ...], "max_new_tokens": int, "ignore_eos": true}
by streaming newline-delimited JSON objects {"ids": [int, ...]} (the new token ids, in order, any number per line)
and finally {"done": true}. Decoding is greedy. With ignore_eos the server produces exactly max_new_tokens tokens.

Timing (client side, this module): time to first token (TTFT) from sending the request to receiving the first id;
time per output token (TPOT) = (completion time - first-token time) / (tokens - 1).

Greedy consistency (greedy_check): the evaluator's own forward pass of the target model (transformers, bfloat16)
over prompt + returned tokens; a returned token is consistent if its logit is within TIE of the largest logit at its
position. A request is consistent if all its returned tokens are. This accepts any exact implementation of greedy
decoding (including speculative decoding with exact verification) up to bfloat16 rounding near ties.
"""
from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TIE = 0.25


def start_server(serve_py: Path, model_dir: Path, port: int, log: Path, ready_timeout: float = 1800.0,
                 env: dict | None = None) -> subprocess.Popen:
    f = open(log, "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(serve_py), "--model", str(model_dir), "--port", str(port)],
                         cwd=str(Path(serve_py).parent), stdout=f, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < ready_timeout:
        if p.poll() is not None:
            raise SystemExit(f"serve.py exited with code {p.returncode} before it was ready (see {log.name})")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return p
        except OSError:
            pass
        time.sleep(2)
    p.kill()
    raise SystemExit("serve.py was not ready in time")


def stop_server(p: subprocess.Popen) -> None:
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()


def request(port: int, input_ids: list, max_new: int, timeout: float = 600.0) -> dict:
    """One streamed request; returns ids, send time, first-token time and completion time (time.perf_counter)."""
    body = json.dumps({"input_ids": [int(x) for x in input_ids], "max_new_tokens": int(max_new), "ignore_eos": True})
    t0 = time.perf_counter()
    ids, t_first = [], None
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("POST", "/generate", body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            return {"ok": False, "ids": [], "t0": t0, "t_first": None, "t_end": time.perf_counter(), "error": r.status}
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("ids"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    ids.extend(int(x) for x in msg["ids"])
                if msg.get("done"):
                    buf = b""
                    break
        return {"ok": len(ids) == max_new, "ids": ids, "t0": t0, "t_first": t_first, "t_end": time.perf_counter()}
    except (OSError, ValueError, http.client.HTTPException) as e:
        return {"ok": False, "ids": ids, "t0": t0, "t_first": t_first, "t_end": time.perf_counter(), "error": str(e)}


def replay(port: int, prompts: list, max_new: list, arrivals: np.ndarray, workers: int = 1024) -> list:
    """Send request i at time arrivals[i] (seconds from the start); returns the results in request order."""
    res = [None] * len(prompts)
    start = time.perf_counter()

    def one(i):
        delay = start + float(arrivals[i]) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        res[i] = request(port, prompts[i], max_new[i])

    with ThreadPoolExecutor(max_workers=min(workers, len(prompts))) as ex:
        list(ex.map(one, range(len(prompts))))
    for r in res:
        r["t0"] -= start
        r["t_end"] -= start
        if r["t_first"] is not None:
            r["t_first"] -= start
    return res


def sequential(port: int, prompts: list, max_new: list) -> list:
    """Requests one after another (batch size 1)."""
    return [request(port, p, m) for p, m in zip(prompts, max_new)]


def latency_stats(results: list) -> dict:
    """Per-request TTFT and TPOT (seconds; NaN for failed requests)."""
    ttft = np.array([(r["t_first"] - r["t0"]) if r["ok"] and r["t_first"] is not None else np.nan for r in results])
    tpot = np.array([((r["t_end"] - r["t_first"]) / max(len(r["ids"]) - 1, 1)) if r["ok"] and r["t_first"] is not None
                     else np.nan for r in results])
    return {"ttft": ttft, "tpot": tpot, "ok": np.array([bool(r["ok"]) for r in results])}


def greedy_check(model_dir: Path, prompts: list, outputs: list, device: str = "cuda", tie: float = TIE) -> np.ndarray:
    """Per request: 1 if every returned token is within `tie` of the arg-max logit of the target model."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    ok = np.zeros(len(prompts), np.int64)
    with torch.no_grad():
        for i, (p, o) in enumerate(zip(prompts, outputs)):
            if not o:
                continue
            ids = torch.as_tensor([list(p) + list(o)], device=device)
            logits = m(ids).logits[0, len(p) - 1:len(p) - 1 + len(o)].float()
            chosen = logits.gather(1, torch.as_tensor(o, device=device)[:, None])[:, 0]
            ok[i] = int(bool(((logits.max(1).values - chosen) <= tie).all()))
    del m
    torch.cuda.empty_cache()
    return ok


class FakeServer:
    """A tiny threaded server speaking the protocol (for plumbing tests): it returns (last input id + k) mod V."""

    def __init__(self, port: int, vocab: int = 1000, delay: float = 0.0):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                req = json.loads(self.rfile.read(n))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                last = req["input_ids"][-1]
                for k in range(req["max_new_tokens"]):
                    time.sleep(outer.delay)
                    self.wfile.write((json.dumps({"ids": [(last + k + 1) % vocab]}) + "\n").encode())
                    self.wfile.flush()
                self.wfile.write(b'{"done": true}\n')

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.delay = delay
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()

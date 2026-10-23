"""PRO-SRV-005 pipeline: your serving system for Qwen3-8B is started on the 4-GPU node and loaded with replayed
production traffic at increasing rates; a sample of its answers is then checked against greedy decoding
(evaluator and self-check use this file; the evaluator runs the whole pipeline twice).

Stage "serve" (starts serve.py; the load generator is evaluator code, serve_bench.py):
  timed   python3 serve.py --model <models>/Qwen3-8B --port PORT, until GET /health answers 200 (≤ 1,800 s)
  finish  for each held-out trace (conversation, code): the trace's requests replayed with their inter-arrival times
          divided by each rate factor of LADDER in turn (stopping after the first factor at which fewer than 90%
          of requests meet the SLO); per request TTFT, TPOT, success; a fixed sample of the answers is saved for
          the next stage; the server is stopped                                                   (≤ 7,200 s)
Stage "check" (evaluator code only, after the server has stopped):
  serve_bench.greedy_check of the saved sample with the evaluator's own forward pass of Qwen3-8B
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import serve_bench as SB
from abench_stage import Context, Stage

MODEL, PORT = "Qwen3-8B", 18000
TTFT_SLO, TPOT_SLO, ATTAIN = 2.0, 0.10, 0.90
LADDER = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
TRACES = ("conv", "code")
N_CHECK, CHECK_TOKENS = 100, 256
PARAMS = {"model": MODEL, "ttft_slo_s": TTFT_SLO, "tpot_slo_s": TPOT_SLO, "attainment": ATTAIN, "ladder": list(LADDER),
          "n_check_per_trace": N_CHECK, "tie": SB.TIE}
HELD_OUT_KEYS = [f"{t}_{k}" for t in TRACES for k in ("ids", "offsets", "arrival", "max_new")] + ["check_idx"]
PIPELINE = [
    {"stage": "serve", "timeout": 1800 + 7200 + 300, "finish_cap": 7200, "tail_slack": 60, "threads": 64,
     "held_out": True, "models": True},
    {"stage": "check", "timeout": 2400, "finish_cap": 1800, "tail_slack": 30, "threads": 16, "held_out": False,
     "models": True, "carry": {"serve": ["sample.npz"]}},
]


def unpack(ids: np.ndarray, offsets: np.ndarray) -> list:
    return [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def attainment(results: list) -> float:
    st = SB.latency_stats(results)
    ok = st["ok"] & (np.nan_to_num(st["ttft"], nan=1e9) <= TTFT_SLO) & (np.nan_to_num(st["tpot"], nan=1e9) <= TPOT_SLO)
    return float(ok.mean())


def ladder(port: int, prompts: list, max_new: list, arrival: np.ndarray, factors=LADDER) -> dict:
    """SLO attainment and throughput at each rate factor until the first factor below ATTAIN."""
    att, tput, last = [], [], None
    for f in factors:
        res = SB.replay(port, prompts, max_new, arrival / f)
        att.append(attainment(res))
        span = max(r["t_end"] for r in res) - min(r["t0"] for r in res)
        tput.append(sum(len(r["ids"]) for r in res) / max(span, 1e-9))
        last = res
        if att[-1] < ATTAIN:
            break
    return {"attainment": np.asarray(att), "tokens_per_s": np.asarray(tput), "last": last}


def _timed_serve(ctx: Context, _):
    return SB.start_server(ctx.submission / "serve.py", ctx.models / MODEL, PORT, ctx.work / "server_log.txt")


def _finish_serve(ctx: Context, proc):
    out, sample = {}, {}
    try:
        with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
            held = {k: z[k] for k in z.files}
        check = held["check_idx"].astype(int)
        for t in TRACES:
            prompts = unpack(held[f"{t}_ids"], held[f"{t}_offsets"])
            max_new = held[f"{t}_max_new"].astype(int).tolist()
            arr = held[f"{t}_arrival"].astype(np.float64)
            r = ladder(PORT, prompts, max_new, arr)
            out[f"{t}_attainment"], out[f"{t}_tokens_per_s"] = r["attainment"], r["tokens_per_s"]
            out[f"{t}_native_rate"] = np.asarray([len(arr) / max(arr[-1] - arr[0], 1e-9)])
            idx = check[check < len(prompts)]
            chk = SB.sequential(PORT, [prompts[i] for i in idx], [min(CHECK_TOKENS, max_new[i]) for i in idx])
            for k, i in enumerate(idx):
                sample[f"{t}_{k}_p"] = np.asarray(prompts[i], np.int64)
                sample[f"{t}_{k}_o"] = np.asarray(chk[k]["ids"], np.int64)
            out[f"{t}_check_complete"] = np.asarray([float(np.mean([c["ok"] for c in chk]))])
    finally:
        SB.stop_server(proc)
    np.savez(ctx.work / "sample.npz", **sample)
    return out


def _finish_check(ctx: Context, _):
    with np.load(ctx.work / "sample.npz", allow_pickle=False) as z:
        keys = sorted({k.rsplit("_", 1)[0] for k in z.files})
        prompts = [z[k + "_p"].tolist() for k in keys]
        outs = [z[k + "_o"].tolist() for k in keys]
    ok = SB.greedy_check(ctx.models / MODEL, prompts, outs, ctx.device)
    return {"consistent": ok}


_id = lambda ctx, s: s                                                       # noqa: E731
STAGES = {"serve": Stage(timed=_timed_serve, finish=_finish_serve), "check": Stage(timed=_id, finish=_finish_check)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the public dev windows of both traces (dev.npz, prompts from FineWeb training documents);
    --quick: the first 200 requests of each and 20 checked answers."""
    with np.load(public / "dev.npz", allow_pickle=False) as z:
        held = {k: z[k] for k in z.files}
    if quick:
        for t in TRACES:
            n = min(200, len(held[f"{t}_arrival"]))
            held[f"{t}_ids"] = held[f"{t}_ids"][:held[f"{t}_offsets"][n]]
            held[f"{t}_offsets"] = held[f"{t}_offsets"][:n + 1]
            held[f"{t}_arrival"], held[f"{t}_max_new"] = held[f"{t}_arrival"][:n], held[f"{t}_max_new"][:n]
        held["check_idx"] = held["check_idx"][:20]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}

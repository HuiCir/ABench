#!/usr/bin/env python3
"""ABench agent CLI; workspaces, adapters and runtime lifecycles have separate modules."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
sys.path.insert(0,str(TOOLS))
# Stable Python entrypoints used by operators and existing tests.
from agents.workspace import (sha256_bytes, sha256_file, now, make_readonly, make_writable,
    validate_hidden_key, view_rel, build_agent_view, agent_env, agent_prompt, paper_index_info,
    filtered_paper_index, collect_retrieval_log, collect_submission)
from agents.trajectory import summarize_stream
from runtime.process import GpuMeter, budget_check, run_agent

def run(args):
    from agents.runner import run as execute
    return execute(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ABench v2 OJ-style runner")
    ap.add_argument("--case", required=True)
    from agents import REGISTRY
    ap.add_argument("--harness", "--agent", dest="harness", choices=list(REGISTRY), default="claude")
    ap.add_argument("--backend", choices=["docker", "host"], default="docker")
    ap.add_argument("--gpus")
    ap.add_argument("--allow-nonreference-gpu", action="store_true")
    ap.add_argument("--allow-shared-gpu", action="store_true", help="unscored Budget validation on sufficiently free, low-load GPUs")
    ap.add_argument("--image")
    ap.add_argument("--model", help="model configuration in spec/models.yaml (default: the harness's default)")
    ap.add_argument("--effort", help="override the model configuration's reasoning effort, e.g. high")
    ap.add_argument("--base-url")
    ap.add_argument("--allow-private-model-endpoint", action="store_true")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--suite", default="all")
    ap.add_argument("--command", help="command harness: the command to run; $ABENCH_* placeholders allowed")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--claude-model")
    ap.add_argument("--claude-effort")
    ap.add_argument("--claude-max-budget-usd", type=float)
    ap.add_argument("--pass-env", action="append", default=[], help="name of a host environment variable passed explicitly to the agent; repeatable")
    ap.add_argument("--hidden-key-file", help="hidden-set key; used only in the evaluation stage")
    ap.add_argument("--network", choices=["egress", "none", "open"], help="logged proxy, offline, or unscored unrestricted network")
    ap.add_argument("--defer-evaluation", action="store_true",
                    help="collect now without a final evaluation; judge later with ./abench judge --run")
    ap.add_argument("--device")
    ap.add_argument("--timeout-seconds", type=float, help="session deadline; defaults to the case declaration")
    ap.add_argument("--result-root")
    ap.add_argument("--run-id")
    ap.add_argument("--smoke", action="store_true", help="mark as smoke: records provenance=smoke_real and never enters formal scoring")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--papers-index", help="SQLite index from tools/paper_search/build_paper_index.py "
                                           "(default: $ABENCH_PAPERS_INDEX or ~/.abench/papers/current.sqlite)")
    ap.add_argument("--models-dir", help="frozen pretrained files for cases that declare data.models_dir "
                                         "(default: the path in oj.yaml)")
    a = ap.parse_args(argv)
    if a.harness == "command" and not a.command:
        ap.error("--command is required for --harness command")
    try:
        return run(a)
    except (ValueError, RuntimeError, OSError) as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())

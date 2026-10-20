#!/usr/bin/env bash
# ABench: launch the CPU-only standard case TC-FT-001 end to end on this machine, with the local Claude Code CLI
# (`claude -p`) as the agent.
#
#   bash ~/Documents/ABench/sandbox/run_local_claude_tc_ft_001.sh               # all steps
#   bash ~/Documents/ABench/sandbox/run_local_claude_tc_ft_001.sh preflight     # one step (see below)
#   DATA=synthetic bash .../run_local_claude_tc_ft_001.sh preflight key data mock judge   # plumbing only, no download
#
# Steps (default order: preflight key data mock agent judge)
#   preflight  Python packages; `claude` CLI; the EXACT argv the runner will use, run with the runner's isolated HOME
#              and the prompt on stdin (proves authentication and that every flag is accepted)
#   key        ~/.abench/hidden.key (created once, 0600, outside the repository)
#   data       DATA=real (default): download AG News at a pinned Hugging Face revision and freeze it.
#              DATA=synthetic: generated plumbing data (never scorable; rerun `data` with DATA=real afterwards)
#   mock       the plumbing agent sandbox/mock_agent.sh through the same runner: reads the view, calls the paper tool,
#              trains the reference code, runs selfcheck, writes a template report, then is judged. ~1-3 minutes, no
#              tokens. If this fails, the claude run would fail for the same reason.
#   agent      tools/sandbox_runner.py --harness claude --smoke: the real agent run (AGENT_TIMEOUT seconds, default 3600)
#   judge      tools/oj_eval.py on the last agent run's submission and on the reference submission (its checkpoint is
#              trained here first): calibration runs (the case stays awaiting_data, so they are never scorable)
#
# Options (environment): DATA=real|synthetic, AGENT_TIMEOUT, CLAUDE_MODEL, CLAUDE_EFFORT, MAX_BUDGET_USD, RUN_ID,
# PYTHON. Authentication for the isolated HOME: export CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) or
# ANTHROPIC_API_KEY. Optional paper index: ~/.abench/papers/current.sqlite (tools/paper_search/README.md).
#
# Writes: runs/TC-FT-001/<run id>/ (immutable run records), ~/.abench/ (key, raw downloads, last run ids), and the frozen
# data of TC-FT-001 (sandbox/data, sandbox/hidden, case/data_lock.yaml, case/oj.yaml). Nothing else in the repository.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE=TC-FT-001
PY="${PYTHON:-python3}"
DATA="${DATA:-real}"
KEY="$HOME/.abench/hidden.key"
RAW="$HOME/.abench/raw/$CASE"
STATE="$HOME/.abench/state"
REV=eb185aade064a813bc0b7f42de02595523103ca4
cd "$ROOT"
mkdir -p "$STATE"

say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }
[ "$DATA" = real ] || [ "$DATA" = synthetic ] || fail "DATA must be real or synthetic"
[ "$DATA" = synthetic ] && export ABENCH_ALLOW_SYNTHETIC=1
new_run_id() { printf '%s-%s-%s' "$1" "$(date +%Y%m%d-%H%M%S)" "$("$PY" -c 'import uuid; print(uuid.uuid4().hex[:4])')"; }
auth_env() { for v in ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN; do [ -n "${!v:-}" ] && printf '%s\n' "$v"; done; return 0; }

preflight() {
  say "preflight"
  "$PY" - <<'PY' || fail "Python packages"
import importlib, sys
print("  python", sys.version.split()[0], sys.executable)
need = {"numpy": True, "yaml": True, "sklearn": True, "scipy": True, "pandas": False, "pyarrow": False}
missing = []
for m, required in need.items():
    try:
        mod = importlib.import_module(m)
        print(f"  {m:8s} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {m:8s} MISSING" + ("" if required else "  (needed only for DATA=real)"))
        if required:
            missing.append(m)
import sqlite3
try:
    sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(a)")
    print("  sqlite   FTS5 ok (paper search)")
except sqlite3.OperationalError:
    print("  sqlite   no FTS5: the paper search tool will not work")
if missing:
    names = {"yaml": "pyyaml", "sklearn": "scikit-learn"}
    print("install with:  python3 -m pip install --user " + " ".join(names.get(m, m) for m in missing))
    sys.exit(1)
PY
  if [ "$DATA" = real ]; then "$PY" -c "import pandas, pyarrow" 2>/dev/null || fail "DATA=real needs pandas and pyarrow: $PY -m pip install --user pandas pyarrow"; fi
  command -v claude >/dev/null || fail "claude CLI not found on PATH"
  echo "  claude   $(claude --version 2>/dev/null | head -1)"
  [ -f "$HOME/.abench/papers/current.sqlite" ] && echo "  papers   $HOME/.abench/papers/current.sqlite" \
    || echo "  papers   no index (optional): the agent's paper search will report it is unavailable"
  # the exact argv of tools/sandbox_runner.py (harness_command, no wrapper), prompt on stdin, isolated HOME
  local tmp out; tmp="$(mktemp -d)"
  local envs=(PATH="$PATH" HOME="$tmp" TMPDIR="$tmp")
  while read -r v; do [ -n "$v" ] && envs+=("$v=${!v}"); done < <(auth_env)
  local argv=(claude -p --output-format stream-json --verbose --no-session-persistence)
  [ -n "${CLAUDE_MODEL:-}" ] && argv+=(--model "$CLAUDE_MODEL")
  [ -n "${CLAUDE_EFFORT:-}" ] && argv+=(--effort "$CLAUDE_EFFORT")
  argv+=(--permission-mode acceptEdits --allowedTools Bash,Read,Write,Edit,Glob,Grep --add-dir "$tmp")
  out="$(cd "$tmp" && printf 'Reply with exactly the word: pong\n' | env -i "${envs[@]}" "${argv[@]}" 2>&1 || true)"
  rm -rf "$tmp"
  if printf '%s' "$out" | grep -q '"type":"result"' && printf '%s' "$out" | grep -qi pong; then
    echo "  claude -p accepts the runner's flags and authenticates with an isolated HOME"
  else
    printf '%s\n' "$out" | tail -5
    fail "claude -p did not answer with the runner's argv. If it is an authentication error: run 'claude setup-token' once and export CLAUDE_CODE_OAUTH_TOKEN=<token> (or ANTHROPIC_API_KEY)"
  fi
}

key() {
  say "key"
  mkdir -p "$HOME/.abench"; chmod 700 "$HOME/.abench"
  [ -f "$KEY" ] || "$PY" tools/hidden_store.py keygen "$KEY"
  chmod 600 "$KEY"; echo "  $KEY"
}

data() {
  say "data ($DATA)"
  [ -f "$KEY" ] || fail "run the key step first"
  if [ "$DATA" = synthetic ]; then
    "$PY" tools/prepare/run.py --case "$CASE" --synthetic 3000 --key-file "$KEY"
  else
    mkdir -p "$RAW"
    for f in train test; do
      out="$RAW/$f-00000-of-00001.parquet"
      [ -s "$out" ] || curl -fL --retry 3 -o "$out" \
        "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/$REV/data/$f-00000-of-00001.parquet"
      shasum -a 256 "$out"
    done
    "$PY" tools/prepare/run.py --case "$CASE" --raw-dir "$RAW" --key-file "$KEY"
  fi
  grep -E "synthetic|activation" "cases/$CASE/case/data_lock.yaml" | sed 's/^/  /'
}

lock_matches() {   # the frozen data must match DATA, so a synthetic plumbing run is never mistaken for a real one
  local syn; syn="$("$PY" -c "import yaml; print(bool(yaml.safe_load(open('cases/$CASE/case/data_lock.yaml')).get('synthetic')))" 2>/dev/null || echo none)"
  [ "$syn" = none ] && fail "no frozen data: run the data step"
  if [ "$DATA" = real ] && [ "$syn" = True ]; then fail "the frozen data is SYNTHETIC; rerun: DATA=real $0 data"; fi
  if [ "$DATA" = synthetic ] && [ "$syn" = False ]; then fail "the frozen data is real but DATA=synthetic"; fi
  return 0
}

evaluate() {   # evaluate <run dir> <submission dir> <out name>
  "$PY" tools/oj_eval.py --case "$CASE" --submission "$2" --hidden-key-file "$KEY" --out "$1/$3" >/dev/null 2>"$1/$3.stderr" || true
  if [ -s "$1/$3" ]; then
    "$PY" - "$1/$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  verdict={d.get('verdict')} calibration={d.get('calibration')} synthetic={d.get('synthetic_data', False)} "
      f"scorable={d.get('scorable')}")
for e in d.get("errors") or d.get("format_errors") or []:
    print("  error:", e)
m = d.get("metrics") or {}
if m:
    print("  " + "  ".join(f"{k}={v}" for k, v in m.items()))
PY
  else
    tail -5 "$1/$3.stderr"
  fi
}

run_agent() {   # run_agent <prefix> <runner args...>
  local prefix="$1"; shift
  local id="${RUN_ID:-$(new_run_id "$prefix")}"
  local pidx=(); [ -f "$HOME/.abench/papers/current.sqlite" ] && pidx=(--papers-index "$HOME/.abench/papers/current.sqlite")
  "$PY" tools/sandbox_runner.py --case "$CASE" --smoke --device cpu --run-id "$id" \
      ${pidx[@]+"${pidx[@]}"} "$@" > "$STATE/runner_$prefix.out" 2> "$STATE/runner_$prefix.err" || true
  echo "$id" > "$STATE/last_${prefix}_$CASE"
  local dir="runs/$CASE/$id"
  [ -f "$dir/agent_run.json" ] || { tail -20 "$STATE/runner_$prefix.err"; fail "runner did not produce $dir/agent_run.json"; }
  "$PY" - "$dir/agent_run.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  agent status={d['status']} wall={d['agent_wall_seconds']}s  kept={d['submission']['kept']}")
r = d.get("retrieval_log") or {}
print(f"  paper search calls={r.get('entries')} chain_ok={r.get('chain_ok')}")
t = d.get("trajectory") or {}
if t:
    print(f"  tool calls={t.get('tool_calls')}  result={(t.get('result') or {}).get('subtype')} "
          f"turns={(t.get('result') or {}).get('num_turns')} cost_usd={(t.get('result') or {}).get('total_cost_usd')}")
PY
  echo "  run record: $ROOT/$dir"
}

mock() {
  say "mock agent (plumbing, no tokens)"
  lock_matches
  run_agent mock --harness command --command "bash $ROOT/sandbox/mock_agent.sh" --timeout-seconds 900
  local dir="runs/$CASE/$(cat "$STATE/last_mock_$CASE")"
  grep -q '^\[mock-agent\] done' "$dir/logs/agent_stdout.log" || { tail -30 "$dir/logs/agent_stdout.log"; tail -20 "$dir/logs/agent_stderr.log"; fail "mock agent did not finish"; }
  evaluate "$dir" "$dir/submission" calibration_evaluation.json
  echo "  mock run OK: view, paper tool, baseline training, selfcheck, collection and judge all work"
}

agent() {
  say "agent (claude -p, smoke, host_soft)"
  lock_matches
  local args=(--harness claude --timeout-seconds "${AGENT_TIMEOUT:-3600}")
  while read -r v; do [ -n "$v" ] && args+=(--pass-env "$v"); done < <(auth_env)
  [ -n "${CLAUDE_MODEL:-}" ] && args+=(--claude-model "$CLAUDE_MODEL")
  [ -n "${CLAUDE_EFFORT:-}" ] && args+=(--claude-effort "$CLAUDE_EFFORT")
  [ -n "${MAX_BUDGET_USD:-}" ] && args+=(--claude-max-budget-usd "$MAX_BUDGET_USD")
  run_agent agent "${args[@]}"
}

judge() {
  say "judge (calibration runs, never scorable)"
  lock_matches
  local id="${RUN_ID:-$(cat "$STATE/last_agent_$CASE" 2>/dev/null || true)}"
  [ -n "$id" ] || fail "no agent run yet (run the agent step, or set RUN_ID)"
  local dir="runs/$CASE/$id"
  echo "  agent submission ($id):"
  if [ -n "$(ls -A "$dir/submission" 2>/dev/null)" ]; then
    evaluate "$dir" "$dir/submission" calibration_evaluation.json
  else
    echo "  nothing was collected from the agent"
  fi
  echo "  reference submission (checkpoint trained now with its train.py):"
  local ref="$dir/reference_submission"
  if [ ! -d "$ref/checkpoint" ]; then
    rm -rf "$ref"; cp -R "cases/$CASE/example_submission" "$ref"; rm -rf "$ref/__pycache__"
    (cd "$ref" && PYTHONDONTWRITEBYTECODE=1 "$PY" train.py --data "$ROOT/sandbox/data/$CASE" --out checkpoint)
  fi
  evaluate "$dir" "$ref" calibration_reference_submission.json
  echo "  run record: $ROOT/$dir"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(preflight key data mock agent judge)
for s in "${steps[@]}"; do
  case "$s" in
    preflight|key|data|mock|agent|judge) "$s" ;;
    *) fail "unknown step $s (preflight key data mock agent judge)" ;;
  esac
done
say "done"

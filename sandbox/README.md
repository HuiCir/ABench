# sandbox/

Operator-side files of the sandbox. The runner is `tools/sandbox_runner.py` ([spec/sandbox_runner.md](../spec/sandbox_runner.md));
the evaluation contract is [spec/oj_contract.md](../spec/oj_contract.md); the agent's network is governed by
[spec/egress_policy.yaml](../spec/egress_policy.yaml).

| Path | Contents |
|---|---|
| `templates/` | report templates copied into every agent view (`design_report.html`, `judgment_report.html`, `references.yaml`) |
| `data/<case>/` | frozen public data (written by `tools/prepare/run.py`; not in git) |
| `hidden/<case>/` | encrypted held-out splits (not in git); the key stays outside the repository (`~/.abench/`, mode 0600) |
| `models/<case>/`, `judge_models/<case>/` | frozen pretrained files (not in git) |
| `calibration/` | code-only OJ calibration: `calibrate.sh` (freeze, train the reference submission, judge it, summarize), `cases.yaml`, `summarize.py` |
| `mock_agent.sh` | the plumbing agent used by `run_local_claude_tc_ft_001.sh` |
| `run_local_claude_tc_ft_001.sh` | launch TC-FT-001 end to end with the local `claude` CLI (smoke on macOS: no user switch or egress firewall there) |

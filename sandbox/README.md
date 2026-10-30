# sandbox/

Operator files used by `./abench`; see [the runner contract](../spec/sandbox_runner.md),
[the OJ contract](../spec/oj_contract.md) and [the egress policy](../spec/egress_policy.yaml).

| Path | Contents |
|---|---|
| `templates/` | report and reference templates copied into the agent view |
| `data/<case>/` | frozen public data, not in git |
| `hidden/<case>/` | encrypted held-out splits, not in git; host-local key under `~/.abench/`, mode 0600 |
| `models/<case>/`, `judge_models/<case>/` | frozen model files, not in git |
| `calibration/cases.yaml` | oracle training flags and small synthetic data sizes |
| `calibration/summarize.py` | summarizes evaluation records; also available through `./abench summarize` |
| `mock_agent.sh` | deterministic agent for pipeline checks, invoked by the mock adapter |

Use `./abench calibrate ID --data synthetic` for an isolated Docker calibration. For real data use
`--data /path/to/raw`; generated held-out sets use `--data generated`. Run records remain under `runs/ID/RUN/`.
The former sudo calibration and local launch scripts have been replaced by `./abench calibrate` and
`./abench run`. The Docker CLI and ordinary user permissions are sufficient for current commands.

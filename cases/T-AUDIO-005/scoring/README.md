# T-AUDIO-005 scoring

Scoring is run server-side according to `case/eval_protocol.yaml` and `case/scoring_config.yaml`. Only real runs that pass the frozen-weight, resource, preprocessing, determinism and quality constraints enter the Pareto computation; without real references and hidden evaluation the result stays pending.

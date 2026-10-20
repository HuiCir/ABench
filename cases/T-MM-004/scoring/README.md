# T-MM-004 scoring contract

The scorer accepts only real run records produced by the benchmark operator and does not generate simulated results. It first applies legality, budget, equivalence, resource-hash and leakage gates, then computes the non-dominated set and hypervolume according to `case/scoring_config.yaml`. When real resources, reference runs or hidden evaluation are missing it returns `pending_reference_runs`.

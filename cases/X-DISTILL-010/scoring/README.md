# X-DISTILL-010 scoring contract

The scorer accepts only real run records produced by the benchmark operator; it first applies provenance, teacher/student-hash, budget, leakage, numerical-finiteness and resource gates, then computes the non-dominated set and hypervolume according to `case/scoring_config.yaml`. When real resources, reference runs or hidden evaluation are missing it returns `pending_reference_runs`.

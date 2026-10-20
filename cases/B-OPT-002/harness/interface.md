# Interface (B-OPT-002)

## Search space (frozen)

Candidate programs can only be program dicts accepted by `primitive_search.validate_program`:

- states: `m_fast`, `m_slow` (same shape as the parameter; the EMA recurrence is fixed by the framework with coefficients `beta_fast`, `beta_slow`);
- terminals: `g`, `m_fast`, `m_slow`, `delta`, `zero`, `half` and the constants `alpha`, `tau`;
- Unary primitives: `identity`, `sign`, `abs`, `sqrt`, `normalize_rms`;
- Binary primitives: `add`, `sub`, `mul`, `safe_div`, `min`, `max`;
- the update has the form `-lr * (update + weight_decay * p)`; `weight_decay` is added by the framework, and candidates may not carry their own learning-rate schedule.

Grammar version: `lion-primitive-v1`.

## Search tools

- `primitive_search.fixed_stream(seed=17, steps=64)`: the fixed synthetic gradient stream;
- `primitive_search.proxy_score(program, streams)`: a candidate's proxy score on the fixed gradient stream (for search only);
- `primitive_search.replay(program, streams)`: replays a candidate's updates step by step.

The search tries at most 256 candidates with the random seed fixed at 17. The search must replay deterministically: rerunning `search.py --seed 17`
should regenerate, line by line, the same candidates and proxy scores as `search_log.jsonl`.

## Final rule

The selected program is written as a `FunctionalRule` in `optimizer.py` (interface in `optim_api.py`, the same as B-OPT-001),
exported under the module-level name `RULE`; hyperparameters are declared in `hparam_space.yaml` (at most 4 tunable hyperparameters).
The total number of elements in state tensors must not exceed 3× the number of parameter elements.

## Submission directory

```
optimizer.py          # entry point, exports RULE
search.py             # search script, replayable
search_spec.yaml      # grammar version, seed, candidate cap, target, selected program_id
search_log.jsonl      # one line per candidate: program_id, program, proxy score, seed, time
hparam_space.yaml
report/index.html     # design report (arXiv HTML layout)
references.yaml
```

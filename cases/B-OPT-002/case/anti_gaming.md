# Anti-gaming checklist (B-OPT-002)

| # | Gaming method | Protection | Where it applies |
|---|---|---|---|
| 1 | Arbitrary Python search or implicitly added undeclared operators | Frozen AST grammar; every search_log line is parsed and unknown nodes are rejected | S1/S4 |
| 2 | Overfitting the fixed stream | The search stream is separate from the hidden validation stream; final evaluation reuses the scale, long-training and large-batch points of B-OPT-001 | S5 |
| 3 | Submitting a renamed Lion | Fixed-gradient-stream behavioral probes + hyperparameter-perturbation noise floor; the behavioral divergence threshold still applies | S5 |
| 4 | Reporting only the best candidate and hiding failed attempts | search_log must cover consecutive program_ids from 1 to best_eval, recording seed, status and score | S4 |
| 5 | Swapping the search budget for a slower proxy | At most 256 candidates, STREAM timing, step time no more than 1.25× AdamW | S4/S5 |
| 6 | Reading the validation set or training loss | Rules receive only `(p, g, step, state, lr)`; the searcher receives only the fixed gradient stream | harness |
| 7 | Cherry-picking seeds | The benchmark operator fixes seed=17 and replays independently, reporting rank correlation and evaluations to first reach the target | S5 |
| 8 | Padding with highly cited papers to inflate Q | Only sources actually used for the grammar/transfer count; decorative sources get usage=0 | S5 |
| 9 | Fabricated local discoveries | ML precedents searched in the hidden snapshot, each claim ablated, and sources required to resolve | S5 X |

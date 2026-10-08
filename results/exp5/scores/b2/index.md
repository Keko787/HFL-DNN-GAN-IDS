# Exp 5 scores: batch2

τ = 0.71, 0.82 (the first is the primary); Holm per study; alpha 0.05. Each study's page has its tables; the CSVs beside it hold every column.

| study | primary metric | variants | comparisons | claims | trials scored |
|---|---|---|---|---|---|
| [s51](s51.md) | sim_s_to_tau0.71 | 20 | 16 | 0 | 400/400 |
| [s52](s52.md) | deadline_miss_rate | 8 | 6 | 0 | 160/160 |
| [s53x](s53x.md) | sim_s_to_tau0.71 | 41 | 36 | 10 | 800/820 |
| [s54](s54.md) | network_aou_mean | 8 | 6 | 0 | 160/160 |
| [s58](s58.md) | network_aou_mean | 16 | 14 | 5 | 320/320 |
| [s55](s55.md) | sim_s_to_tau0.71 | 8 | 6 | 0 | 160/160 |
| [s57](s57.md) | round_close_rate_kmin1 | 8 | 6 | 2 | 160/160 |
| [s59x](s59x.md) | sim_s_to_tau0.71 | 10 | 8 | 1 | 200/200 |
| [s513](s513.md) | sim_s_to_tau0.71 | 5 | 4 | 3 | 100/100 |

Outputs the score step does not read (FerrySim and tool jobs; each is a JSON report of its own):

- `s54/o1_base`: `results/exp5/b2/s54/o1_base.json`

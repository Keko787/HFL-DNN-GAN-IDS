# Exp 5 scores: batch1

τ = 0.71, 0.82 (the first is the primary); Holm per study; alpha 0.05. Each study's page has its tables; the CSVs beside it hold every column.

| study | primary metric | variants | comparisons | claims | trials scored |
|---|---|---|---|---|---|
| [s53](s53.md) | sim_s_to_tau0.71 | 32 | 28 | 8 | 640/640 |
| [s59](s59.md) | sim_s_to_tau0.71 | 15 | 12 | 5 | 300/300 |
| [s511b](s511b.md) | sim_s_to_tau0.71 | 20 | 15 | 3 | 400/400 |
| [s514](s514.md) | network_aou_mean | 18 | 14 | 2 | 720/720 |

Outputs the score step does not read (FerrySim and tool jobs; each is a JSON report of its own):

- `s511a/auto`: `results/exp5/b1/s511a/auto.json`
- `s511a/subsets`: `results/exp5/b1/s511a/subsets.json`
- `s511a/local`: `results/exp5/b1/s511a/local.json`

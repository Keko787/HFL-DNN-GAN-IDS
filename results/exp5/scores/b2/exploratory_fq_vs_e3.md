# Exploratory: FQ (learned per-stop score inside FeRRy's plan) vs E3 (Chen et al.)

Not pre-registered (added 8 Oct 2026). N = 6, one mule, 20 paired seeds per cell; time to tau = 0.71 on the simulated clock over complete pairs; difference = reference - FQ (positive: FQ faster); Holm across the eight comparisons below. FQ flies Study 5.5's gamma = 0.75 pick. The 5.3 scores are unchanged.

## Comparisons

| cell | reference | n pairs | reference mean | FQ mean | difference [95% CI] | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|
| n6k1_knee | E3 | 20 | 279.9 | 180.3 | 99.6 [62.8, 140.5] | 0.0000 | 0.0001 | FQ-g75 |
| n6k1_knee | F | 20 | 194.0 | 180.3 | 13.8 [-0.4, 31.5] | 0.0995 | 0.4645 | no claim |
| n6k1_knee | FX | 20 | 178.1 | 180.3 | -2.2 [-10.3, 6.6] | 0.6121 | 1.0000 | no claim |
| n6k1_knee | D4 | 20 | 264.5 | 180.3 | 84.2 [52.2, 121.2] | 0.0000 | 0.0002 | FQ-g75 |
| n6k1_stress | E3 | 19 | 211.1 | 174.0 | 37.1 [-41.4, 102.9] | 0.1042 | 0.4645 | no claim |
| n6k1_stress | F | 20 | 178.0 | 181.6 | -3.6 [-15.1, 3.7] | 1.0000 | 1.0000 | no claim |
| n6k1_stress | FX | 20 | 175.3 | 181.6 | -6.4 [-17.4, 0.6] | 0.0929 | 0.4645 | no claim |
| n6k1_stress | D4 | 20 | 264.5 | 181.6 | 82.9 [51.5, 118.3] | 0.0000 | 0.0001 | FQ-g75 |

## Arms (descriptive)

| cell | arm | trials ok | reach tau | mean time to tau | updates/round | mission (s) | transit (s) | dwell (s) | band shares |
|---|---|---|---|---|---|---|---|---|---|
| n6k1_knee | FQ-g75 | 20 | 1.00 | 180.3 | 3.54 | 155 | 15 | 94 | {'medium': 0.44, 'narrow': 0.54, 'wide': 0.02} |
| n6k1_knee | E3 | 20 | 1.00 | 279.9 | 3.46 | 228 | 144 | 11 | {'wide': 1.0} |
| n6k1_knee | F | 20 | 1.00 | 194.0 | 3.54 | 168 | 15 | 107 | {'narrow': 0.89, 'medium': 0.11} |
| n6k1_knee | FX | 20 | 1.00 | 178.1 | 3.54 | 149 | 15 | 88 | {'medium': 0.46, 'narrow': 0.49, 'wide': 0.05} |
| n6k1_knee | D4 | 20 | 1.00 | 264.5 | 3.49 | 215 | 138 | 11 | {'wide': 1.0} |
| n6k1_stress | FQ-g75 | 20 | 1.00 | 181.6 | 3.23 | 151 | 24 | 81 | {'medium': 0.34, 'narrow': 0.53, 'wide': 0.13} |
| n6k1_stress | E3 | 20 | 0.95 | 211.1 | 2.69 | 181 | 103 | 10 | {'wide': 1.0} |
| n6k1_stress | F | 20 | 1.00 | 178.0 | 3.21 | 155 | 24 | 85 | {'medium': 0.31, 'narrow': 0.67, 'wide': 0.02} |
| n6k1_stress | FX | 20 | 1.00 | 175.3 | 3.23 | 143 | 24 | 72 | {'medium': 0.47, 'narrow': 0.39, 'wide': 0.14} |
| n6k1_stress | D4 | 20 | 1.00 | 264.5 | 3.49 | 215 | 138 | 11 | {'wide': 1.0} |

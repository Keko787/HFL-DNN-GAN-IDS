# Exploratory: FQ (learned per-stop score inside FeRRy's plan) vs E3 (Chen et al.)

Not pre-registered (added 8 Oct 2026). N = 6, one mule, 40 paired seeds per cell; time to tau = 0.71 on the simulated clock over complete pairs; difference = reference - FQ (positive: FQ faster); Holm across the eight comparisons below. FQ flies Study 5.5's gamma = 0.75 pick. The 5.3 scores are unchanged.

## Comparisons

| cell | reference | n pairs | reference mean | FQ mean | difference [95% CI] | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|
| n6k1_knee | E3 | 40 | 252.1 | 168.7 | 83.4 [53.5, 118.6] | 0.0000 | 0.0000 | FQ-g75 |
| n6k1_knee | F | 40 | 178.9 | 168.7 | 10.2 [1.5, 20.1] | 0.0228 | 0.1138 | no claim |
| n6k1_knee | FX | 40 | 168.1 | 168.7 | -0.6 [-6.0, 4.9] | 0.7776 | 1.0000 | no claim |
| n6k1_knee | D4 | 40 | 232.1 | 168.7 | 63.4 [36.0, 92.8] | 0.0000 | 0.0000 | FQ-g75 |
| n6k1_stress | E3 | 39 | 200.8 | 162.3 | 38.5 [-11.1, 84.0] | 0.0258 | 0.1138 | no claim |
| n6k1_stress | F | 40 | 167.6 | 166.4 | 1.3 [-9.9, 12.3] | 0.6377 | 1.0000 | no claim |
| n6k1_stress | FX | 40 | 165.1 | 166.4 | -1.3 [-7.3, 5.5] | 0.1823 | 0.5469 | no claim |
| n6k1_stress | D4 | 40 | 232.1 | 166.4 | 65.8 [33.9, 95.5] | 0.0000 | 0.0001 | FQ-g75 |

## Arms (descriptive)

| cell | arm | trials ok | reach tau | mean time to tau | updates/round | mission (s) | transit (s) | dwell (s) | band shares |
|---|---|---|---|---|---|---|---|---|---|
| n6k1_knee | FQ-g75 | 40 | 1.00 | 168.7 | 3.44 | 157 | 17 | 95 | {'medium': 0.42, 'narrow': 0.56, 'wide': 0.03} |
| n6k1_knee | E3 | 40 | 1.00 | 252.1 | 3.37 | 225 | 142 | 11 | {'wide': 1.0} |
| n6k1_knee | F | 40 | 1.00 | 178.9 | 3.44 | 169 | 18 | 106 | {'narrow': 0.83, 'medium': 0.17} |
| n6k1_knee | FX | 40 | 1.00 | 168.1 | 3.44 | 154 | 17 | 92 | {'medium': 0.41, 'narrow': 0.54, 'wide': 0.05} |
| n6k1_knee | D4 | 40 | 1.00 | 232.1 | 3.40 | 215 | 139 | 11 | {'wide': 1.0} |
| n6k1_stress | FQ-g75 | 40 | 1.00 | 166.4 | 3.14 | 146 | 20 | 81 | {'medium': 0.43, 'narrow': 0.49, 'wide': 0.08} |
| n6k1_stress | E3 | 40 | 0.97 | 200.8 | 2.58 | 180 | 103 | 10 | {'wide': 1.0} |
| n6k1_stress | F | 40 | 1.00 | 167.6 | 3.11 | 155 | 20 | 90 | {'medium': 0.28, 'narrow': 0.71, 'wide': 0.01} |
| n6k1_stress | FX | 40 | 1.00 | 165.1 | 3.13 | 143 | 20 | 78 | {'medium': 0.47, 'narrow': 0.45, 'wide': 0.08} |
| n6k1_stress | D4 | 40 | 1.00 | 232.1 | 3.40 | 215 | 139 | 11 | {'wide': 1.0} |

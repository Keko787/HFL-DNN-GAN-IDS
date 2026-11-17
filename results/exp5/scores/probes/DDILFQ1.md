# DDILFQ1: FQ-g75 and FQ-s1 on step 1's physics

Exploratory, not pre-registered (DeveloperDocs/Experiment_5_Loss_Probes_Plan.md). Reference F. Time to tau = 0.71 over the trials both arms reach it, and reach at tau (exact McNemar on the paired outcomes), each with a 95% bootstrap CI. Holm per family: the baselines on time, on reach, and FX on each apart. Difference = F - arm: time, negative when F is faster; reach, positive when F reaches tau more often.

DDIL build D0's endpoints (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md), paired by trial and seed, each in a Holm family of its own: hover per mission (sim_dwell_s_mean, lower is better), updates merged per mission (update_yield, higher), and time to tau with a trial that missed it given the trial's end, so no trial is dropped (lower).

Other pairs, in families of their own: FQ-s1 against FQ-g75, FQ-s1 against FX, FQ-g75 against FX; the difference is the second arm minus the first.

Flown and scored: 44 of 44 arm-cells.

## Where a baseline beats F

On time to τ, reach, updates per mission, or time with misses at the end. A baseline's lower hover is listed in the cells' tables, not here: it comes from leaving the members below the floor unserved.

- none (so far)

## n6k1_knee_d1base

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 166.6 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 148.0 | 18.6 [8.8, 30.0] | 0.0147 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 286.3 | -119.6 [-180.0, -70.7] | 0.0002 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 276.9 | -110.3 [-159.0, -70.0] | 0.0002 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 265.7 | -99.1 [-145.4, -64.7] | 0.0001 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 290.1 | -123.5 [-185.2, -74.8] | 0.0002 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 246.1 | -79.5 [-116.4, -50.9] | 0.0002 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 266.7 | -100.1 [-148.9, -59.7] | 0.0002 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| FQ-g75 | 20 | 1.00 | 157.5 | 9.1 [0.5, 18.8] | 0.3916 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-g75 only | 1.0000 | no claim |
| FQ-s1 | 20 | 1.00 | 165.1 | 1.5 [-4.2, 7.5] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-s1 only | 1.0000 | no claim |

D0's endpoints in n6k1_knee_d1base, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 76.0 |  |  |  | 3.55 |  |  |  | 166.6 |  |  |  |
| FX | 67.7 | 8.4 [4.6, 12.7] | 0.0197 | FX | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 148.0 | 18.6 [8.8, 30.0] | 0.0147 | FX |
| H1 | 11.9 | 64.2 [55.0, 73.0] | 0.0000 | H1 | 3.40 | 0.15 [0.04, 0.31] | 0.1413 | no claim | 286.3 | -119.6 [-180.0, -70.7] | 0.0002 | F |
| MAX-AoI | 11.6 | 64.5 [55.4, 73.3] | 0.0000 | MAX-AoI | 3.42 | 0.12 [0.04, 0.25] | 0.1413 | no claim | 276.9 | -110.3 [-159.0, -70.0] | 0.0002 | F |
| Oort | 11.6 | 64.5 [54.9, 73.6] | 0.0000 | Oort | 3.48 | 0.07 [0.00, 0.16] | 0.3264 | no claim | 265.7 | -99.1 [-145.4, -64.7] | 0.0001 | F |
| Whittle | 10.8 | 65.3 [55.9, 74.2] | 0.0000 | Whittle | 3.40 | 0.15 [0.04, 0.29] | 0.1413 | no claim | 290.1 | -123.5 [-185.2, -74.8] | 0.0002 | F |
| FedEx route | 11.3 | 64.7 [55.6, 73.2] | 0.0000 | FedEx route | 3.54 | 0.01 [0.00, 0.04] | 0.3264 | no claim | 246.1 | -79.5 [-116.4, -50.9] | 0.0002 | F |
| FedCS | 11.4 | 64.7 [55.3, 73.6] | 0.0000 | FedCS | 3.52 | 0.03 [0.00, 0.06] | 0.3264 | no claim | 266.7 | -100.1 [-148.9, -59.7] | 0.0002 | F |
| FQ-g75 | 73.3 | 2.7 [-1.9, 6.9] | 0.9307 | no claim | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 157.5 | 9.1 [0.5, 18.8] | 0.4272 | no claim |
| FQ-s1 | 73.3 | 2.7 [-0.4, 5.8] | 0.2618 | no claim | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 165.1 | 1.5 [-4.2, 7.5] | 1.0000 | no claim |

## n6k1_knee_d1ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.9 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 125.4 | 25.6 [9.9, 46.8] | 0.0206 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 277.2 | -126.2 [-183.6, -69.5] | 0.0044 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 287.7 | -136.8 [-204.3, -72.4] | 0.0052 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 0.95 | 246.3 | -98.4 [-143.1, -41.5] | 0.0152 | F | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 0.95 | 263.4 | -115.5 [-173.9, -70.3] | 0.0010 | F | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 215.9 | -65.0 [-111.3, -10.7] | 0.0660 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 260.0 | -109.0 [-163.7, -56.4] | 0.0088 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| FQ-g75 | 20 | 1.00 | 127.3 | 23.6 [8.3, 44.4] | 0.0206 | FQ-g75 | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-g75 only | 1.0000 | no claim |
| FQ-s1 | 20 | 0.95 | 132.3 | 19.4 [4.2, 40.8] | 0.0829 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by FQ-s1 only | 1.0000 | no claim |

D0's endpoints in n6k1_knee_d1ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 59.3 |  |  |  | 3.54 |  |  |  | 150.9 |  |  |  |
| FX | 46.6 | 12.8 [8.5, 17.3] | 0.0066 | FX | 3.55 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 125.4 | 25.6 [9.9, 46.8] | 0.0206 | FX |
| H1 | 15.0 | 44.3 [40.4, 48.3] | 0.0000 | H1 | 2.73 | 0.81 [0.61, 1.01] | 0.0024 | F | 277.2 | -126.2 [-183.6, -69.5] | 0.0044 | F |
| MAX-AoI | 15.5 | 43.9 [39.7, 47.5] | 0.0000 | MAX-AoI | 2.79 | 0.75 [0.57, 0.94] | 0.0019 | F | 287.7 | -136.8 [-204.3, -72.4] | 0.0052 | F |
| Oort | 14.4 | 45.0 [39.9, 49.8] | 0.0000 | Oort | 2.86 | 0.68 [0.54, 0.82] | 0.0018 | F | 264.8 | -113.9 [-166.2, -56.5] | 0.0088 | F |
| Whittle | 17.7 | 41.7 [36.0, 47.0] | 0.0000 | Whittle | 2.94 | 0.60 [0.42, 0.78] | 0.0028 | F | 281.0 | -130.1 [-195.1, -77.3] | 0.0005 | F |
| FedEx route | 16.5 | 42.8 [37.5, 47.8] | 0.0000 | FedEx route | 2.94 | 0.60 [0.40, 0.82] | 0.0024 | F | 215.9 | -65.0 [-111.3, -10.7] | 0.0660 | no claim |
| FedCS | 14.8 | 44.5 [39.3, 49.4] | 0.0000 | FedCS | 2.94 | 0.60 [0.39, 0.80] | 0.0028 | F | 260.0 | -109.0 [-163.7, -56.4] | 0.0088 | F |
| FQ-g75 | 49.4 | 9.9 [6.6, 13.2] | 0.0066 | FQ-g75 | 3.55 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 127.3 | 23.6 [8.3, 44.4] | 0.0206 | FQ-g75 |
| FQ-s1 | 48.4 | 10.9 [7.2, 15.1] | 0.0128 | FQ-s1 | 3.52 | 0.01 [-0.04, 0.07] | 1.0000 | no claim | 144.0 | 6.9 [-25.5, 36.3] | 0.6188 | no claim |

## n6k1_knee_d1ampharshpc

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.3 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 144.1 | 6.2 [-6.2, 17.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 284.3 | -133.9 [-203.4, -63.9] | 0.0203 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 286.2 | -135.9 [-206.6, -64.9] | 0.0203 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 246.4 | -96.0 [-147.0, -48.6] | 0.0119 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.2 | -113.8 [-184.0, -43.8] | 0.0284 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 259.8 | -109.5 [-174.0, -49.1] | 0.0233 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 257.3 | -106.9 [-171.8, -40.0] | 0.0389 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| FQ-g75 | 20 | 1.00 | 143.6 | 6.7 [-7.2, 20.2] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-g75 only | 1.0000 | no claim |
| FQ-s1 | 20 | 1.00 | 131.3 | 19.1 [1.4, 38.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-s1 only | 1.0000 | no claim |

D0's endpoints in n6k1_knee_d1ampharshpc, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 58.3 |  |  |  | 3.54 |  |  |  | 150.3 |  |  |  |
| FX | 54.6 | 3.7 [-1.3, 9.3] | 1.0000 | no claim | 3.54 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 144.1 | 6.2 [-6.2, 17.7] | 1.0000 | no claim |
| H1 | 16.1 | 42.2 [36.8, 47.2] | 0.0000 | H1 | 2.83 | 0.71 [0.53, 0.90] | 0.0024 | F | 284.3 | -133.9 [-203.4, -63.9] | 0.0203 | F |
| MAX-AoI | 16.0 | 42.3 [36.8, 47.4] | 0.0000 | MAX-AoI | 2.79 | 0.75 [0.59, 0.91] | 0.0024 | F | 286.2 | -135.9 [-206.6, -64.9] | 0.0203 | F |
| Oort | 15.3 | 43.0 [37.8, 48.0] | 0.0000 | Oort | 2.83 | 0.71 [0.51, 0.91] | 0.0028 | F | 246.4 | -96.0 [-147.0, -48.6] | 0.0111 | F |
| Whittle | 15.8 | 42.5 [36.9, 48.0] | 0.0000 | Whittle | 2.85 | 0.69 [0.54, 0.84] | 0.0019 | F | 264.2 | -113.8 [-184.0, -43.8] | 0.0284 | F |
| FedEx route | 16.3 | 42.0 [36.3, 47.9] | 0.0000 | FedEx route | 3.04 | 0.50 [0.33, 0.68] | 0.0028 | F | 259.8 | -109.5 [-174.0, -49.1] | 0.0233 | F |
| FedCS | 17.3 | 41.0 [35.6, 46.1] | 0.0000 | FedCS | 2.90 | 0.64 [0.44, 0.82] | 0.0034 | F | 257.3 | -106.9 [-171.8, -40.0] | 0.0389 | F |
| FQ-g75 | 55.5 | 2.8 [-2.6, 9.3] | 1.0000 | no claim | 3.54 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 143.6 | 6.7 [-7.2, 20.2] | 1.0000 | no claim |
| FQ-s1 | 53.9 | 4.4 [-1.0, 11.0] | 1.0000 | no claim | 3.55 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 131.3 | 19.1 [1.4, 38.1] | 1.0000 | no claim |

## n6k1_knee_d1burst40

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 229.7 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 217.1 | 12.6 [-5.4, 30.2] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 265.4 | -35.7 [-103.2, 44.8] | 0.6636 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 0.95 | 256.9 | -36.1 [-108.6, 49.6] | 0.6636 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 271.3 | -41.6 [-123.4, 47.7] | 0.6636 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 225.7 | 4.0 [-56.6, 77.1] | 0.6742 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 269.2 | -39.5 [-92.9, 15.0] | 0.6636 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 283.7 | -54.0 [-120.6, 27.0] | 0.4175 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| FQ-g75 | 20 | 1.00 | 226.7 | 3.0 [-12.9, 18.4] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-g75 only | 1.0000 | no claim |
| FQ-s1 | 20 | 1.00 | 230.6 | -0.9 [-6.7, 5.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FQ-s1 only | 1.0000 | no claim |

D0's endpoints in n6k1_knee_d1burst40, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 110.0 |  |  |  | 3.24 |  |  |  | 229.7 |  |  |  |
| FX | 96.7 | 13.4 [4.5, 22.7] | 0.1850 | no claim | 3.27 | -0.04 [-0.28, 0.19] | 1.0000 | no claim | 217.1 | 12.6 [-5.4, 30.2] | 1.0000 | no claim |
| H1 | 12.7 | 97.3 [86.5, 107.9] | 0.0000 | H1 | 2.48 | 0.76 [0.56, 0.99] | 0.0028 | F | 265.4 | -35.7 [-103.2, 44.8] | 0.6636 | no claim |
| MAX-AoI | 12.7 | 97.4 [87.1, 108.2] | 0.0000 | MAX-AoI | 2.45 | 0.79 [0.55, 1.05] | 0.0028 | F | 269.1 | -39.4 [-108.3, 41.4] | 0.6636 | no claim |
| Oort | 12.3 | 97.7 [87.2, 107.7] | 0.0000 | Oort | 2.42 | 0.81 [0.59, 1.05] | 0.0028 | F | 271.3 | -41.6 [-123.4, 47.7] | 0.6636 | no claim |
| Whittle | 12.5 | 97.5 [87.1, 107.8] | 0.0000 | Whittle | 2.65 | 0.59 [0.34, 0.84] | 0.0077 | F | 225.7 | 4.0 [-56.6, 77.1] | 0.6742 | no claim |
| FedEx route | 12.0 | 98.0 [87.5, 109.2] | 0.0000 | FedEx route | 2.19 | 1.05 [0.68, 1.49] | 0.0034 | F | 269.2 | -39.5 [-92.9, 15.0] | 0.6636 | no claim |
| FedCS | 12.6 | 97.4 [85.4, 108.8] | 0.0000 | FedCS | 2.34 | 0.90 [0.61, 1.21] | 0.0027 | F | 283.7 | -54.0 [-120.6, 27.0] | 0.4175 | no claim |
| FQ-g75 | 107.4 | 2.6 [-7.2, 12.0] | 1.0000 | no claim | 3.33 | -0.09 [-0.30, 0.14] | 1.0000 | no claim | 226.7 | 3.0 [-12.9, 18.4] | 1.0000 | no claim |
| FQ-s1 | 105.7 | 4.3 [-0.8, 9.0] | 0.3075 | no claim | 3.24 | 0.00 [-0.04, 0.04] | 1.0000 | no claim | 230.6 | -0.9 [-6.7, 5.0] | 1.0000 | no claim |

## n12k1_knee_d1ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 0.95 | 391.6 |  |  |  |  |  |  |  |
| FX | 20 | 0.95 | 371.6 | 20.0 [-13.9, 52.6] | 0.4986 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| FQ-g75 | 20 | 0.90 | 388.8 | -9.8 [-54.5, 23.7] | 1.0000 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by FQ-g75 only | 1.0000 | no claim |
| FQ-s1 | 20 | 0.90 | 363.7 | 15.4 [-15.5, 47.0] | 1.0000 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by FQ-s1 only | 1.0000 | no claim |

D0's endpoints in n12k1_knee_d1ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 106.0 |  |  |  | 6.58 |  |  |  | 406.5 |  |  |  |
| FX | 96.5 | 9.4 [-0.5, 19.8] | 0.6949 | no claim | 6.67 | -0.10 [-0.28, 0.04] | 1.0000 | no claim | 387.5 | 19.0 [-13.2, 50.2] | 0.5485 | no claim |
| FQ-g75 | 101.1 | 4.8 [-2.0, 11.8] | 0.9307 | no claim | 6.72 | -0.15 [-0.41, 0.04] | 1.0000 | no claim | 416.0 | -9.5 [-48.9, 22.2] | 1.0000 | no claim |
| FQ-s1 | 100.0 | 6.0 [-3.1, 15.1] | 0.9307 | no claim | 6.74 | -0.16 [-0.42, 0.03] | 1.0000 | no claim | 393.1 | 13.4 [-14.8, 41.5] | 1.0000 | no claim |

## Other pairs

Difference = the second arm − the first (time and hover: negative when the second is faster or hovers less; reach and updates: positive when the second reaches τ more often or merges more).

| cell | arm | against | measure | arm | against | against − arm [95% CI] | pairs | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| n6k1_knee_d1base | FQ-s1 | FQ-g75 | time to tau | 165.1 | 157.5 | -7.6 [-16.8, 0.4] | 20 | 0.7474 | no claim |
| n6k1_knee_d1base | FQ-s1 | FQ-g75 | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-s1 | FQ-g75 | hover per mission | 73.3 | 73.3 | -0.0 [-3.0, 3.1] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-s1 | FQ-g75 | updates per mission | 3.54 | 3.54 | 0.00 [-0.04, 0.04] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-s1 | FQ-g75 | time to tau, miss at the end | 165.1 | 157.5 | -7.6 [-16.8, 0.4] | 20 | 0.6726 | no claim |
| n6k1_knee_d1base | FQ-s1 | FX | time to tau | 165.1 | 148.0 | -17.1 [-26.8, -8.8] | 20 | 0.0098 | FX |
| n6k1_knee_d1base | FQ-s1 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-s1 | FX | hover per mission | 73.3 | 67.7 | -5.7 [-9.3, -2.5] | 20 | 0.1580 | no claim |
| n6k1_knee_d1base | FQ-s1 | FX | updates per mission | 3.54 | 3.54 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-s1 | FX | time to tau, miss at the end | 165.1 | 148.0 | -17.1 [-26.8, -8.8] | 20 | 0.0098 | FX |
| n6k1_knee_d1base | FQ-g75 | FX | time to tau | 157.5 | 148.0 | -9.5 [-17.7, -3.0] | 20 | 0.1406 | no claim |
| n6k1_knee_d1base | FQ-g75 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-g75 | FX | hover per mission | 73.3 | 67.7 | -5.7 [-9.6, -2.1] | 20 | 0.1348 | no claim |
| n6k1_knee_d1base | FQ-g75 | FX | updates per mission | 3.54 | 3.54 | 0.00 [-0.04, 0.04] | 20 | 1.0000 | no claim |
| n6k1_knee_d1base | FQ-g75 | FX | time to tau, miss at the end | 157.5 | 148.0 | -9.5 [-17.7, -3.0] | 20 | 0.1406 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | time to tau | 132.3 | 126.8 | -5.5 [-11.7, -1.7] | 19 | 0.1076 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | reach | 0.95 | 1.00 | 0.05 [0.00, 0.15] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | hover per mission | 48.4 | 49.4 | 1.0 [-1.3, 3.5] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | updates per mission | 3.52 | 3.55 | 0.03 [0.00, 0.07] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | time to tau, miss at the end | 144.0 | 127.3 | -16.7 [-41.9, -2.1] | 20 | 0.0709 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FX | time to tau | 132.3 | 124.7 | -7.5 [-15.3, -1.8] | 19 | 0.1076 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FX | reach | 0.95 | 1.00 | 0.05 [0.00, 0.15] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FX | hover per mission | 48.4 | 46.6 | -1.9 [-4.1, -0.1] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FX | updates per mission | 3.52 | 3.55 | 0.03 [0.00, 0.07] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-s1 | FX | time to tau, miss at the end | 144.0 | 125.4 | -18.7 [-45.0, -2.7] | 20 | 0.0709 | no claim |
| n6k1_knee_d1ampharsh | FQ-g75 | FX | time to tau | 127.3 | 125.4 | -1.9 [-5.8, 0.0] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-g75 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-g75 | FX | hover per mission | 49.4 | 46.6 | -2.9 [-6.2, -0.4] | 20 | 0.1580 | no claim |
| n6k1_knee_d1ampharsh | FQ-g75 | FX | updates per mission | 3.55 | 3.55 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharsh | FQ-g75 | FX | time to tau, miss at the end | 127.3 | 125.4 | -1.9 [-5.8, 0.0] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FQ-g75 | time to tau | 131.3 | 143.6 | 12.4 [-1.3, 29.4] | 20 | 0.7474 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FQ-g75 | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FQ-g75 | hover per mission | 53.9 | 55.5 | 1.6 [-1.4, 4.5] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FQ-g75 | updates per mission | 3.55 | 3.54 | -0.01 [-0.04, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FQ-g75 | time to tau, miss at the end | 131.3 | 143.6 | 12.4 [-1.3, 29.4] | 20 | 0.6726 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FX | time to tau | 131.3 | 144.1 | 12.9 [-0.0, 30.3] | 20 | 0.9307 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FX | hover per mission | 53.9 | 54.6 | 0.7 [-0.7, 2.6] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FX | updates per mission | 3.55 | 3.54 | -0.01 [-0.04, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-s1 | FX | time to tau, miss at the end | 131.3 | 144.1 | 12.9 [-0.0, 30.3] | 20 | 0.8110 | no claim |
| n6k1_knee_d1ampharshpc | FQ-g75 | FX | time to tau | 143.6 | 144.1 | 0.5 [-12.2, 14.3] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-g75 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-g75 | FX | hover per mission | 55.5 | 54.6 | -0.9 [-4.1, 2.6] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-g75 | FX | updates per mission | 3.54 | 3.54 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1ampharshpc | FQ-g75 | FX | time to tau, miss at the end | 143.6 | 144.1 | 0.5 [-12.2, 14.3] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FQ-g75 | time to tau | 230.6 | 226.7 | -3.9 [-18.0, 10.3] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FQ-g75 | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FQ-g75 | hover per mission | 105.7 | 107.4 | 1.7 [-5.4, 9.5] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FQ-g75 | updates per mission | 3.24 | 3.33 | 0.09 [-0.12, 0.31] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FQ-g75 | time to tau, miss at the end | 230.6 | 226.7 | -3.9 [-18.0, 10.3] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FX | time to tau | 230.6 | 217.1 | -13.5 [-31.5, 4.4] | 20 | 0.9307 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FX | hover per mission | 105.7 | 96.7 | -9.0 [-19.0, 0.5] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FX | updates per mission | 3.24 | 3.27 | 0.04 [-0.17, 0.28] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-s1 | FX | time to tau, miss at the end | 230.6 | 217.1 | -13.5 [-31.5, 4.4] | 20 | 0.8110 | no claim |
| n6k1_knee_d1burst40 | FQ-g75 | FX | time to tau | 226.7 | 217.1 | -9.6 [-21.6, -1.0] | 20 | 0.4743 | no claim |
| n6k1_knee_d1burst40 | FQ-g75 | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-g75 | FX | hover per mission | 107.4 | 96.7 | -10.8 [-19.4, -2.4] | 20 | 0.3943 | no claim |
| n6k1_knee_d1burst40 | FQ-g75 | FX | updates per mission | 3.33 | 3.27 | -0.05 [-0.14, 0.00] | 20 | 1.0000 | no claim |
| n6k1_knee_d1burst40 | FQ-g75 | FX | time to tau, miss at the end | 226.7 | 217.1 | -9.6 [-21.6, -1.0] | 20 | 0.4743 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | time to tau | 363.7 | 388.8 | 25.2 [-12.4, 74.0] | 18 | 0.9307 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | reach | 0.90 | 0.90 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | hover per mission | 100.0 | 101.1 | 1.2 [-3.8, 5.7] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | updates per mission | 6.74 | 6.72 | -0.01 [-0.05, 0.03] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FQ-g75 | time to tau, miss at the end | 393.1 | 416.0 | 22.9 [-11.3, 66.5] | 20 | 0.8110 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FX | time to tau | 363.7 | 358.1 | -5.6 [-25.9, 13.4] | 18 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FX | reach | 0.90 | 0.95 | 0.05 [0.00, 0.15] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FX | hover per mission | 100.0 | 96.5 | -3.4 [-9.7, 2.3] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FX | updates per mission | 6.74 | 6.67 | -0.06 [-0.16, 0.01] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-s1 | FX | time to tau, miss at the end | 393.1 | 387.5 | -5.6 [-25.1, 12.2] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-g75 | FX | time to tau | 388.8 | 358.1 | -30.7 [-78.9, 2.6] | 18 | 0.7474 | no claim |
| n12k1_knee_d1ampharsh | FQ-g75 | FX | reach | 0.90 | 0.95 | 0.05 [0.00, 0.15] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-g75 | FX | hover per mission | 101.1 | 96.5 | -4.6 [-10.0, 0.7] | 20 | 0.8095 | no claim |
| n12k1_knee_d1ampharsh | FQ-g75 | FX | updates per mission | 6.72 | 6.67 | -0.05 [-0.15, 0.03] | 20 | 1.0000 | no claim |
| n12k1_knee_d1ampharsh | FQ-g75 | FX | time to tau, miss at the end | 416.0 | 387.5 | -28.5 [-69.2, 2.7] | 20 | 0.6403 | no claim |

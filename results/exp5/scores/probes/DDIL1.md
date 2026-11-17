# DDIL1: Step 1: the honest DDIL channel

Exploratory, not pre-registered (DeveloperDocs/Experiment_5_Loss_Probes_Plan.md). Reference F. Time to tau = 0.71 over the trials both arms reach it, and reach at tau (exact McNemar on the paired outcomes), each with a 95% bootstrap CI. Holm per family: the baselines on time, on reach, and FX on each apart. Difference = F - arm: time, negative when F is faster; reach, positive when F reaches tau more often.

DDIL build D0's endpoints (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md), paired by trial and seed, each in a Holm family of its own: hover per mission (sim_dwell_s_mean, lower is better), updates merged per mission (update_yield, higher), and time to tau with a trial that missed it given the trial's end, so no trial is dropped (lower).

Flown and scored: 140 of 140 arm-cells.

## Where a baseline beats F

On time to τ, reach, updates per mission, or time with misses at the end. A baseline's lower hover is listed in the cells' tables, not here: it comes from leaving the members below the floor unserved.

- none (so far)

## main: n6k1_knee_d1base

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 166.6 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 148.0 | 18.6 [8.8, 30.0] | 0.0167 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 286.3 | -119.6 [-180.0, -70.7] | 0.0009 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 276.9 | -110.3 [-159.0, -70.0] | 0.0009 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 265.7 | -99.1 [-145.4, -64.7] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 290.1 | -123.5 [-185.2, -74.8] | 0.0009 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 246.1 | -79.5 [-116.4, -50.9] | 0.0009 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 266.7 | -100.1 [-148.9, -59.7] | 0.0009 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1base, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 76.0 |  |  |  | 3.55 |  |  |  | 166.6 |  |  |  |
| FX | 67.7 | 8.4 [4.6, 12.7] | 0.0197 | FX | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 148.0 | 18.6 [8.8, 30.0] | 0.0167 | FX |
| H1 | 11.9 | 64.2 [55.0, 73.0] | 0.0002 | H1 | 3.40 | 0.15 [0.04, 0.31] | 0.4709 | no claim | 286.3 | -119.6 [-180.0, -70.7] | 0.0009 | F |
| MAX-AoI | 11.6 | 64.5 [55.4, 73.3] | 0.0002 | MAX-AoI | 3.42 | 0.12 [0.04, 0.25] | 0.4709 | no claim | 276.9 | -110.3 [-159.0, -70.0] | 0.0009 | F |
| Oort | 11.6 | 64.5 [54.9, 73.6] | 0.0002 | Oort | 3.48 | 0.07 [0.00, 0.16] | 1.0000 | no claim | 265.7 | -99.1 [-145.4, -64.7] | 0.0006 | F |
| Whittle | 10.8 | 65.3 [55.9, 74.2] | 0.0002 | Whittle | 3.40 | 0.15 [0.04, 0.29] | 0.4709 | no claim | 290.1 | -123.5 [-185.2, -74.8] | 0.0009 | F |
| FedEx route | 11.3 | 64.7 [55.6, 73.2] | 0.0002 | FedEx route | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 246.1 | -79.5 [-116.4, -50.9] | 0.0009 | F |
| FedCS | 11.4 | 64.7 [55.3, 73.6] | 0.0002 | FedCS | 3.52 | 0.03 [0.00, 0.06] | 1.0000 | no claim | 266.7 | -100.1 [-148.9, -59.7] | 0.0009 | F |

## main: n6k1_knee_d1amp11

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 152.1 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 136.4 | 15.6 [7.2, 25.2] | 0.0226 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 272.8 | -120.7 [-179.3, -69.0] | 0.0086 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 257.7 | -105.6 [-157.1, -60.6] | 0.0108 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 246.1 | -94.0 [-149.1, -30.1] | 0.1608 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 262.6 | -110.6 [-166.2, -62.7] | 0.0226 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 261.3 | -109.2 [-158.0, -70.9] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 272.8 | -120.7 [-183.5, -64.3] | 0.0193 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1amp11, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 65.6 |  |  |  | 3.55 |  |  |  | 152.1 |  |  |  |
| FX | 59.1 | 6.6 [3.1, 10.6] | 0.0249 | FX | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 136.4 | 15.6 [7.2, 25.2] | 0.0226 | FX |
| H1 | 15.9 | 49.7 [43.5, 55.8] | 0.0002 | H1 | 3.29 | 0.26 [0.14, 0.41] | 0.0683 | no claim | 272.8 | -120.7 [-179.3, -69.0] | 0.0086 | F |
| MAX-AoI | 16.3 | 49.3 [42.8, 55.5] | 0.0002 | MAX-AoI | 3.34 | 0.21 [0.11, 0.33] | 0.1019 | no claim | 257.7 | -105.6 [-157.1, -60.6] | 0.0108 | F |
| Oort | 16.2 | 49.5 [42.8, 55.5] | 0.0002 | Oort | 3.29 | 0.26 [0.14, 0.42] | 0.0431 | F | 246.1 | -94.0 [-149.1, -30.1] | 0.1576 | no claim |
| Whittle | 15.0 | 50.6 [44.7, 56.2] | 0.0002 | Whittle | 3.35 | 0.20 [0.11, 0.30] | 0.1019 | no claim | 262.6 | -110.6 [-166.2, -62.7] | 0.0226 | F |
| FedEx route | 15.4 | 50.2 [43.7, 56.1] | 0.0002 | FedEx route | 3.39 | 0.16 [0.09, 0.25] | 0.1229 | no claim | 261.3 | -109.2 [-158.0, -70.9] | 0.0006 | F |
| FedCS | 14.4 | 51.2 [44.0, 57.7] | 0.0002 | FedCS | 3.30 | 0.25 [0.15, 0.35] | 0.0325 | F | 272.8 | -120.7 [-183.5, -64.3] | 0.0193 | F |

## main: n6k1_knee_d1ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.9 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 125.4 | 25.6 [9.9, 46.8] | 0.0221 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 277.2 | -126.2 [-183.6, -69.5] | 0.0193 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 287.7 | -136.8 [-204.3, -72.4] | 0.0226 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 0.95 | 246.3 | -98.4 [-143.1, -41.5] | 0.0714 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 0.95 | 263.4 | -115.5 [-173.9, -70.3] | 0.0045 | F | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 215.9 | -65.0 [-111.3, -10.7] | 0.3963 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 260.0 | -109.0 [-163.7, -56.4] | 0.0386 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 59.3 |  |  |  | 3.54 |  |  |  | 150.9 |  |  |  |
| FX | 46.6 | 12.8 [8.5, 17.3] | 0.0074 | FX | 3.55 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 125.4 | 25.6 [9.9, 46.8] | 0.0221 | FX |
| H1 | 15.0 | 44.3 [40.4, 48.3] | 0.0002 | H1 | 2.73 | 0.81 [0.61, 1.01] | 0.0108 | F | 277.2 | -126.2 [-183.6, -69.5] | 0.0193 | F |
| MAX-AoI | 15.5 | 43.9 [39.7, 47.5] | 0.0002 | MAX-AoI | 2.79 | 0.75 [0.57, 0.94] | 0.0082 | F | 287.7 | -136.8 [-204.3, -72.4] | 0.0226 | F |
| Oort | 14.4 | 45.0 [39.9, 49.8] | 0.0002 | Oort | 2.86 | 0.68 [0.54, 0.82] | 0.0078 | F | 264.8 | -113.9 [-166.2, -56.5] | 0.0381 | F |
| Whittle | 17.7 | 41.7 [36.0, 47.0] | 0.0002 | Whittle | 2.94 | 0.60 [0.42, 0.78] | 0.0147 | F | 281.0 | -130.1 [-195.1, -77.3] | 0.0023 | F |
| FedEx route | 16.5 | 42.8 [37.5, 47.8] | 0.0002 | FedEx route | 2.94 | 0.60 [0.40, 0.82] | 0.0108 | F | 215.9 | -65.0 [-111.3, -10.7] | 0.3869 | no claim |
| FedCS | 14.8 | 44.5 [39.3, 49.4] | 0.0002 | FedCS | 2.94 | 0.60 [0.39, 0.80] | 0.0144 | F | 260.0 | -109.0 [-163.7, -56.4] | 0.0381 | F |

## main: n6k1_knee_d1npl

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 184.2 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 173.2 | 11.1 [2.9, 21.7] | 0.0456 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 273.0 | -88.8 [-124.0, -58.9] | 0.0124 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 272.4 | -88.2 [-122.6, -58.5] | 0.0124 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 262.3 | -78.1 [-107.7, -52.4] | 0.0124 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.7 | -80.5 [-109.4, -54.2] | 0.0124 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 242.9 | -58.6 [-80.3, -38.6] | 0.0124 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 262.5 | -78.3 [-113.6, -48.4] | 0.0124 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1npl, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 69.8 |  |  |  | 3.55 |  |  |  | 184.2 |  |  |  |
| FX | 65.9 | 3.9 [0.8, 8.1] | 0.1556 | no claim | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 173.2 | 11.1 [2.9, 21.7] | 0.0456 | FX |
| H1 | 8.9 | 60.9 [50.9, 70.1] | 0.0002 | H1 | 3.46 | 0.09 [0.00, 0.23] | 1.0000 | no claim | 273.0 | -88.8 [-124.0, -58.9] | 0.0124 | F |
| MAX-AoI | 8.9 | 60.9 [50.9, 70.1] | 0.0002 | MAX-AoI | 3.46 | 0.09 [0.00, 0.20] | 1.0000 | no claim | 272.4 | -88.2 [-122.6, -58.5] | 0.0124 | F |
| Oort | 9.1 | 60.7 [50.7, 69.9] | 0.0002 | Oort | 3.49 | 0.06 [0.00, 0.15] | 1.0000 | no claim | 262.3 | -78.1 [-107.7, -52.4] | 0.0124 | F |
| Whittle | 9.2 | 60.7 [50.5, 69.9] | 0.0002 | Whittle | 3.48 | 0.07 [0.01, 0.15] | 0.7598 | no claim | 264.7 | -80.5 [-109.4, -54.2] | 0.0124 | F |
| FedEx route | 9.1 | 60.8 [50.7, 69.9] | 0.0002 | FedEx route | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 242.9 | -58.6 [-80.3, -38.6] | 0.0124 | F |
| FedCS | 8.7 | 61.1 [51.2, 70.3] | 0.0002 | FedCS | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 262.5 | -78.3 [-113.6, -48.4] | 0.0124 | F |

## main: n6k1_knee_d1sigma8

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 133.6 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 114.4 | 19.1 [12.2, 26.5] | 0.0124 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 258.1 | -124.6 [-174.7, -85.8] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 257.8 | -124.3 [-174.5, -85.1] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 223.3 | -89.7 [-125.3, -42.3] | 0.0553 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.9 | -131.3 [-184.5, -91.1] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 251.0 | -117.4 [-165.2, -81.2] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 262.8 | -129.3 [-184.7, -84.3] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1sigma8, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 58.1 |  |  |  | 3.55 |  |  |  | 133.6 |  |  |  |
| FX | 48.1 | 10.0 [6.3, 13.7] | 0.0096 | FX | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 114.4 | 19.1 [12.2, 26.5] | 0.0124 | FX |
| H1 | 8.3 | 49.8 [44.2, 55.0] | 0.0002 | H1 | 3.45 | 0.10 [0.03, 0.18] | 0.5765 | no claim | 258.1 | -124.6 [-174.7, -85.8] | 0.0004 | F |
| MAX-AoI | 8.3 | 49.8 [44.1, 55.0] | 0.0002 | MAX-AoI | 3.41 | 0.14 [0.05, 0.24] | 0.3468 | no claim | 257.8 | -124.3 [-174.5, -85.1] | 0.0004 | F |
| Oort | 8.1 | 50.0 [44.8, 54.8] | 0.0002 | Oort | 3.39 | 0.16 [0.06, 0.26] | 0.2413 | no claim | 223.3 | -89.7 [-125.3, -42.3] | 0.0536 | no claim |
| Whittle | 7.8 | 50.3 [44.7, 55.1] | 0.0002 | Whittle | 3.39 | 0.16 [0.09, 0.24] | 0.1229 | no claim | 264.9 | -131.3 [-184.5, -91.1] | 0.0004 | F |
| FedEx route | 7.6 | 50.5 [45.3, 55.3] | 0.0002 | FedEx route | 3.48 | 0.07 [0.03, 0.12] | 0.3290 | no claim | 251.0 | -117.4 [-165.2, -81.2] | 0.0004 | F |
| FedCS | 8.6 | 49.5 [43.8, 54.8] | 0.0002 | FedCS | 3.50 | 0.05 [0.01, 0.10] | 0.6370 | no claim | 262.8 | -129.3 [-184.7, -84.3] | 0.0004 | F |

## main: n6k1_knee_d1burst10

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 170.3 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 158.3 | 12.0 [5.4, 19.6] | 0.0221 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 307.4 | -137.0 [-200.0, -84.3] | 0.0012 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 298.3 | -127.9 [-179.6, -81.7] | 0.0012 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 274.3 | -104.0 [-136.1, -73.8] | 0.0012 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 316.7 | -146.3 [-211.2, -89.6] | 0.0017 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 258.4 | -88.0 [-130.3, -52.8] | 0.0017 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 286.2 | -115.9 [-171.4, -68.8] | 0.0023 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1burst10, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 79.3 |  |  |  | 3.54 |  |  |  | 170.3 |  |  |  |
| FX | 71.0 | 8.3 [3.2, 14.1] | 0.2109 | no claim | 3.48 | 0.06 [0.00, 0.15] | 1.0000 | no claim | 158.3 | 12.0 [5.4, 19.6] | 0.0221 | FX |
| H1 | 12.4 | 66.9 [57.3, 75.7] | 0.0002 | H1 | 3.25 | 0.29 [0.14, 0.46] | 0.1130 | no claim | 307.4 | -137.0 [-200.0, -84.3] | 0.0012 | F |
| MAX-AoI | 12.0 | 67.3 [57.6, 76.3] | 0.0002 | MAX-AoI | 3.26 | 0.28 [0.14, 0.45] | 0.1130 | no claim | 298.3 | -127.9 [-179.6, -81.7] | 0.0012 | F |
| Oort | 11.4 | 67.9 [58.3, 76.6] | 0.0002 | Oort | 3.23 | 0.31 [0.16, 0.49] | 0.0792 | no claim | 274.3 | -104.0 [-136.1, -73.8] | 0.0012 | F |
| Whittle | 12.0 | 67.3 [57.4, 76.3] | 0.0002 | Whittle | 3.19 | 0.35 [0.16, 0.56] | 0.1130 | no claim | 316.7 | -146.3 [-211.2, -89.6] | 0.0017 | F |
| FedEx route | 12.1 | 67.2 [57.8, 75.8] | 0.0002 | FedEx route | 3.27 | 0.26 [0.11, 0.45] | 0.1982 | no claim | 258.4 | -88.0 [-130.3, -52.8] | 0.0017 | F |
| FedCS | 11.7 | 67.6 [58.0, 76.3] | 0.0002 | FedCS | 3.35 | 0.19 [0.10, 0.28] | 0.1317 | no claim | 286.2 | -115.9 [-171.4, -68.8] | 0.0023 | F |

## main: n6k1_knee_d1burst40

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 229.7 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 217.1 | 12.6 [-5.4, 30.2] | 0.5470 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 265.4 | -35.7 [-103.2, 44.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 0.95 | 256.9 | -36.1 [-108.6, 49.6] | 1.0000 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 271.3 | -41.6 [-123.4, 47.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 225.7 | 4.0 [-56.6, 77.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 269.2 | -39.5 [-92.9, 15.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 283.7 | -54.0 [-120.6, 27.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1burst40, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 110.0 |  |  |  | 3.24 |  |  |  | 229.7 |  |  |  |
| FX | 96.7 | 13.4 [4.5, 22.7] | 0.1346 | no claim | 3.27 | -0.04 [-0.28, 0.19] | 1.0000 | no claim | 217.1 | 12.6 [-5.4, 30.2] | 0.5470 | no claim |
| H1 | 12.7 | 97.3 [86.5, 107.9] | 0.0002 | H1 | 2.48 | 0.76 [0.56, 0.99] | 0.0147 | F | 265.4 | -35.7 [-103.2, 44.8] | 1.0000 | no claim |
| MAX-AoI | 12.7 | 97.4 [87.1, 108.2] | 0.0002 | MAX-AoI | 2.45 | 0.79 [0.55, 1.05] | 0.0147 | F | 269.1 | -39.4 [-108.3, 41.4] | 1.0000 | no claim |
| Oort | 12.3 | 97.7 [87.2, 107.7] | 0.0002 | Oort | 2.42 | 0.81 [0.59, 1.05] | 0.0147 | F | 271.3 | -41.6 [-123.4, 47.7] | 1.0000 | no claim |
| Whittle | 12.5 | 97.5 [87.1, 107.8] | 0.0002 | Whittle | 2.65 | 0.59 [0.34, 0.84] | 0.0553 | no claim | 225.7 | 4.0 [-56.6, 77.1] | 1.0000 | no claim |
| FedEx route | 12.0 | 98.0 [87.5, 109.2] | 0.0002 | FedEx route | 2.19 | 1.05 [0.68, 1.49] | 0.0227 | F | 269.2 | -39.5 [-92.9, 15.0] | 1.0000 | no claim |
| FedCS | 12.6 | 97.4 [85.4, 108.8] | 0.0002 | FedCS | 2.34 | 0.90 [0.61, 1.21] | 0.0137 | F | 283.7 | -54.0 [-120.6, 27.0] | 1.0000 | no claim |

## main: n6k1_knee_d1ampharshpc

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.3 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 144.1 | 6.2 [-6.2, 17.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 284.3 | -133.9 [-203.4, -63.9] | 0.0963 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 286.2 | -135.9 [-206.6, -64.9] | 0.0963 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 246.4 | -96.0 [-147.0, -48.6] | 0.0553 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.2 | -113.8 [-184.0, -43.8] | 0.1608 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 259.8 | -109.5 [-174.0, -49.1] | 0.1232 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 257.3 | -106.9 [-171.8, -40.0] | 0.2236 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1ampharshpc, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 58.3 |  |  |  | 3.54 |  |  |  | 150.3 |  |  |  |
| FX | 54.6 | 3.7 [-1.3, 9.3] | 0.5611 | no claim | 3.54 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 144.1 | 6.2 [-6.2, 17.7] | 1.0000 | no claim |
| H1 | 16.1 | 42.2 [36.8, 47.2] | 0.0002 | H1 | 2.83 | 0.71 [0.53, 0.90] | 0.0110 | F | 284.3 | -133.9 [-203.4, -63.9] | 0.0963 | no claim |
| MAX-AoI | 16.0 | 42.3 [36.8, 47.4] | 0.0002 | MAX-AoI | 2.79 | 0.75 [0.59, 0.91] | 0.0110 | F | 286.2 | -135.9 [-206.6, -64.9] | 0.0963 | no claim |
| Oort | 15.3 | 43.0 [37.8, 48.0] | 0.0002 | Oort | 2.83 | 0.71 [0.51, 0.91] | 0.0147 | F | 246.4 | -96.0 [-147.0, -48.6] | 0.0536 | no claim |
| Whittle | 15.8 | 42.5 [36.9, 48.0] | 0.0002 | Whittle | 2.85 | 0.69 [0.54, 0.84] | 0.0082 | F | 264.2 | -113.8 [-184.0, -43.8] | 0.1576 | no claim |
| FedEx route | 16.3 | 42.0 [36.3, 47.9] | 0.0002 | FedEx route | 3.04 | 0.50 [0.33, 0.68] | 0.0168 | F | 259.8 | -109.5 [-174.0, -49.1] | 0.1232 | no claim |
| FedCS | 17.3 | 41.0 [35.6, 46.1] | 0.0002 | FedCS | 2.90 | 0.64 [0.44, 0.82] | 0.0227 | F | 257.3 | -106.9 [-171.8, -40.0] | 0.2138 | no claim |

## main: n6k1_knee_d1baseo

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 166.6 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 147.2 | 19.4 [9.3, 30.8] | 0.0167 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1baseo, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 77.7 |  |  |  | 3.55 |  |  |  | 166.6 |  |  |  |
| FX | 67.5 | 10.3 [6.9, 14.0] | 0.0053 | FX | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 147.2 | 19.4 [9.3, 30.8] | 0.0167 | FX |

## main: n6k1_knee_d1ampharsho

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 151.4 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 124.5 | 26.9 [11.4, 47.4] | 0.0124 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |

D0's endpoints in main: n6k1_knee_d1ampharsho, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 59.9 |  |  |  | 3.54 |  |  |  | 151.4 |  |  |  |
| FX | 47.3 | 12.5 [8.1, 17.3] | 0.0074 | FX | 3.54 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 124.5 | 26.9 [11.4, 47.4] | 0.0124 | FX |

## main: n6k1_stress_d1ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 171.5 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 152.0 | 19.6 [5.9, 40.1] | 0.0226 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 0.85 | 248.7 | -80.7 [-141.0, -11.6] | 0.6000 | no claim | 0.15 [0.00, 0.30] | 3 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 293.7 | -122.2 [-204.5, -37.9] | 0.3963 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 0.90 | 212.3 | -32.7 [-84.4, 25.5] | 1.0000 | no claim | 0.10 [0.00, 0.25] | 2 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 241.6 | -70.0 [-155.6, 13.6] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 215.9 | -44.4 [-100.5, 20.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 0.95 | 247.7 | -87.0 [-154.5, -11.4] | 0.9836 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in main: n6k1_stress_d1ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 51.6 |  |  |  | 3.09 |  |  |  | 171.5 |  |  |  |
| FX | 41.1 | 10.5 [6.2, 15.2] | 0.0044 | FX | 3.08 | 0.01 [-0.07, 0.11] | 1.0000 | no claim | 152.0 | 19.6 [5.9, 40.1] | 0.0226 | FX |
| H1 | 10.6 | 41.0 [34.9, 47.1] | 0.0002 | H1 | 1.69 | 1.40 [1.07, 1.75] | 0.0082 | F | 275.7 | -104.1 [-166.1, -39.5] | 0.1576 | no claim |
| MAX-AoI | 13.2 | 38.3 [31.2, 45.0] | 0.0002 | MAX-AoI | 1.94 | 1.15 [0.75, 1.59] | 0.0150 | F | 293.7 | -122.2 [-204.5, -37.9] | 0.3869 | no claim |
| Oort | 10.6 | 41.0 [34.2, 47.5] | 0.0002 | Oort | 1.75 | 1.34 [0.96, 1.76] | 0.0109 | F | 232.6 | -61.0 [-125.4, 3.3] | 0.9464 | no claim |
| Whittle | 12.8 | 38.7 [32.1, 44.8] | 0.0002 | Whittle | 1.89 | 1.20 [0.88, 1.55] | 0.0111 | F | 241.6 | -70.0 [-155.6, 13.6] | 1.0000 | no claim |
| FedEx route | 16.5 | 35.0 [28.3, 41.0] | 0.0002 | FedEx route | 2.94 | 0.15 [-0.15, 0.45] | 1.0000 | no claim | 215.9 | -44.4 [-100.5, 20.0] | 1.0000 | no claim |
| FedCS | 13.3 | 38.2 [31.4, 44.5] | 0.0002 | FedCS | 2.08 | 1.01 [0.74, 1.30] | 0.0150 | F | 258.6 | -87.1 [-151.7, -21.4] | 0.7734 | no claim |

## checks: n6k1_knee_a1sess10

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 151.1 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 144.0 | 7.1 [-8.8, 23.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 294.7 | -143.6 [-217.8, -66.4] | 0.0859 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 297.7 | -146.6 [-221.9, -67.9] | 0.0859 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 247.0 | -95.8 [-147.4, -48.7] | 0.0553 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 261.8 | -110.7 [-183.3, -38.9] | 0.1754 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 259.3 | -108.2 [-172.3, -48.4] | 0.1232 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 266.3 | -115.2 [-182.0, -43.1] | 0.1984 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_a1sess10, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 58.8 |  |  |  | 3.38 |  |  |  | 151.1 |  |  |  |
| FX | 57.4 | 1.3 [-4.0, 7.5] | 0.8361 | no claim | 3.51 | -0.14 [-0.25, -0.04] | 0.4863 | no claim | 144.0 | 7.1 [-8.8, 23.8] | 1.0000 | no claim |
| H1 | 17.1 | 41.6 [36.3, 46.6] | 0.0002 | H1 | 2.77 | 0.60 [0.39, 0.84] | 0.0142 | F | 294.7 | -143.6 [-217.8, -66.4] | 0.0859 | no claim |
| MAX-AoI | 17.8 | 40.9 [35.5, 46.2] | 0.0002 | MAX-AoI | 2.71 | 0.66 [0.45, 0.89] | 0.0135 | F | 297.7 | -146.6 [-221.9, -67.9] | 0.0859 | no claim |
| Oort | 15.5 | 43.2 [39.0, 47.4] | 0.0002 | Oort | 2.76 | 0.61 [0.36, 0.89] | 0.0325 | F | 247.0 | -95.8 [-147.4, -48.7] | 0.0536 | no claim |
| Whittle | 15.4 | 43.3 [37.8, 48.9] | 0.0002 | Whittle | 2.66 | 0.71 [0.49, 0.95] | 0.0178 | F | 261.8 | -110.7 [-183.3, -38.9] | 0.1681 | no claim |
| FedEx route | 15.8 | 42.9 [36.3, 49.5] | 0.0002 | FedEx route | 2.95 | 0.42 [0.19, 0.68] | 0.1653 | no claim | 259.3 | -108.2 [-172.3, -48.4] | 0.1232 | no claim |
| FedCS | 16.0 | 42.8 [37.6, 47.7] | 0.0002 | FedCS | 2.81 | 0.56 [0.33, 0.79] | 0.0540 | no claim | 266.3 | -115.2 [-182.0, -43.1] | 0.1899 | no claim |

## checks: n6k1_knee_a1sess2

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 166.4 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 150.7 | 15.6 [2.5, 29.0] | 0.2373 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 260.4 | -94.1 [-165.5, -19.8] | 0.9341 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 261.6 | -95.3 [-168.5, -19.8] | 0.9341 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 227.4 | -61.1 [-130.7, 8.9] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 261.2 | -94.8 [-170.4, -17.8] | 0.9554 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 228.1 | -61.7 [-125.0, -1.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 239.5 | -73.2 [-135.0, -4.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_a1sess2, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 59.4 |  |  |  | 3.35 |  |  |  | 166.4 |  |  |  |
| FX | 55.8 | 3.6 [-0.8, 7.9] | 0.3848 | no claim | 3.40 | -0.05 [-0.16, 0.05] | 1.0000 | no claim | 150.7 | 15.6 [2.5, 29.0] | 0.2373 | no claim |
| H1 | 11.4 | 48.0 [43.1, 52.6] | 0.0002 | H1 | 2.52 | 0.82 [0.59, 1.05] | 0.0180 | F | 260.4 | -94.1 [-165.5, -19.8] | 0.8383 | no claim |
| MAX-AoI | 10.5 | 48.9 [44.0, 53.5] | 0.0002 | MAX-AoI | 2.51 | 0.84 [0.59, 1.07] | 0.0227 | F | 261.6 | -95.3 [-168.5, -19.8] | 0.8383 | no claim |
| Oort | 11.6 | 47.8 [42.5, 52.3] | 0.0002 | Oort | 2.40 | 0.95 [0.71, 1.19] | 0.0147 | F | 227.4 | -61.1 [-130.7, 8.9] | 1.0000 | no claim |
| Whittle | 10.5 | 48.9 [44.0, 53.5] | 0.0002 | Whittle | 2.42 | 0.93 [0.62, 1.23] | 0.0180 | F | 261.2 | -94.8 [-170.4, -17.8] | 0.8792 | no claim |
| FedEx route | 9.8 | 49.7 [44.4, 54.5] | 0.0002 | FedEx route | 2.60 | 0.75 [0.47, 1.00] | 0.0248 | F | 228.1 | -61.7 [-125.0, -1.0] | 1.0000 | no claim |
| FedCS | 10.9 | 48.5 [43.2, 53.5] | 0.0002 | FedCS | 2.49 | 0.86 [0.64, 1.07] | 0.0147 | F | 239.5 | -73.2 [-135.0, -4.7] | 1.0000 | no claim |

## checks: n6k1_knee_a1repoll

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 264.7 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 215.5 | 49.3 [11.0, 93.8] | 0.2289 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 264.8 | -0.1 [-83.1, 87.6] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 274.2 | -9.4 [-96.4, 81.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 268.8 | -4.1 [-84.6, 73.2] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 277.0 | -12.3 [-99.2, 82.5] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 244.6 | 20.2 [-42.8, 90.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 260.3 | 4.4 [-68.0, 76.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_a1repoll, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 145.7 |  |  |  | 3.51 |  |  |  | 264.7 |  |  |  |
| FX | 116.6 | 29.1 [7.4, 53.3] | 0.2109 | no claim | 3.51 | 0.00 [-0.10, 0.07] | 1.0000 | no claim | 215.5 | 49.3 [11.0, 93.8] | 0.2289 | no claim |
| H1 | 9.5 | 136.1 [112.7, 161.3] | 0.0002 | H1 | 2.70 | 0.81 [0.57, 1.05] | 0.0178 | F | 264.8 | -0.1 [-83.1, 87.6] | 1.0000 | no claim |
| MAX-AoI | 9.2 | 136.5 [112.4, 162.1] | 0.0002 | MAX-AoI | 2.77 | 0.74 [0.51, 0.97] | 0.0147 | F | 274.2 | -9.4 [-96.4, 81.3] | 1.0000 | no claim |
| Oort | 10.2 | 135.5 [111.2, 161.5] | 0.0002 | Oort | 2.83 | 0.69 [0.46, 0.91] | 0.0180 | F | 268.8 | -4.1 [-84.6, 73.2] | 1.0000 | no claim |
| Whittle | 9.3 | 136.4 [112.6, 161.9] | 0.0002 | Whittle | 2.76 | 0.75 [0.51, 0.99] | 0.0180 | F | 277.0 | -12.3 [-99.2, 82.5] | 1.0000 | no claim |
| FedEx route | 10.5 | 135.2 [111.0, 161.0] | 0.0002 | FedEx route | 3.10 | 0.41 [0.23, 0.60] | 0.0756 | no claim | 244.6 | 20.2 [-42.8, 90.3] | 1.0000 | no claim |
| FedCS | 10.4 | 135.2 [111.5, 161.2] | 0.0002 | FedCS | 2.92 | 0.59 [0.44, 0.71] | 0.0173 | F | 260.3 | 4.4 [-68.0, 76.8] | 1.0000 | no claim |

## checks: n6k1_knee_a1burst40

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 0.95 | 377.1 |  |  |  |  |  |  |  |
| FX | 20 | 0.95 | 309.1 | 68.0 [23.2, 122.0] | 0.1383 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 274.6 | 93.6 [2.3, 205.5] | 1.0000 | no claim | -0.05 [-0.15, 0.00] | 0 reached by F only, 1 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 276.4 | 91.8 [-0.5, 204.0] | 1.0000 | no claim | -0.05 [-0.15, 0.00] | 0 reached by F only, 1 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 248.5 | 122.3 [39.6, 207.5] | 0.9554 | no claim | -0.05 [-0.15, 0.00] | 0 reached by F only, 1 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 262.5 | 106.8 [27.5, 191.3] | 0.9836 | no claim | -0.05 [-0.15, 0.00] | 0 reached by F only, 1 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 256.4 | 137.0 [64.2, 209.4] | 0.2398 | no claim | -0.05 [-0.15, 0.00] | 0 reached by F only, 1 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 243.1 | 126.8 [34.7, 233.2] | 0.9554 | no claim | -0.05 [-0.15, 0.00] | 0 reached by F only, 1 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_a1burst40, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 216.9 |  |  |  | 2.69 |  |  |  | 397.7 |  |  |  |
| FX | 157.8 | 59.0 [29.7, 89.1] | 0.0249 | FX | 2.80 | -0.11 [-0.36, 0.09] | 1.0000 | no claim | 326.6 | 71.1 [28.2, 118.5] | 0.0869 | no claim |
| H1 | 9.6 | 207.3 [171.3, 240.6] | 0.0002 | H1 | 2.33 | 0.36 [0.07, 0.64] | 0.4482 | no claim | 274.6 | 123.1 [20.8, 233.8] | 1.0000 | no claim |
| MAX-AoI | 9.6 | 207.3 [171.5, 240.5] | 0.0002 | MAX-AoI | 2.34 | 0.35 [0.05, 0.64] | 0.5658 | no claim | 276.4 | 121.3 [18.9, 232.7] | 1.0000 | no claim |
| Oort | 9.9 | 206.9 [170.5, 240.8] | 0.0002 | Oort | 2.25 | 0.44 [0.10, 0.78] | 0.5658 | no claim | 248.5 | 149.1 [58.6, 240.5] | 0.5310 | no claim |
| Whittle | 8.7 | 208.1 [171.9, 241.7] | 0.0002 | Whittle | 2.29 | 0.40 [0.06, 0.76] | 0.8357 | no claim | 262.5 | 135.2 [45.2, 230.2] | 0.5666 | no claim |
| FedEx route | 8.8 | 208.0 [172.0, 241.2] | 0.0002 | FedEx route | 2.44 | 0.25 [-0.06, 0.57] | 1.0000 | no claim | 256.4 | 141.3 [75.4, 206.7] | 0.1383 | no claim |
| FedCS | 9.5 | 207.4 [171.3, 241.0] | 0.0002 | FedCS | 2.30 | 0.39 [0.07, 0.71] | 0.7006 | no claim | 243.1 | 154.6 [51.9, 258.8] | 0.5310 | no claim |

## checks: n6k1_knee_a1est

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 246.5 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 204.5 | 42.0 [14.9, 73.9] | 0.1383 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 264.8 | -18.3 [-101.7, 69.4] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 274.2 | -27.6 [-117.1, 63.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 268.2 | -21.7 [-98.2, 57.5] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 277.2 | -30.6 [-118.2, 64.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 244.3 | 2.3 [-61.8, 70.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 260.3 | -13.8 [-84.4, 60.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_a1est, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 125.6 |  |  |  | 3.31 |  |  |  | 246.5 |  |  |  |
| FX | 110.4 | 15.2 [1.9, 32.3] | 0.3848 | no claim | 3.45 | -0.14 [-0.28, 0.00] | 1.0000 | no claim | 204.5 | 42.0 [14.9, 73.9] | 0.1210 | no claim |
| H1 | 9.4 | 116.2 [92.7, 139.9] | 0.0002 | H1 | 2.65 | 0.66 [0.44, 0.90] | 0.0222 | F | 264.8 | -18.3 [-101.7, 69.4] | 1.0000 | no claim |
| MAX-AoI | 9.2 | 116.4 [92.8, 140.5] | 0.0002 | MAX-AoI | 2.77 | 0.54 [0.31, 0.78] | 0.0540 | no claim | 274.2 | -27.6 [-117.1, 63.3] | 1.0000 | no claim |
| Oort | 10.3 | 115.3 [91.7, 139.2] | 0.0002 | Oort | 2.75 | 0.56 [0.31, 0.81] | 0.0595 | no claim | 268.2 | -21.7 [-98.2, 57.5] | 1.0000 | no claim |
| Whittle | 9.3 | 116.3 [92.5, 140.3] | 0.0002 | Whittle | 2.73 | 0.59 [0.34, 0.82] | 0.0553 | no claim | 277.2 | -30.6 [-118.2, 64.3] | 1.0000 | no claim |
| FedEx route | 10.1 | 115.5 [91.8, 139.2] | 0.0002 | FedEx route | 3.00 | 0.31 [0.11, 0.53] | 0.2660 | no claim | 244.3 | 2.3 [-61.8, 70.1] | 1.0000 | no claim |
| FedCS | 9.8 | 115.8 [92.4, 139.9] | 0.0002 | FedCS | 2.85 | 0.46 [0.28, 0.65] | 0.0589 | no claim | 260.3 | -13.8 [-84.4, 60.0] | 1.0000 | no claim |

## checks: n6k1_knee_d1ampharshres

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.3 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 125.3 | 25.0 [9.6, 45.9] | 0.0237 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 276.4 | -126.1 [-182.9, -69.6] | 0.0193 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 286.4 | -136.1 [-202.4, -72.0] | 0.0193 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 0.95 | 246.7 | -99.4 [-144.3, -42.8] | 0.0598 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 0.95 | 262.7 | -115.4 [-172.7, -70.7] | 0.0045 | F | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 215.9 | -65.6 [-111.9, -11.5] | 0.3573 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 258.9 | -108.6 [-162.7, -56.0] | 0.0328 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_d1ampharshres, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 59.1 |  |  |  | 3.54 |  |  |  | 150.3 |  |  |  |
| FX | 46.5 | 12.6 [7.8, 17.7] | 0.0109 | FX | 3.55 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 125.3 | 25.0 [9.6, 45.9] | 0.0237 | FX |
| H1 | 15.3 | 43.8 [40.1, 47.6] | 0.0002 | H1 | 2.70 | 0.84 [0.66, 1.02] | 0.0108 | F | 276.4 | -126.1 [-182.9, -69.6] | 0.0193 | F |
| MAX-AoI | 15.6 | 43.5 [39.2, 47.4] | 0.0002 | MAX-AoI | 2.79 | 0.75 [0.56, 0.94] | 0.0082 | F | 286.4 | -136.1 [-202.4, -72.0] | 0.0193 | F |
| Oort | 14.7 | 44.5 [39.3, 49.5] | 0.0002 | Oort | 2.83 | 0.71 [0.56, 0.88] | 0.0080 | F | 265.2 | -114.9 [-167.8, -57.3] | 0.0328 | F |
| Whittle | 16.5 | 42.7 [37.1, 47.7] | 0.0002 | Whittle | 2.92 | 0.61 [0.45, 0.78] | 0.0180 | F | 280.4 | -130.1 [-195.6, -77.7] | 0.0023 | F |
| FedEx route | 16.8 | 42.3 [37.8, 46.4] | 0.0002 | FedEx route | 2.96 | 0.57 [0.40, 0.78] | 0.0108 | F | 215.9 | -65.6 [-111.9, -11.5] | 0.3490 | no claim |
| FedCS | 14.9 | 44.2 [39.3, 48.9] | 0.0002 | FedCS | 2.99 | 0.55 [0.37, 0.74] | 0.0144 | F | 258.9 | -108.6 [-162.7, -56.0] | 0.0328 | F |

## checks: n6k1_knee_d1burst10l

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 183.5 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 164.1 | 19.4 [6.9, 39.7] | 0.0221 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 294.8 | -111.3 [-180.5, -48.1] | 0.0963 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 286.0 | -102.5 [-159.9, -46.3] | 0.0963 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 282.9 | -99.4 [-155.1, -47.1] | 0.0859 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 302.2 | -118.7 [-185.9, -57.5] | 0.0328 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 238.7 | -55.2 [-109.0, 1.5] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 273.2 | -89.7 [-147.4, -33.6] | 0.1608 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_d1burst10l, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 86.1 |  |  |  | 3.54 |  |  |  | 183.5 |  |  |  |
| FX | 75.6 | 10.5 [6.1, 15.6] | 0.0178 | FX | 3.52 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 164.1 | 19.4 [6.9, 39.7] | 0.0221 | FX |
| H1 | 12.8 | 73.3 [62.4, 83.5] | 0.0002 | H1 | 3.31 | 0.23 [0.09, 0.40] | 0.2226 | no claim | 294.8 | -111.3 [-180.5, -48.1] | 0.0963 | no claim |
| MAX-AoI | 12.7 | 73.4 [62.3, 83.7] | 0.0002 | MAX-AoI | 3.33 | 0.21 [0.10, 0.35] | 0.2168 | no claim | 286.0 | -102.5 [-159.9, -46.3] | 0.0963 | no claim |
| Oort | 12.0 | 74.1 [62.9, 84.4] | 0.0002 | Oort | 3.25 | 0.29 [0.15, 0.45] | 0.1130 | no claim | 282.9 | -99.4 [-155.1, -47.1] | 0.0859 | no claim |
| Whittle | 11.6 | 74.5 [63.2, 85.0] | 0.0002 | Whittle | 3.30 | 0.24 [0.10, 0.40] | 0.1638 | no claim | 302.2 | -118.7 [-185.9, -57.5] | 0.0328 | F |
| FedEx route | 13.0 | 73.1 [62.6, 83.1] | 0.0002 | FedEx route | 3.31 | 0.23 [0.09, 0.39] | 0.2226 | no claim | 238.7 | -55.2 [-109.0, 1.5] | 1.0000 | no claim |
| FedCS | 12.4 | 73.7 [63.1, 83.8] | 0.0002 | FedCS | 3.35 | 0.19 [0.07, 0.31] | 0.2226 | no claim | 273.2 | -89.7 [-147.4, -33.6] | 0.1576 | no claim |

## checks: n6k1_knee_d1burst40l

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 213.7 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 192.8 | 20.9 [-10.2, 55.6] | 0.2289 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 314.7 | -101.0 [-183.4, -22.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 307.6 | -93.8 [-170.1, -20.7] | 0.9836 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 262.7 | -48.9 [-110.7, 22.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 347.4 | -133.7 [-224.1, -50.5] | 0.2811 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 237.4 | -23.7 [-69.3, 27.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 290.5 | -76.8 [-159.4, 2.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |

D0's endpoints in checks: n6k1_knee_d1burst40l, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 109.1 |  |  |  | 3.27 |  |  |  | 213.7 |  |  |  |
| FX | 92.3 | 16.8 [7.8, 25.4] | 0.0278 | FX | 3.41 | -0.14 [-0.29, -0.03] | 1.0000 | no claim | 192.8 | 20.9 [-10.2, 55.6] | 0.2289 | no claim |
| H1 | 14.6 | 94.6 [80.9, 107.4] | 0.0002 | H1 | 2.46 | 0.81 [0.57, 1.02] | 0.0147 | F | 314.7 | -101.0 [-183.4, -22.1] | 0.9830 | no claim |
| MAX-AoI | 14.4 | 94.8 [81.2, 107.8] | 0.0002 | MAX-AoI | 2.52 | 0.75 [0.51, 0.96] | 0.0147 | F | 307.6 | -93.8 [-170.1, -20.7] | 0.9464 | no claim |
| Oort | 13.7 | 95.5 [82.2, 108.4] | 0.0002 | Oort | 2.56 | 0.71 [0.41, 0.99] | 0.0553 | no claim | 262.7 | -48.9 [-110.7, 22.1] | 1.0000 | no claim |
| Whittle | 11.3 | 97.9 [84.8, 111.6] | 0.0002 | Whittle | 2.34 | 0.94 [0.71, 1.15] | 0.0082 | F | 347.4 | -133.7 [-224.1, -50.5] | 0.2748 | no claim |
| FedEx route | 15.0 | 94.2 [81.1, 106.9] | 0.0002 | FedEx route | 2.51 | 0.76 [0.40, 1.11] | 0.0595 | no claim | 237.4 | -23.7 [-69.3, 27.8] | 1.0000 | no claim |
| FedCS | 14.3 | 94.8 [81.1, 108.6] | 0.0002 | FedCS | 2.45 | 0.82 [0.51, 1.12] | 0.0526 | no claim | 290.5 | -76.8 [-159.4, 2.8] | 1.0000 | no claim |

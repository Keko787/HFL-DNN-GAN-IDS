# DDIL2: Step 2: the risk-priced dwell and the leave-by time

Exploratory, not pre-registered (DeveloperDocs/Experiment_5_Loss_Probes_Plan.md). Reference F. Time to tau = 0.71 over the trials both arms reach it, and reach at tau (exact McNemar on the paired outcomes), each with a 95% bootstrap CI. Holm per family: the baselines on time, on reach, and FX on each apart. Difference = F - arm: time, negative when F is faster; reach, positive when F reaches tau more often.

DDIL build D0's endpoints (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md), paired by trial and seed, each in a Holm family of its own: hover per mission (sim_dwell_s_mean, lower is better), updates merged per mission (update_yield, higher), and time to tau with a trial that missed it given the trial's end, so no trial is dropped (lower).

Other pairs, in families of their own: F-rpnl against F-rp, F-rp against FX; the difference is the second arm minus the first.

Flown and scored: 110 of 110 arm-cells.

## Where a baseline beats F

On time to τ, reach, updates per mission, or time with misses at the end. A baseline's lower hover is listed in the cells' tables, not here: it comes from leaving the members below the floor unserved.

- none (so far)

## honest: n6k1_knee_d1base

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 166.6 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 148.0 | 18.6 [8.8, 30.0] | 0.0196 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 286.3 | -119.6 [-180.0, -70.7] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 276.9 | -110.3 [-159.0, -70.0] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 265.7 | -99.1 [-145.4, -64.7] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 290.1 | -123.5 [-185.2, -74.8] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 246.1 | -79.5 [-116.4, -50.9] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 266.7 | -100.1 [-148.9, -59.7] | 0.0006 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 224.1 | -57.5 [-84.4, -34.3] | 0.0097 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1base, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 76.0 |  |  |  | 3.55 |  |  |  | 166.6 |  |  |  |
| FX | 67.7 | 8.4 [4.6, 12.7] | 0.0132 | FX | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 148.0 | 18.6 [8.8, 30.0] | 0.0196 | FX |
| H1 | 11.9 | 64.2 [55.0, 73.0] | 0.0001 | H1 | 3.40 | 0.15 [0.04, 0.31] | 0.5651 | no claim | 286.3 | -119.6 [-180.0, -70.7] | 0.0006 | F |
| MAX-AoI | 11.6 | 64.5 [55.4, 73.3] | 0.0001 | MAX-AoI | 3.42 | 0.12 [0.04, 0.25] | 0.5651 | no claim | 276.9 | -110.3 [-159.0, -70.0] | 0.0006 | F |
| Oort | 11.6 | 64.5 [54.9, 73.6] | 0.0001 | Oort | 3.48 | 0.07 [0.00, 0.16] | 1.0000 | no claim | 265.7 | -99.1 [-145.4, -64.7] | 0.0004 | F |
| Whittle | 10.8 | 65.3 [55.9, 74.2] | 0.0001 | Whittle | 3.40 | 0.15 [0.04, 0.29] | 0.5651 | no claim | 290.1 | -123.5 [-185.2, -74.8] | 0.0006 | F |
| FedEx route | 11.3 | 64.7 [55.6, 73.2] | 0.0001 | FedEx route | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 246.1 | -79.5 [-116.4, -50.9] | 0.0006 | F |
| FedCS | 11.4 | 64.7 [55.3, 73.6] | 0.0001 | FedCS | 3.52 | 0.03 [0.00, 0.06] | 1.0000 | no claim | 266.7 | -100.1 [-148.9, -59.7] | 0.0006 | F |
| F-rp | 21.2 | 54.8 [45.4, 63.4] | 0.0000 | F-rp | 3.52 | 0.03 [0.00, 0.06] | 1.0000 | no claim | 224.1 | -57.5 [-84.4, -34.3] | 0.0097 | F |

## honest: n6k1_knee_d1amp11

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 152.1 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 136.4 | 15.6 [7.2, 25.2] | 0.0321 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 272.8 | -120.7 [-179.3, -69.0] | 0.0051 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 257.7 | -105.6 [-157.1, -60.6] | 0.0064 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 246.1 | -94.0 [-149.1, -30.1] | 0.0851 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 262.6 | -110.6 [-166.2, -62.7] | 0.0122 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 261.3 | -109.2 [-158.0, -70.9] | 0.0004 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 272.8 | -120.7 [-183.5, -64.3] | 0.0105 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 230.4 | -78.3 [-123.1, -42.6] | 0.0111 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1amp11, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 65.6 |  |  |  | 3.55 |  |  |  | 152.1 |  |  |  |
| FX | 59.1 | 6.6 [3.1, 10.6] | 0.0160 | FX | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 136.4 | 15.6 [7.2, 25.2] | 0.0321 | FX |
| H1 | 15.9 | 49.7 [43.5, 55.8] | 0.0001 | H1 | 3.29 | 0.26 [0.14, 0.41] | 0.0668 | no claim | 272.8 | -120.7 [-179.3, -69.0] | 0.0051 | F |
| MAX-AoI | 16.3 | 49.3 [42.8, 55.5] | 0.0001 | MAX-AoI | 3.34 | 0.21 [0.11, 0.33] | 0.0994 | no claim | 257.7 | -105.6 [-157.1, -60.6] | 0.0064 | F |
| Oort | 16.2 | 49.5 [42.8, 55.5] | 0.0001 | Oort | 3.29 | 0.26 [0.14, 0.42] | 0.0383 | F | 246.1 | -94.0 [-149.1, -30.1] | 0.0851 | no claim |
| Whittle | 15.0 | 50.6 [44.7, 56.2] | 0.0001 | Whittle | 3.35 | 0.20 [0.11, 0.30] | 0.0994 | no claim | 262.6 | -110.6 [-166.2, -62.7] | 0.0122 | F |
| FedEx route | 15.4 | 50.2 [43.7, 56.1] | 0.0001 | FedEx route | 3.39 | 0.16 [0.09, 0.25] | 0.1229 | no claim | 261.3 | -109.2 [-158.0, -70.9] | 0.0004 | F |
| FedCS | 14.4 | 51.2 [44.0, 57.7] | 0.0001 | FedCS | 3.30 | 0.25 [0.15, 0.35] | 0.0285 | F | 272.8 | -120.7 [-183.5, -64.3] | 0.0105 | F |
| F-rp | 21.8 | 43.8 [37.4, 50.0] | 0.0000 | F-rp | 3.40 | 0.15 [0.05, 0.31] | 0.2168 | no claim | 230.4 | -78.3 [-123.1, -42.6] | 0.0111 | F |

## honest: n6k1_knee_d1ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.9 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 125.4 | 25.6 [9.9, 46.8] | 0.0280 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 277.2 | -126.2 [-183.6, -69.5] | 0.0105 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 287.7 | -136.8 [-204.3, -72.4] | 0.0122 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 0.95 | 246.3 | -98.4 [-143.1, -41.5] | 0.0375 | F | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 0.95 | 263.4 | -115.5 [-173.9, -70.3] | 0.0027 | F | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 215.9 | -65.0 [-111.3, -10.7] | 0.1887 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 260.0 | -109.0 [-163.7, -56.4] | 0.0205 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 224.2 | -73.3 [-129.3, -26.7] | 0.0463 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |
| F-rpnl | 20 | 1.00 | 224.9 | -74.0 [-129.4, -27.5] | 0.0463 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rpnl only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 59.3 |  |  |  | 3.54 |  |  |  | 150.9 |  |  |  |
| FX | 46.6 | 12.8 [8.5, 17.3] | 0.0048 | FX | 3.55 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 125.4 | 25.6 [9.9, 46.8] | 0.0280 | FX |
| H1 | 15.0 | 44.3 [40.4, 48.3] | 0.0001 | H1 | 2.73 | 0.81 [0.61, 1.01] | 0.0079 | F | 277.2 | -126.2 [-183.6, -69.5] | 0.0105 | F |
| MAX-AoI | 15.5 | 43.9 [39.7, 47.5] | 0.0001 | MAX-AoI | 2.79 | 0.75 [0.57, 0.94] | 0.0058 | F | 287.7 | -136.8 [-204.3, -72.4] | 0.0122 | F |
| Oort | 14.4 | 45.0 [39.9, 49.8] | 0.0001 | Oort | 2.86 | 0.68 [0.54, 0.82] | 0.0055 | F | 264.8 | -113.9 [-166.2, -56.5] | 0.0205 | F |
| Whittle | 17.7 | 41.7 [36.0, 47.0] | 0.0001 | Whittle | 2.94 | 0.60 [0.42, 0.78] | 0.0108 | F | 281.0 | -130.1 [-195.1, -77.3] | 0.0015 | F |
| FedEx route | 16.5 | 42.8 [37.5, 47.8] | 0.0001 | FedEx route | 2.94 | 0.60 [0.40, 0.82] | 0.0077 | F | 215.9 | -65.0 [-111.3, -10.7] | 0.1793 | no claim |
| FedCS | 14.8 | 44.5 [39.3, 49.4] | 0.0001 | FedCS | 2.94 | 0.60 [0.39, 0.80] | 0.0106 | F | 260.0 | -109.0 [-163.7, -56.4] | 0.0205 | F |
| F-rp | 18.6 | 40.7 [35.1, 46.1] | 0.0000 | F-rp | 2.86 | 0.68 [0.46, 0.88] | 0.0042 | F | 224.2 | -73.3 [-129.3, -26.7] | 0.0463 | F |
| F-rpnl | 19.8 | 39.5 [34.1, 44.9] | 0.0000 | F-rpnl | 2.92 | 0.61 [0.41, 0.81] | 0.0042 | F | 224.9 | -74.0 [-129.4, -27.5] | 0.0463 | F |

## honest: n6k1_knee_d1npl

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 184.2 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 173.2 | 11.1 [2.9, 21.7] | 0.0607 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 273.0 | -88.8 [-124.0, -58.9] | 0.0071 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 272.4 | -88.2 [-122.6, -58.5] | 0.0071 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 262.3 | -78.1 [-107.7, -52.4] | 0.0071 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.7 | -80.5 [-109.4, -54.2] | 0.0071 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 242.9 | -58.6 [-80.3, -38.6] | 0.0071 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 262.5 | -78.3 [-113.6, -48.4] | 0.0071 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 224.2 | -39.9 [-55.2, -24.8] | 0.0116 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1npl, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 69.8 |  |  |  | 3.55 |  |  |  | 184.2 |  |  |  |
| FX | 65.9 | 3.9 [0.8, 8.1] | 0.1112 | no claim | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 173.2 | 11.1 [2.9, 21.7] | 0.0607 | no claim |
| H1 | 8.9 | 60.9 [50.9, 70.1] | 0.0001 | H1 | 3.46 | 0.09 [0.00, 0.23] | 1.0000 | no claim | 273.0 | -88.8 [-124.0, -58.9] | 0.0071 | F |
| MAX-AoI | 8.9 | 60.9 [50.9, 70.1] | 0.0001 | MAX-AoI | 3.46 | 0.09 [0.00, 0.20] | 1.0000 | no claim | 272.4 | -88.2 [-122.6, -58.5] | 0.0071 | F |
| Oort | 9.1 | 60.7 [50.7, 69.9] | 0.0001 | Oort | 3.49 | 0.06 [0.00, 0.15] | 1.0000 | no claim | 262.3 | -78.1 [-107.7, -52.4] | 0.0071 | F |
| Whittle | 9.2 | 60.7 [50.5, 69.9] | 0.0001 | Whittle | 3.48 | 0.07 [0.01, 0.15] | 1.0000 | no claim | 264.7 | -80.5 [-109.4, -54.2] | 0.0071 | F |
| FedEx route | 9.1 | 60.8 [50.7, 69.9] | 0.0001 | FedEx route | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 242.9 | -58.6 [-80.3, -38.6] | 0.0071 | F |
| FedCS | 8.7 | 61.1 [51.2, 70.3] | 0.0001 | FedCS | 3.54 | 0.01 [0.00, 0.04] | 1.0000 | no claim | 262.5 | -78.3 [-113.6, -48.4] | 0.0071 | F |
| F-rp | 17.3 | 52.5 [42.5, 62.1] | 0.0000 | F-rp | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 224.2 | -39.9 [-55.2, -24.8] | 0.0116 | F |

## honest: n6k1_knee_d1sigma8

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 133.6 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 114.4 | 19.1 [12.2, 26.5] | 0.0144 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 258.1 | -124.6 [-174.7, -85.8] | 0.0003 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 257.8 | -124.3 [-174.5, -85.1] | 0.0003 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 223.3 | -89.7 [-125.3, -42.3] | 0.0289 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.9 | -131.3 [-184.5, -91.1] | 0.0003 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 251.0 | -117.4 [-165.2, -81.2] | 0.0003 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 262.8 | -129.3 [-184.7, -84.3] | 0.0003 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 216.0 | -82.4 [-119.1, -52.4] | 0.0056 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1sigma8, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 58.1 |  |  |  | 3.55 |  |  |  | 133.6 |  |  |  |
| FX | 48.1 | 10.0 [6.3, 13.7] | 0.0058 | FX | 3.55 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 114.4 | 19.1 [12.2, 26.5] | 0.0144 | FX |
| H1 | 8.3 | 49.8 [44.2, 55.0] | 0.0001 | H1 | 3.45 | 0.10 [0.03, 0.18] | 0.7687 | no claim | 258.1 | -124.6 [-174.7, -85.8] | 0.0003 | F |
| MAX-AoI | 8.3 | 49.8 [44.1, 55.0] | 0.0001 | MAX-AoI | 3.41 | 0.14 [0.05, 0.24] | 0.4414 | no claim | 257.8 | -124.3 [-174.5, -85.1] | 0.0003 | F |
| Oort | 8.1 | 50.0 [44.8, 54.8] | 0.0001 | Oort | 3.39 | 0.16 [0.06, 0.26] | 0.2993 | no claim | 223.3 | -89.7 [-125.3, -42.3] | 0.0281 | F |
| Whittle | 7.8 | 50.3 [44.7, 55.1] | 0.0001 | Whittle | 3.39 | 0.16 [0.09, 0.24] | 0.1229 | no claim | 264.9 | -131.3 [-184.5, -91.1] | 0.0003 | F |
| FedEx route | 7.6 | 50.5 [45.3, 55.3] | 0.0001 | FedEx route | 3.48 | 0.07 [0.03, 0.12] | 0.4149 | no claim | 251.0 | -117.4 [-165.2, -81.2] | 0.0003 | F |
| FedCS | 8.6 | 49.5 [43.8, 54.8] | 0.0001 | FedCS | 3.50 | 0.05 [0.01, 0.10] | 0.8645 | no claim | 262.8 | -129.3 [-184.7, -84.3] | 0.0003 | F |
| F-rp | 16.3 | 41.8 [35.2, 47.8] | 0.0017 | F-rp | 3.46 | 0.09 [0.04, 0.14] | 0.1630 | no claim | 216.0 | -82.4 [-119.1, -52.4] | 0.0056 | F |

## honest: n6k1_knee_d1burst10

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 170.3 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 158.3 | 12.0 [5.4, 19.6] | 0.0280 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 307.4 | -137.0 [-200.0, -84.3] | 0.0008 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 298.3 | -127.9 [-179.6, -81.7] | 0.0008 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 274.3 | -104.0 [-136.1, -73.8] | 0.0008 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 316.7 | -146.3 [-211.2, -89.6] | 0.0011 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 258.4 | -88.0 [-130.3, -52.8] | 0.0011 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 286.2 | -115.9 [-171.4, -68.8] | 0.0015 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 235.1 | -64.8 [-99.8, -34.5] | 0.0155 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1burst10, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 79.3 |  |  |  | 3.54 |  |  |  | 170.3 |  |  |  |
| FX | 71.0 | 8.3 [3.2, 14.1] | 0.1406 | no claim | 3.48 | 0.06 [0.00, 0.15] | 1.0000 | no claim | 158.3 | 12.0 [5.4, 19.6] | 0.0280 | FX |
| H1 | 12.4 | 66.9 [57.3, 75.7] | 0.0001 | H1 | 3.25 | 0.29 [0.14, 0.46] | 0.1101 | no claim | 307.4 | -137.0 [-200.0, -84.3] | 0.0008 | F |
| MAX-AoI | 12.0 | 67.3 [57.6, 76.3] | 0.0001 | MAX-AoI | 3.26 | 0.28 [0.14, 0.45] | 0.1101 | no claim | 298.3 | -127.9 [-179.6, -81.7] | 0.0008 | F |
| Oort | 11.4 | 67.9 [58.3, 76.6] | 0.0001 | Oort | 3.23 | 0.31 [0.16, 0.49] | 0.0773 | no claim | 274.3 | -104.0 [-136.1, -73.8] | 0.0008 | F |
| Whittle | 12.0 | 67.3 [57.4, 76.3] | 0.0001 | Whittle | 3.19 | 0.35 [0.16, 0.56] | 0.1101 | no claim | 316.7 | -146.3 [-211.2, -89.6] | 0.0011 | F |
| FedEx route | 12.1 | 67.2 [57.8, 75.8] | 0.0001 | FedEx route | 3.27 | 0.26 [0.11, 0.45] | 0.2114 | no claim | 258.4 | -88.0 [-130.3, -52.8] | 0.0011 | F |
| FedCS | 11.7 | 67.6 [58.0, 76.3] | 0.0001 | FedCS | 3.35 | 0.19 [0.10, 0.28] | 0.1317 | no claim | 286.2 | -115.9 [-171.4, -68.8] | 0.0015 | F |
| F-rp | 19.1 | 60.2 [49.8, 70.2] | 0.0000 | F-rp | 3.34 | 0.20 [0.06, 0.38] | 0.2881 | no claim | 235.1 | -64.8 [-99.8, -34.5] | 0.0155 | F |

## honest: n6k1_knee_d1burst40

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 229.7 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 217.1 | 12.6 [-5.4, 30.2] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 265.4 | -35.7 [-103.2, 44.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 0.95 | 256.9 | -36.1 [-108.6, 49.6] | 1.0000 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 271.3 | -41.6 [-123.4, 47.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 225.7 | 4.0 [-56.6, 77.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 269.2 | -39.5 [-92.9, 15.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 283.7 | -54.0 [-120.6, 27.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 0.95 | 271.4 | -40.2 [-105.8, 22.6] | 1.0000 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1burst40, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 110.0 |  |  |  | 3.24 |  |  |  | 229.7 |  |  |  |
| FX | 96.7 | 13.4 [4.5, 22.7] | 0.1009 | no claim | 3.27 | -0.04 [-0.28, 0.19] | 1.0000 | no claim | 217.1 | 12.6 [-5.4, 30.2] | 1.0000 | no claim |
| H1 | 12.7 | 97.3 [86.5, 107.9] | 0.0001 | H1 | 2.48 | 0.76 [0.56, 0.99] | 0.0108 | F | 265.4 | -35.7 [-103.2, 44.8] | 1.0000 | no claim |
| MAX-AoI | 12.7 | 97.4 [87.1, 108.2] | 0.0001 | MAX-AoI | 2.45 | 0.79 [0.55, 1.05] | 0.0112 | F | 269.1 | -39.4 [-108.3, 41.4] | 1.0000 | no claim |
| Oort | 12.3 | 97.7 [87.2, 107.7] | 0.0001 | Oort | 2.42 | 0.81 [0.59, 1.05] | 0.0108 | F | 271.3 | -41.6 [-123.4, 47.7] | 1.0000 | no claim |
| Whittle | 12.5 | 97.5 [87.1, 107.8] | 0.0001 | Whittle | 2.65 | 0.59 [0.34, 0.84] | 0.0520 | no claim | 225.7 | 4.0 [-56.6, 77.1] | 1.0000 | no claim |
| FedEx route | 12.0 | 98.0 [87.5, 109.2] | 0.0001 | FedEx route | 2.19 | 1.05 [0.68, 1.49] | 0.0197 | F | 269.2 | -39.5 [-92.9, 15.0] | 1.0000 | no claim |
| FedCS | 12.6 | 97.4 [85.4, 108.8] | 0.0001 | FedCS | 2.34 | 0.90 [0.61, 1.21] | 0.0099 | F | 283.7 | -54.0 [-120.6, 27.0] | 0.9741 | no claim |
| F-rp | 16.2 | 93.8 [81.2, 104.8] | 0.0000 | F-rp | 2.33 | 0.91 [0.55, 1.26] | 0.0148 | F | 281.5 | -51.8 [-115.4, 12.4] | 0.9324 | no claim |

## honest: n6k1_knee_d1ampharshpc

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 150.3 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 144.1 | 6.2 [-6.2, 17.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 284.3 | -133.9 [-203.4, -63.9] | 0.0507 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 286.2 | -135.9 [-206.6, -64.9] | 0.0507 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 246.4 | -96.0 [-147.0, -48.6] | 0.0289 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 264.2 | -113.8 [-184.0, -43.8] | 0.0851 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 259.8 | -109.5 [-174.0, -49.1] | 0.0651 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 257.3 | -106.9 [-171.8, -40.0] | 0.1115 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 235.7 | -85.3 [-142.2, -38.0] | 0.0334 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_knee_d1ampharshpc, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 58.3 |  |  |  | 3.54 |  |  |  | 150.3 |  |  |  |
| FX | 54.6 | 3.7 [-1.3, 9.3] | 0.5611 | no claim | 3.54 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 144.1 | 6.2 [-6.2, 17.7] | 1.0000 | no claim |
| H1 | 16.1 | 42.2 [36.8, 47.2] | 0.0001 | H1 | 2.83 | 0.71 [0.53, 0.90] | 0.0079 | F | 284.3 | -133.9 [-203.4, -63.9] | 0.0507 | no claim |
| MAX-AoI | 16.0 | 42.3 [36.8, 47.4] | 0.0001 | MAX-AoI | 2.79 | 0.75 [0.59, 0.91] | 0.0079 | F | 286.2 | -135.9 [-206.6, -64.9] | 0.0507 | no claim |
| Oort | 15.3 | 43.0 [37.8, 48.0] | 0.0001 | Oort | 2.83 | 0.71 [0.51, 0.91] | 0.0108 | F | 246.4 | -96.0 [-147.0, -48.6] | 0.0281 | F |
| Whittle | 15.8 | 42.5 [36.9, 48.0] | 0.0001 | Whittle | 2.85 | 0.69 [0.54, 0.84] | 0.0058 | F | 264.2 | -113.8 [-184.0, -43.8] | 0.0851 | no claim |
| FedEx route | 16.3 | 42.0 [36.3, 47.9] | 0.0001 | FedEx route | 3.04 | 0.50 [0.33, 0.68] | 0.0127 | F | 259.8 | -109.5 [-174.0, -49.1] | 0.0651 | no claim |
| FedCS | 17.3 | 41.0 [35.6, 46.1] | 0.0001 | FedCS | 2.90 | 0.64 [0.44, 0.82] | 0.0197 | F | 257.3 | -106.9 [-171.8, -40.0] | 0.1067 | no claim |
| F-rp | 22.5 | 35.8 [28.5, 42.9] | 0.0000 | F-rp | 3.08 | 0.46 [0.29, 0.64] | 0.0128 | F | 235.7 | -85.3 [-142.2, -38.0] | 0.0334 | F |

## honest: n6k1_stress_d1ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 171.5 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 152.0 | 19.6 [5.9, 40.1] | 0.0321 | FX | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 0.85 | 248.7 | -80.7 [-141.0, -11.6] | 0.2700 | no claim | 0.15 [0.00, 0.30] | 3 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 293.7 | -122.2 [-204.5, -37.9] | 0.1887 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 0.90 | 212.3 | -32.7 [-84.4, 25.5] | 1.0000 | no claim | 0.10 [0.00, 0.25] | 2 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 241.6 | -70.0 [-155.6, 13.6] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 215.9 | -44.4 [-100.5, 20.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 0.95 | 247.7 | -87.0 [-154.5, -11.4] | 0.4629 | no claim | 0.05 [0.00, 0.15] | 1 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 165.9 | 5.6 [-66.0, 79.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |
| F-rpnl | 20 | 1.00 | 159.4 | 12.1 [-60.9, 85.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rpnl only | 1.0000 | no claim |

D0's endpoints in honest: n6k1_stress_d1ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 51.6 |  |  |  | 3.09 |  |  |  | 171.5 |  |  |  |
| FX | 41.1 | 10.5 [6.2, 15.2] | 0.0028 | FX | 3.08 | 0.01 [-0.07, 0.11] | 1.0000 | no claim | 152.0 | 19.6 [5.9, 40.1] | 0.0321 | FX |
| H1 | 10.6 | 41.0 [34.9, 47.1] | 0.0001 | H1 | 1.69 | 1.40 [1.07, 1.75] | 0.0058 | F | 275.7 | -104.1 [-166.1, -39.5] | 0.0851 | no claim |
| MAX-AoI | 13.2 | 38.3 [31.2, 45.0] | 0.0001 | MAX-AoI | 1.94 | 1.15 [0.75, 1.59] | 0.0115 | F | 293.7 | -122.2 [-204.5, -37.9] | 0.1793 | no claim |
| Oort | 10.6 | 41.0 [34.2, 47.5] | 0.0001 | Oort | 1.75 | 1.34 [0.96, 1.76] | 0.0079 | F | 232.6 | -61.0 [-125.4, 3.3] | 0.4436 | no claim |
| Whittle | 12.8 | 38.7 [32.1, 44.8] | 0.0001 | Whittle | 1.89 | 1.20 [0.88, 1.55] | 0.0079 | F | 241.6 | -70.0 [-155.6, 13.6] | 1.0000 | no claim |
| FedEx route | 16.5 | 35.0 [28.3, 41.0] | 0.0001 | FedEx route | 2.94 | 0.15 [-0.15, 0.45] | 1.0000 | no claim | 215.9 | -44.4 [-100.5, 20.0] | 1.0000 | no claim |
| FedCS | 13.3 | 38.2 [31.4, 44.5] | 0.0001 | FedCS | 2.08 | 1.01 [0.74, 1.30] | 0.0115 | F | 258.6 | -87.1 [-151.7, -21.4] | 0.3652 | no claim |
| F-rp | 17.4 | 34.2 [27.3, 40.5] | 0.0000 | F-rp | 1.90 | 1.19 [0.86, 1.49] | 0.0037 | F | 165.9 | 5.6 [-66.0, 79.1] | 1.0000 | no claim |
| F-rpnl | 17.7 | 33.9 [27.6, 39.8] | 0.0000 | F-rpnl | 1.89 | 1.20 [0.88, 1.50] | 0.0037 | F | 159.4 | 12.1 [-60.9, 85.0] | 1.0000 | no claim |

## recorded: n6k1_knee_base

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 194.0 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 178.1 | 16.0 [3.9, 31.8] | 0.0845 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 281.8 | -87.8 [-130.9, -51.5] | 0.0020 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 281.8 | -87.8 [-132.2, -51.4] | 0.0020 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 265.8 | -71.7 [-111.9, -41.8] | 0.0041 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 279.6 | -85.5 [-128.5, -49.5] | 0.0020 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 264.5 | -70.4 [-111.7, -37.2] | 0.0142 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 274.6 | -80.5 [-124.5, -44.1] | 0.0064 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 232.8 | -38.8 [-70.0, -13.7] | 0.1565 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in recorded: n6k1_knee_base, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 107.2 |  |  |  | 3.54 |  |  |  | 194.0 |  |  |  |
| FX | 88.2 | 19.1 [11.6, 27.6] | 0.0050 | FX | 3.54 | 0.00 [0.00, 0.00] | 1.0000 | no claim | 178.1 | 16.0 [3.9, 31.8] | 0.0845 | no claim |
| H1 | 11.0 | 96.2 [80.7, 111.4] | 0.0001 | H1 | 3.46 | 0.07 [0.00, 0.15] | 1.0000 | no claim | 281.8 | -87.8 [-130.9, -51.5] | 0.0019 | F |
| MAX-AoI | 10.8 | 96.4 [80.7, 111.7] | 0.0001 | MAX-AoI | 3.39 | 0.15 [0.04, 0.26] | 0.5121 | no claim | 281.8 | -87.8 [-132.2, -51.4] | 0.0019 | F |
| Oort | 10.8 | 96.5 [80.8, 111.9] | 0.0001 | Oort | 3.39 | 0.15 [0.03, 0.30] | 0.6517 | no claim | 265.8 | -71.7 [-111.9, -41.8] | 0.0041 | F |
| Whittle | 11.5 | 95.7 [80.1, 111.1] | 0.0001 | Whittle | 3.40 | 0.14 [0.04, 0.24] | 0.4758 | no claim | 279.6 | -85.5 [-128.5, -49.5] | 0.0019 | F |
| FedEx route | 10.7 | 96.5 [80.7, 112.0] | 0.0001 | FedEx route | 3.49 | 0.05 [0.00, 0.10] | 1.0000 | no claim | 264.5 | -70.4 [-111.7, -37.2] | 0.0142 | F |
| FedCS | 10.6 | 96.7 [81.1, 112.0] | 0.0001 | FedCS | 3.49 | 0.05 [0.00, 0.10] | 1.0000 | no claim | 274.6 | -80.5 [-124.5, -44.1] | 0.0064 | F |
| F-rp | 23.6 | 83.6 [67.6, 98.6] | 0.0000 | F-rp | 3.50 | 0.04 [-0.01, 0.09] | 1.0000 | no claim | 232.8 | -38.8 [-70.0, -13.7] | 0.1565 | no claim |

## recorded: n6k1_knee_ampharsh

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 260.5 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 215.5 | 45.1 [8.0, 89.1] | 0.4555 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 264.8 | -4.3 [-85.9, 83.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 274.2 | -13.6 [-102.0, 77.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 268.2 | -7.7 [-87.7, 69.8] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 277.2 | -16.6 [-104.6, 79.0] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 244.3 | 16.2 [-46.8, 86.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 260.3 | 0.2 [-71.7, 75.2] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 225.5 | 35.0 [-27.7, 102.7] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in recorded: n6k1_knee_ampharsh, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 133.0 |  |  |  | 3.31 |  |  |  | 260.5 |  |  |  |
| FX | 112.1 | 20.9 [3.2, 42.2] | 0.2652 | no claim | 3.45 | -0.14 [-0.28, 0.00] | 1.0000 | no claim | 215.5 | 45.1 [8.0, 89.1] | 0.4555 | no claim |
| H1 | 9.4 | 123.6 [102.3, 145.7] | 0.0001 | H1 | 2.65 | 0.66 [0.42, 0.90] | 0.0211 | F | 264.8 | -4.3 [-85.9, 83.0] | 1.0000 | no claim |
| MAX-AoI | 9.2 | 123.8 [102.2, 146.4] | 0.0001 | MAX-AoI | 2.77 | 0.54 [0.30, 0.79] | 0.0580 | no claim | 274.2 | -13.6 [-102.0, 77.3] | 1.0000 | no claim |
| Oort | 10.3 | 122.7 [101.0, 145.4] | 0.0001 | Oort | 2.75 | 0.56 [0.30, 0.82] | 0.0668 | no claim | 268.2 | -7.7 [-87.7, 69.8] | 1.0000 | no claim |
| Whittle | 9.3 | 123.7 [102.0, 146.3] | 0.0001 | Whittle | 2.73 | 0.59 [0.34, 0.84] | 0.0590 | no claim | 277.2 | -16.6 [-104.6, 79.0] | 1.0000 | no claim |
| FedEx route | 10.1 | 122.9 [100.9, 145.5] | 0.0001 | FedEx route | 3.00 | 0.31 [0.10, 0.54] | 0.3672 | no claim | 244.3 | 16.2 [-46.8, 86.3] | 1.0000 | no claim |
| FedCS | 9.8 | 123.2 [101.6, 145.5] | 0.0001 | FedCS | 2.85 | 0.46 [0.28, 0.65] | 0.0569 | no claim | 260.3 | 0.2 [-71.7, 75.2] | 1.0000 | no claim |
| F-rp | 9.9 | 123.1 [101.1, 145.7] | 0.0000 | F-rp | 3.00 | 0.31 [0.10, 0.54] | 0.2203 | no claim | 225.5 | 35.0 [-27.7, 102.7] | 1.0000 | no claim |

## recorded: n6k1_knee_npl

| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 20 | 1.00 | 220.8 |  |  |  |  |  |  |  |
| FX | 20 | 1.00 | 210.8 | 10.0 [-6.2, 29.3] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by FX only | 1.0000 | no claim |
| H1 | 20 | 1.00 | 269.3 | -48.6 [-95.3, -1.8] | 0.1115 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI | 20 | 1.00 | 269.4 | -48.7 [-96.5, -1.5] | 0.1115 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D1 only | 1.0000 | no claim |
| Oort | 20 | 1.00 | 262.7 | -41.9 [-85.8, 0.4] | 0.1026 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D2 only | 1.0000 | no claim |
| Whittle | 20 | 1.00 | 275.9 | -55.1 [-103.4, -7.1] | 0.0851 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D3 only | 1.0000 | no claim |
| FedEx route | 20 | 1.00 | 243.5 | -22.7 [-60.0, 16.1] | 0.4569 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D4 only | 1.0000 | no claim |
| FedCS | 20 | 1.00 | 262.7 | -41.9 [-72.3, -20.1] | 0.0458 | F | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by D5 only | 1.0000 | no claim |
| F-rp | 20 | 1.00 | 224.1 | -3.3 [-37.6, 39.1] | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by F only, 0 by F-rp only | 1.0000 | no claim |

D0's endpoints in recorded: n6k1_knee_npl, F − arm:

| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | updates per mission | updates: F − arm [95% CI] | Holm p | verdict | time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F | 91.0 |  |  |  | 3.51 |  |  |  | 220.8 |  |  |  |
| FX | 87.8 | 3.2 [-5.1, 12.3] | 0.5611 | no claim | 3.52 | -0.01 [-0.04, 0.00] | 1.0000 | no claim | 210.8 | 10.0 [-6.2, 29.3] | 1.0000 | no claim |
| H1 | 8.6 | 82.4 [67.3, 96.8] | 0.0001 | H1 | 3.49 | 0.03 [-0.03, 0.09] | 1.0000 | no claim | 269.3 | -48.6 [-95.3, -1.8] | 0.1067 | no claim |
| MAX-AoI | 9.0 | 82.0 [66.9, 96.8] | 0.0001 | MAX-AoI | 3.44 | 0.07 [0.00, 0.16] | 1.0000 | no claim | 269.4 | -48.7 [-96.5, -1.5] | 0.1067 | no claim |
| Oort | 8.8 | 82.2 [67.0, 96.6] | 0.0001 | Oort | 3.39 | 0.12 [0.04, 0.21] | 0.5121 | no claim | 262.7 | -41.9 [-85.8, 0.4] | 0.0983 | no claim |
| Whittle | 9.1 | 81.9 [66.8, 96.6] | 0.0001 | Whittle | 3.44 | 0.07 [0.00, 0.15] | 1.0000 | no claim | 275.9 | -55.1 [-103.4, -7.1] | 0.0851 | no claim |
| FedEx route | 8.6 | 82.4 [67.1, 96.9] | 0.0001 | FedEx route | 3.51 | 0.00 [-0.05, 0.05] | 1.0000 | no claim | 243.5 | -22.7 [-60.0, 16.1] | 0.4300 | no claim |
| FedCS | 8.4 | 82.6 [67.4, 96.8] | 0.0001 | FedCS | 3.49 | 0.03 [-0.03, 0.09] | 1.0000 | no claim | 262.7 | -41.9 [-72.3, -20.1] | 0.0458 | F |
| F-rp | 17.0 | 74.0 [56.8, 90.7] | 0.0000 | F-rp | 3.51 | 0.00 [-0.04, 0.04] | 1.0000 | no claim | 224.1 | -3.3 [-37.6, 39.1] | 1.0000 | no claim |

## Other pairs

Difference = the second arm − the first (time and hover: negative when the second is faster or hovers less; reach and updates: positive when the second reaches τ more often or merges more).

| cell | arm | against | measure | arm | against | against − arm [95% CI] | pairs | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| honest: n6k1_knee_d1base | F-rp | FX | time to tau | 224.1 | 148.0 | -76.1 [-107.1, -47.5] | 20 | 0.0026 | FX |
| honest: n6k1_knee_d1base | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1base | F-rp | FX | hover per mission | 21.2 | 67.7 | 46.4 [37.2, 55.6] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1base | F-rp | FX | updates per mission | 3.52 | 3.54 | 0.01 [-0.03, 0.05] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1base | F-rp | FX | time to tau, miss at the end | 224.1 | 148.0 | -76.1 [-107.1, -47.5] | 20 | 0.0026 | FX |
| honest: n6k1_knee_d1amp11 | F-rp | FX | time to tau | 230.4 | 136.4 | -94.0 [-136.9, -56.9] | 20 | 0.0030 | FX |
| honest: n6k1_knee_d1amp11 | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1amp11 | F-rp | FX | hover per mission | 21.8 | 59.1 | 37.3 [30.1, 44.0] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1amp11 | F-rp | FX | updates per mission | 3.40 | 3.55 | 0.15 [0.05, 0.31] | 20 | 0.0913 | no claim |
| honest: n6k1_knee_d1amp11 | F-rp | FX | time to tau, miss at the end | 230.4 | 136.4 | -94.0 [-136.9, -56.9] | 20 | 0.0030 | FX |
| honest: n6k1_knee_d1ampharsh | F-rpnl | F-rp | time to tau | 224.9 | 224.2 | -0.7 [-2.2, 0.0] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1ampharsh | F-rpnl | F-rp | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1ampharsh | F-rpnl | F-rp | hover per mission | 19.8 | 18.6 | -1.2 [-3.1, 0.0] | 20 | 0.2176 | no claim |
| honest: n6k1_knee_d1ampharsh | F-rpnl | F-rp | updates per mission | 2.92 | 2.86 | -0.06 [-0.16, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1ampharsh | F-rpnl | F-rp | time to tau, miss at the end | 224.9 | 224.2 | -0.7 [-2.2, 0.0] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1ampharsh | F-rp | FX | time to tau | 224.2 | 125.4 | -98.8 [-157.9, -50.5] | 20 | 0.0058 | FX |
| honest: n6k1_knee_d1ampharsh | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1ampharsh | F-rp | FX | hover per mission | 18.6 | 46.6 | 27.9 [22.2, 33.7] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1ampharsh | F-rp | FX | updates per mission | 2.86 | 3.55 | 0.69 [0.46, 0.90] | 20 | 0.0023 | FX |
| honest: n6k1_knee_d1ampharsh | F-rp | FX | time to tau, miss at the end | 224.2 | 125.4 | -98.8 [-157.9, -50.5] | 20 | 0.0058 | FX |
| honest: n6k1_knee_d1npl | F-rp | FX | time to tau | 224.2 | 173.2 | -51.0 [-70.0, -32.9] | 20 | 0.0035 | FX |
| honest: n6k1_knee_d1npl | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1npl | F-rp | FX | hover per mission | 17.3 | 65.9 | 48.6 [38.4, 58.3] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1npl | F-rp | FX | updates per mission | 3.55 | 3.55 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1npl | F-rp | FX | time to tau, miss at the end | 224.2 | 173.2 | -51.0 [-70.0, -32.9] | 20 | 0.0035 | FX |
| honest: n6k1_knee_d1sigma8 | F-rp | FX | time to tau | 216.0 | 114.4 | -101.6 [-137.8, -70.2] | 20 | 0.0022 | FX |
| honest: n6k1_knee_d1sigma8 | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1sigma8 | F-rp | FX | hover per mission | 16.3 | 48.1 | 31.8 [26.3, 36.9] | 20 | 0.0004 | F-rp |
| honest: n6k1_knee_d1sigma8 | F-rp | FX | updates per mission | 3.46 | 3.55 | 0.09 [0.04, 0.14] | 20 | 0.0734 | no claim |
| honest: n6k1_knee_d1sigma8 | F-rp | FX | time to tau, miss at the end | 216.0 | 114.4 | -101.6 [-137.8, -70.2] | 20 | 0.0022 | FX |
| honest: n6k1_knee_d1burst10 | F-rp | FX | time to tau | 235.1 | 158.3 | -76.8 [-111.8, -44.0] | 20 | 0.0030 | FX |
| honest: n6k1_knee_d1burst10 | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1burst10 | F-rp | FX | hover per mission | 19.1 | 71.0 | 51.9 [43.3, 60.0] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1burst10 | F-rp | FX | updates per mission | 3.34 | 3.48 | 0.14 [-0.04, 0.33] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1burst10 | F-rp | FX | time to tau, miss at the end | 235.1 | 158.3 | -76.8 [-111.8, -44.0] | 20 | 0.0030 | FX |
| honest: n6k1_knee_d1burst40 | F-rp | FX | time to tau | 271.4 | 218.0 | -53.4 [-116.1, 3.1] | 19 | 0.6649 | no claim |
| honest: n6k1_knee_d1burst40 | F-rp | FX | reach | 0.95 | 1.00 | 0.05 [0.00, 0.15] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1burst40 | F-rp | FX | hover per mission | 16.2 | 96.7 | 80.5 [68.5, 91.7] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1burst40 | F-rp | FX | updates per mission | 2.33 | 3.27 | 0.95 [0.66, 1.25] | 20 | 0.0023 | FX |
| honest: n6k1_knee_d1burst40 | F-rp | FX | time to tau, miss at the end | 281.5 | 217.1 | -64.4 [-125.7, -7.3] | 20 | 0.4597 | no claim |
| honest: n6k1_knee_d1ampharshpc | F-rp | FX | time to tau | 235.7 | 144.1 | -91.5 [-149.8, -44.3] | 20 | 0.0085 | FX |
| honest: n6k1_knee_d1ampharshpc | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_knee_d1ampharshpc | F-rp | FX | hover per mission | 22.5 | 54.6 | 32.1 [25.0, 39.3] | 20 | 0.0000 | F-rp |
| honest: n6k1_knee_d1ampharshpc | F-rp | FX | updates per mission | 3.08 | 3.54 | 0.46 [0.29, 0.64] | 20 | 0.0064 | FX |
| honest: n6k1_knee_d1ampharshpc | F-rp | FX | time to tau, miss at the end | 235.7 | 144.1 | -91.5 [-149.8, -44.3] | 20 | 0.0085 | FX |
| honest: n6k1_stress_d1ampharsh | F-rpnl | F-rp | time to tau | 159.4 | 165.9 | 6.5 [-3.1, 23.3] | 20 | 1.0000 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rpnl | F-rp | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rpnl | F-rp | hover per mission | 17.7 | 17.4 | -0.3 [-1.5, 0.8] | 20 | 0.7794 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rpnl | F-rp | updates per mission | 1.89 | 1.90 | 0.01 [-0.05, 0.07] | 20 | 1.0000 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rpnl | F-rp | time to tau, miss at the end | 159.4 | 165.9 | 6.5 [-3.1, 23.3] | 20 | 1.0000 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rp | FX | time to tau | 165.9 | 152.0 | -13.9 [-83.9, 61.3] | 20 | 1.0000 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| honest: n6k1_stress_d1ampharsh | F-rp | FX | hover per mission | 17.4 | 41.1 | 23.7 [17.6, 29.7] | 20 | 0.0000 | F-rp |
| honest: n6k1_stress_d1ampharsh | F-rp | FX | updates per mission | 1.90 | 3.08 | 1.18 [0.85, 1.48] | 20 | 0.0020 | FX |
| honest: n6k1_stress_d1ampharsh | F-rp | FX | time to tau, miss at the end | 165.9 | 152.0 | -13.9 [-83.9, 61.3] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_base | F-rp | FX | time to tau | 232.8 | 178.1 | -54.7 [-82.9, -30.3] | 20 | 0.0048 | FX |
| recorded: n6k1_knee_base | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_base | F-rp | FX | hover per mission | 23.6 | 88.2 | 64.5 [48.7, 80.3] | 20 | 0.0000 | F-rp |
| recorded: n6k1_knee_base | F-rp | FX | updates per mission | 3.50 | 3.54 | 0.04 [-0.01, 0.09] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_base | F-rp | FX | time to tau, miss at the end | 232.8 | 178.1 | -54.7 [-82.9, -30.3] | 20 | 0.0048 | FX |
| recorded: n6k1_knee_ampharsh | F-rp | FX | time to tau | 225.5 | 215.5 | -10.1 [-76.2, 58.4] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_ampharsh | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_ampharsh | F-rp | FX | hover per mission | 9.9 | 112.1 | 102.3 [79.7, 126.8] | 20 | 0.0000 | F-rp |
| recorded: n6k1_knee_ampharsh | F-rp | FX | updates per mission | 3.00 | 3.45 | 0.45 [0.25, 0.65] | 20 | 0.0175 | FX |
| recorded: n6k1_knee_ampharsh | F-rp | FX | time to tau, miss at the end | 225.5 | 215.5 | -10.1 [-76.2, 58.4] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_npl | F-rp | FX | time to tau | 224.1 | 210.8 | -13.3 [-43.7, 20.3] | 20 | 0.5013 | no claim |
| recorded: n6k1_knee_npl | F-rp | FX | reach | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_npl | F-rp | FX | hover per mission | 17.0 | 87.8 | 70.8 [52.7, 88.3] | 20 | 0.0000 | F-rp |
| recorded: n6k1_knee_npl | F-rp | FX | updates per mission | 3.51 | 3.52 | 0.01 [-0.03, 0.05] | 20 | 1.0000 | no claim |
| recorded: n6k1_knee_npl | F-rp | FX | time to tau, miss at the end | 224.1 | 210.8 | -13.3 [-43.7, 20.3] | 20 | 0.4597 | no claim |

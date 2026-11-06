# Exploratory: does changing band in flight save energy?

Not pre-registered (6 Nov 2026). `scripts/exp5/band_energy.py`, from the kept mule traces, 1 MB, one mule, paired by trial and seed against F, which flies the same plan on the plan's band. Difference = arm - F, so negative is a saving. Holm across each arm's six cells, per metric. Energy is simulated: 143.6 W moving and 168.5 W hovering on every band, with no radio power. Uplink drops ignore the band, so a faster band pays for a weaker link only through reach and the SNR floor.

Largest gap between a mission's recorded energy and the split below: 0.000%.

## Whole trial

| arm | devices, budget | pairs | stops off plan | F kJ per mission | arm | difference [95% CI] | Holm p | moving | hovering | kJ per merged update, difference [95% CI] | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FeRRy-Adapt | 6, knee | 40 | 71 of 174 | 22.65 | 20.06 | -2.59 [-3.71, -1.54] * | 0.0003 | -0.21 | -2.37 | -0.79 [-1.14, -0.49] * | 0.0003 |
| FeRRy-Adapt | 6, stress | 40 | 86 of 182 | 20.20 | 18.12 | -2.08 [-3.21, -1.03] * | 0.0038 | +0.05 | -2.13 | -0.74 [-1.19, -0.36] * | 0.0035 |
| FeRRy-Adapt | 12, knee | 40 | 107 of 252 | 35.83 | 34.10 | -1.74 [-2.97, -0.56] * | 0.0222 | -0.11 | -1.63 | -0.26 [-0.42, -0.11] * | 0.0066 |
| FeRRy-Adapt | 12, stress | 40 | 109 of 299 | 27.48 | 27.82 | +0.33 [-0.57, +1.21] | 0.4646 | +0.74 | -0.41 | -0.18 [-0.51, +0.11] | 0.5399 |
| FeRRy-Adapt | 24, knee | 20 | 102 of 315 | 45.62 | 48.11 | +2.49 [+1.38, +3.70] * | 0.0027 | +0.93 | +1.56 | +0.13 [+0.07, +0.20] * | 0.0035 |
| FeRRy-Adapt | 24, stress | 20 | 66 of 282 | 40.23 | 41.01 | +0.78 [-0.34, +1.96] | 0.4195 | -0.09 | +0.87 | +0.09 [-0.16, +0.38] | 0.5435 |
| FeRRy-RL | 6, knee | 40 | 73 of 174 | 22.65 | 20.51 | -2.14 [-3.26, -1.18] * | 0.0003 | -0.21 | -1.92 | -0.67 [-1.02, -0.37] * | 0.0003 |
| FeRRy-RL | 6, stress | 40 | 89 of 182 | 20.20 | 18.68 | -1.52 [-2.69, -0.46] * | 0.0450 | +0.04 | -1.56 | -0.61 [-1.08, -0.19] * | 0.0310 |
| FeRRy-RL | 12, knee | 40 | 119 of 253 | 35.83 | 34.68 | -1.15 [-2.20, -0.15] * | 0.1096 | +0.11 | -1.26 | -0.18 [-0.33, -0.04] * | 0.0679 |
| FeRRy-RL | 12, stress | 40 | 126 of 299 | 27.48 | 28.26 | +0.78 [-0.28, +1.90] | 0.3551 | +0.59 | +0.19 | -0.13 [-0.45, +0.16] | 0.5499 |
| FeRRy-RL | 24, knee | 20 | 176 of 314 | 45.62 | 53.62 | +8.00 [+6.79, +9.32] * | 0.0003 | +2.47 | +5.53 | +0.52 [+0.42, +0.64] * | 0.0003 |
| FeRRy-RL | 24, stress | 20 | 90 of 283 | 40.23 | 41.11 | +0.88 [-0.39, +2.33] | 0.3551 | -0.02 | +0.90 | +0.16 [-0.09, +0.45] | 0.5499 |

## Mission 1: one plan, band kept or left

| arm | devices, budget | group | pairs | F kJ | arm kJ | difference [95% CI] | p | largest difference, kJ | merged updates F, arm |
|---|---|---|---|---|---|---|---|---|---|
| FeRRy-Adapt | 6, knee | kept the band | 28 | 22.39 | 22.39 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 3.54, 3.54 |
| FeRRy-Adapt | 6, knee | left the band | 12 | 24.34 | 19.62 | -4.71 [-7.84, -2.05] | 0.0078 | 15.95 | 3.25, 3.25 |
| FeRRy-Adapt | 6, stress | kept the band | 23 | 19.63 | 19.63 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 3.30, 3.30 |
| FeRRy-Adapt | 6, stress | left the band | 17 | 21.42 | 17.46 | -3.95 [-6.58, -1.53] | 0.0065 | 15.13 | 3.06, 3.12 |
| FeRRy-Adapt | 12, knee | kept the band | 20 | 35.81 | 35.81 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 7.00, 7.00 |
| FeRRy-Adapt | 12, knee | left the band | 20 | 37.05 | 33.68 | -3.37 [-8.69, +1.27] | 0.2310 | 38.76 | 7.20, 7.25 |
| FeRRy-Adapt | 12, stress | kept the band | 17 | 26.50 | 26.61 | +0.11 [+0.00, +0.33] | 1.0000 | 1.86 | 5.88, 5.82 |
| FeRRy-Adapt | 12, stress | left the band | 23 | 26.66 | 28.93 | +2.27 [-0.30, +5.06] | 0.1098 | 18.30 | 5.17, 5.61 |
| FeRRy-Adapt | 24, knee | kept the band | 2 | 40.84 | 40.36 | -0.48 [-0.96, +0.00] | 1.0000 | 0.96 | 9.50, 9.50 |
| FeRRy-Adapt | 24, knee | left the band | 18 | 46.38 | 49.59 | +3.21 [+0.37, +5.96] | 0.0459 | 14.58 | 13.78, 14.11 |
| FeRRy-Adapt | 24, stress | kept the band | 8 | 38.64 | 38.64 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 11.88, 11.88 |
| FeRRy-Adapt | 24, stress | left the band | 12 | 39.05 | 40.85 | +1.80 [-0.80, +5.81] | 0.5034 | 21.60 | 11.08, 11.25 |
| FeRRy-RL | 6, knee | kept the band | 28 | 22.51 | 22.51 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 3.50, 3.50 |
| FeRRy-RL | 6, knee | left the band | 12 | 24.06 | 19.93 | -4.13 [-7.36, -1.22] | 0.0238 | 15.95 | 3.33, 3.33 |
| FeRRy-RL | 6, stress | kept the band | 21 | 20.14 | 20.14 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 3.48, 3.48 |
| FeRRy-RL | 6, stress | left the band | 19 | 20.66 | 18.14 | -2.51 [-5.03, -0.17] | 0.0614 | 15.13 | 2.89, 2.95 |
| FeRRy-RL | 12, knee | kept the band | 19 | 37.90 | 37.90 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 7.05, 7.05 |
| FeRRy-RL | 12, knee | left the band | 21 | 35.10 | 33.81 | -1.29 [-6.79, +3.01] | 0.6522 | 38.76 | 7.14, 7.19 |
| FeRRy-RL | 12, stress | kept the band | 14 | 26.54 | 26.54 | +0.01 [+0.00, +0.03] | 1.0000 | 0.13 | 5.86, 5.93 |
| FeRRy-RL | 12, stress | left the band | 26 | 26.62 | 28.85 | +2.22 [-0.09, +4.94] | 0.0943 | 18.30 | 5.27, 5.58 |
| FeRRy-RL | 24, knee | kept the band | 1 | 45.32 | 45.32 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 13.00, 13.00 |
| FeRRy-RL | 24, knee | left the band | 19 | 45.85 | 53.30 | +7.45 [+4.58, +10.53] | 0.0002 | 20.06 | 13.37, 13.63 |
| FeRRy-RL | 24, stress | kept the band | 5 | 37.59 | 37.59 | +0.00 [+0.00, +0.00] | 1.0000 | 0.00 | 11.00, 11.00 |
| FeRRy-RL | 24, stress | left the band | 15 | 39.32 | 41.37 | +2.05 [-0.30, +5.47] | 0.1757 | 21.60 | 11.53, 11.80 |

## First change in mission 1, against the same stop in F

| arm | devices | change | stops | seconds at the stop, plan band | chosen band | stop energy, kJ | arrival price, plan band, s | chosen band | price said faster | was faster |
|---|---|---|---|---|---|---|---|---|---|---|
| FeRRy-Adapt | 6 | medium to narrow | 2 | 29.4 | 47.8 | +3.11 | 124.4 | 51.3 | 2 | 0 |
| FeRRy-Adapt | 6 | medium to wide | 7 | 7.3 | 2.2 | -0.86 | 8.3 | 2.2 | 7 | 6 |
| FeRRy-Adapt | 6 | narrow to medium | 19 | 60.4 | 35.6 | -4.19 | 106.5 | 31.2 | 19 | 17 |
| FeRRy-Adapt | 6 | narrow to wide | 1 | 2.6 | 0.4 | -0.37 | 5.2 | 0.8 | 1 | 1 |
| FeRRy-Adapt | 12 | medium to narrow | 16 | 37.2 | 66.3 | +4.92 | 79.9 | 50.9 | 16 | 4 |
| FeRRy-Adapt | 12 | medium to wide | 13 | 10.3 | 4.6 | -0.97 | 16.6 | 6.2 | 13 | 13 |
| FeRRy-Adapt | 12 | narrow to medium | 8 | 148.1 | 76.1 | -12.13 | 227.5 | 69.8 | 8 | 8 |
| FeRRy-Adapt | 12 | wide to medium | 5 | 9.3 | 11.5 | +0.38 | 13.0 | 7.1 | 5 | 4 |
| FeRRy-Adapt | 24 | medium to narrow | 7 | 45.3 | 87.8 | +7.16 | 124.4 | 85.9 | 7 | 1 |
| FeRRy-Adapt | 24 | medium to wide | 7 | 12.6 | 4.4 | -1.37 | 20.7 | 5.8 | 7 | 7 |
| FeRRy-Adapt | 24 | wide to medium | 11 | 8.3 | 6.0 | -0.39 | 9.0 | 6.7 | 11 | 10 |
| FeRRy-RL | 6 | medium to narrow | 7 | 23.9 | 30.3 | +1.08 | 48.2 | 34.0 | 3 | 3 |
| FeRRy-RL | 6 | medium to wide | 6 | 5.3 | 2.5 | -0.47 | 5.8 | 2.9 | 5 | 4 |
| FeRRy-RL | 6 | narrow to medium | 17 | 64.4 | 40.6 | -4.01 | 106.2 | 37.1 | 13 | 14 |
| FeRRy-RL | 6 | narrow to wide | 1 | 2.6 | 0.4 | -0.37 | 5.2 | 0.8 | 1 | 1 |
| FeRRy-RL | 12 | medium to narrow | 22 | 22.7 | 48.3 | +4.32 | 49.5 | 39.8 | 13 | 4 |
| FeRRy-RL | 12 | medium to wide | 12 | 13.5 | 7.9 | -0.94 | 18.3 | 10.9 | 8 | 9 |
| FeRRy-RL | 12 | narrow to medium | 8 | 128.4 | 74.7 | -9.06 | 225.0 | 74.4 | 7 | 8 |
| FeRRy-RL | 12 | wide to medium | 3 | 6.4 | 3.7 | -0.45 | 7.8 | 5.2 | 3 | 3 |
| FeRRy-RL | 12 | wide to narrow | 2 | 0.6 | 7.1 | +1.11 | 0.7 | 9.0 | 0 | 0 |
| FeRRy-RL | 24 | medium to narrow | 15 | 24.4 | 55.6 | +5.26 | 63.3 | 58.8 | 7 | 0 |
| FeRRy-RL | 24 | medium to wide | 4 | 9.6 | 6.6 | -0.50 | 10.6 | 5.5 | 3 | 3 |
| FeRRy-RL | 24 | wide to medium | 10 | 8.4 | 6.2 | -0.37 | 8.2 | 7.4 | 7 | 8 |
| FeRRy-RL | 24 | wide to narrow | 4 | 2.6 | 7.2 | +0.77 | 3.6 | 10.3 | 0 | 0 |

## First change, all device counts

| arm | devices | change | stops | seconds at the stop, plan band | chosen band | stop energy, kJ | arrival price, plan band, s | chosen band | price said faster | was faster |
|---|---|---|---|---|---|---|---|---|---|---|
| FeRRy-Adapt | all | medium to narrow | 25 | 38.8 | 70.9 | +5.40 | 95.9 | 60.7 | 25 | 5 |
| FeRRy-Adapt | all | medium to wide | 27 | 10.1 | 3.9 | -1.05 | 15.5 | 5.1 | 27 | 26 |
| FeRRy-Adapt | all | narrow to medium | 27 | 86.4 | 47.6 | -6.54 | 142.3 | 42.7 | 27 | 25 |
| FeRRy-Adapt | all | narrow to wide | 1 | 2.6 | 0.4 | -0.37 | 5.2 | 0.8 | 1 | 1 |
| FeRRy-Adapt | all | wide to medium | 16 | 8.6 | 7.7 | -0.15 | 10.3 | 6.8 | 16 | 14 |
| FeRRy-RL | all | medium to narrow | 44 | 23.5 | 47.9 | +4.12 | 54.0 | 45.4 | 23 | 7 |
| FeRRy-RL | all | medium to wide | 22 | 10.5 | 6.2 | -0.73 | 13.5 | 7.7 | 16 | 16 |
| FeRRy-RL | all | narrow to medium | 25 | 84.9 | 51.5 | -5.63 | 144.2 | 49.1 | 20 | 22 |
| FeRRy-RL | all | narrow to wide | 1 | 2.6 | 0.4 | -0.37 | 5.2 | 0.8 | 1 | 1 |
| FeRRy-RL | all | wide to medium | 13 | 7.9 | 5.6 | -0.39 | 8.1 | 6.9 | 10 | 11 |
| FeRRy-RL | all | wide to narrow | 6 | 1.9 | 7.2 | +0.89 | 2.6 | 9.9 | 0 | 0 |

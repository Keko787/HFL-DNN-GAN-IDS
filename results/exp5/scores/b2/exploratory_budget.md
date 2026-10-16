# Exploratory: time to tau as the mission budget grows (N = 6, one mule)

Not pre-registered as a sweep (16 Oct 2026). Stress and knee as scored for 5.3 (their pre-registered comparisons are in s53x.md); relaxed = 300 s (5.3's third budget, 2x knee) and 1800 s (12x knee), flown on 16 Oct on the same 20 seeds. Mean time to tau = 0.71 over the trials that reached it (simulated seconds).

## Time to tau by budget

| arm | 75 s (stress) | 150 s (knee) | 300 s (relaxed) | 1800 s (relaxed) |
|---|---|---|---|---|
| F | 178.0 (20/20) | 194.0 (20/20) | 194.0 (20/20) | 194.0 (20/20) |
| FX | 175.3 (20/20) | 178.1 (20/20) | 178.1 (20/20) | 178.1 (20/20) |
| FQ | 181.6 (20/20) | 180.3 (20/20) | 179.3 (20/20) | 179.3 (20/20) |
| H1 | 220.4 (18/20) | 281.8 (20/20) | 286.0 (20/20) | 286.0 (20/20) |
| MAX-AoI | 238.5 (20/20) | 281.8 (20/20) | 286.0 (20/20) | 286.0 (20/20) |
| Oort | 228.8 (18/20) | 265.8 (20/20) | 267.2 (20/20) | 267.2 (20/20) |
| Whittle (Cui) | 209.0 (20/20) | 279.6 (20/20) | 281.0 (20/20) | 281.0 (20/20) |
| FedEx route | 264.5 (20/20) | 264.5 (20/20) | 264.5 (20/20) | 264.5 (20/20) |
| FedCS | 197.6 (19/20) | 274.6 (20/20) | 261.0 (20/20) | 261.0 (20/20) |
| DQN (Chen) | 211.1 (19/20) | 279.9 (20/20) | 289.7 (20/20) | 289.9 (20/20) |
| FedEx route + merge | 280.6 (19/20) | 280.6 (19/20) | 280.6 (19/20) | 280.6 (19/20) |

## F against each arm at the relaxed budgets

Difference = F - arm (negative: F faster); Holm across each budget's ten.

### 300 s: F faster than 7 of 7 baselines

| reference | arm | n pairs | F mean | arm mean | difference [95% CI] | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|
| F | FX | 20 | 194.0 | 178.1 | 16.0 [3.9, 31.8] | 0.0077 | 0.0231 | FX |
| F | FQ | 20 | 194.0 | 179.3 | 14.8 [1.0, 32.2] | 0.0745 | 0.0745 | no claim |
| F | H1 | 20 | 194.0 | 286.0 | -92.0 [-134.7, -55.9] | 0.0000 | 0.0003 | F |
| F | MAX-AoI | 20 | 194.0 | 286.0 | -92.0 [-135.4, -55.4] | 0.0000 | 0.0003 | F |
| F | Oort | 20 | 194.0 | 267.2 | -73.2 [-113.8, -43.8] | 0.0001 | 0.0005 | F |
| F | Whittle (Cui) | 20 | 194.0 | 281.0 | -86.9 [-129.1, -50.3] | 0.0000 | 0.0003 | F |
| F | FedEx route | 20 | 194.0 | 264.5 | -70.4 [-111.7, -37.2] | 0.0004 | 0.0016 | F |
| F | FedCS | 20 | 194.0 | 261.0 | -67.0 [-104.6, -37.1] | 0.0001 | 0.0007 | F |
| F | DQN (Chen) | 20 | 194.0 | 289.7 | -95.7 [-142.7, -57.8] | 0.0000 | 0.0003 | F |
| F | FedEx route + merge | 19 | 195.1 | 280.6 | -85.5 [-171.3, -7.6] | 0.0230 | 0.0460 | F |

### 1800 s: F faster than 7 of 7 baselines

| reference | arm | n pairs | F mean | arm mean | difference [95% CI] | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|
| F | FX | 20 | 194.0 | 178.1 | 16.0 [3.9, 31.8] | 0.0077 | 0.0231 | FX |
| F | FQ | 20 | 194.0 | 179.3 | 14.8 [1.0, 32.2] | 0.0745 | 0.0745 | no claim |
| F | H1 | 20 | 194.0 | 286.0 | -92.0 [-134.7, -55.9] | 0.0000 | 0.0004 | F |
| F | MAX-AoI | 20 | 194.0 | 286.0 | -92.0 [-135.4, -55.4] | 0.0000 | 0.0004 | F |
| F | Oort | 20 | 194.0 | 267.2 | -73.2 [-113.8, -43.8] | 0.0001 | 0.0005 | F |
| F | Whittle (Cui) | 20 | 194.0 | 281.0 | -86.9 [-129.1, -50.3] | 0.0000 | 0.0004 | F |
| F | FedEx route | 20 | 194.0 | 264.5 | -70.4 [-111.7, -37.2] | 0.0004 | 0.0016 | F |
| F | FedCS | 20 | 194.0 | 261.0 | -67.0 [-104.6, -37.1] | 0.0001 | 0.0007 | F |
| F | DQN (Chen) | 20 | 194.0 | 289.9 | -95.9 [-140.4, -58.0] | 0.0001 | 0.0004 | F |
| F | FedEx route + merge | 19 | 195.1 | 280.6 | -85.5 [-171.3, -7.6] | 0.0230 | 0.0460 | F |

## Same seeds, same result?

Trials whose time to tau is identical between two budgets (same seeds per cell and trial).

| arm | 150 s = 300 s | 300 s = 1800 s |
|---|---|---|
| F | 20 of 20 | 20 of 20 |
| FX | 20 of 20 | 20 of 20 |
| FQ | 17 of 20 | 20 of 20 |
| H1 | 18 of 20 | 20 of 20 |
| MAX-AoI | 18 of 20 | 20 of 20 |
| Oort | 19 of 20 | 20 of 20 |
| Whittle (Cui) | 19 of 20 | 20 of 20 |
| FedEx route | 20 of 20 | 20 of 20 |
| FedCS | 19 of 20 | 20 of 20 |
| DQN (Chen) | 9 of 20 | 5 of 20 |
| FedEx route + merge | 20 of 20 | 20 of 20 |


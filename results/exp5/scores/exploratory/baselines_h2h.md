# The baselines head to head

Exploratory, not pre-registered (15 Nov 2026; `scripts/exp5/baseline_h2h.py`). No flights: the scored trial files of the campaign and the loss probes. FeRRy's arms are left out. In each scenario the fastest baseline (the lowest mean time to tau = 0.71 over its own reached trials) is compared with every other baseline: time to tau by the scorer's paired comparison over the trials both reach, and reach at tau by exact McNemar, each with a 95% bootstrap CI. Holm within each scenario, one family per metric. A claim needs the CI to exclude 0 and Holm p < 0.05. Difference = fastest − arm: time, negative when the fastest is faster; reach, positive when the fastest reaches tau more often. When a time claim favoured another arm over the one with the lowest mean, that arm is the fastest.

## Each baseline

Over the scenarios it flew in: fastest; level with the fastest (no time claim against it); significantly slower than the fastest; last by mean.

| baseline | scenarios flown | fastest | level | significantly slower | last |
|---|---|---|---|---|---|
| FedEx (D4) | 39 | 21 | 15 | 3 | 4 |
| Model-Aided RL (E3) | 8 | 1 | 6 | 1 | 1 |
| FedCS (D5) | 34 | 8 | 19 | 7 | 3 |
| FedEx-Merge (D4fedex) | 8 | 0 | 6 | 2 | 7 |
| Oort (D2) | 31 | 1 | 21 | 9 | 5 |
| Value-Aware FL (D3) | 42 | 4 | 24 | 14 | 12 |
| Deadline-Heuristic (H1) | 42 | 7 | 19 | 16 | 8 |
| MAX-AoI (D1) | 31 | 0 | 17 | 14 | 2 |

## The fastest baseline in each scenario

| group | scenario | trials | fastest: mean time to tau (s), reach | significantly slower than it (fastest − arm, s) |
|---|---|---|---|---|
| Budget, N = 6, 1 mule | knee 150 s | 40 | FedEx (D4): 232, 1.00 | MAX-AoI (D1) (-31) |
| Budget, N = 6, 1 mule | stress 75 s | 40 | Model-Aided RL (E3): 201, 0.97 | none |
| Budget, N = 6, 1 mule | 60 s | 20 | FedCS (D5): 175, 0.95 | none |
| Budget, N = 6, 1 mule | 45 s | 20 | FedCS (D5): 171, 1.00 | FedEx (D4) (-94) |
| Budget, N = 6, 1 mule | 300 s | 40 | FedCS (D5): 229, 1.00 | Deadline-Heuristic (H1) (-18), MAX-AoI (D1) (-18), Model-Aided RL (E3) (-19) |
| Budget, N = 6, 1 mule | 1800 s | 20 | FedCS (D5): 261, 1.00 | Deadline-Heuristic (H1) (-25), MAX-AoI (D1) (-25) |
| Mules | N = 6, 3 mules, knee | 40 | FedEx (D4): 60, 1.00 | FedEx-Merge (D4fedex) (-51) |
| Mules | N = 6, 3 mules, stress | 40 | Deadline-Heuristic (H1): 60, 1.00 | FedEx-Merge (D4fedex) (-51) |
| Mules | N = 6, 2 mules, knee | 20 | Value-Aware FL (D3): 97, 1.00 | none |
| Mules | N = 12, 2 mules, knee | 20 | FedEx (D4): 222, 1.00 | Deadline-Heuristic (H1) (-66), Value-Aware FL (D3) (-34) |
| Mules | N = 12, 3 mules, knee | 20 | FedEx (D4): 128, 1.00 | none |
| Mules | N = 24, 2 mules, knee | 20 | FedEx (D4): 319, 1.00 | Deadline-Heuristic (H1) (-43), Value-Aware FL (D3) (-41) |
| Mules | N = 24, 3 mules, knee | 20 | FedEx (D4): 222, 1.00 | none |
| Scale, 1 mule | N = 12, knee | 40 | FedEx (D4): 463, 0.93 | none |
| Scale, 1 mule | N = 12, stress | 20 | Oort (D2): 367, 0.85 | none |
| Scale, 1 mule | N = 24, knee | 20 | FedEx (D4): 752, 1.00 | Deadline-Heuristic (H1) (-199), Value-Aware FL (D3) (-159), MAX-AoI (D1) (-196), Oort (D2) (-124), FedCS (D5) (-19) |
| Scale, 1 mule | N = 24, stress | 20 | Deadline-Heuristic (H1): 651, 0.95 | FedEx (D4) (-112) |
| Radio, time-varying backhaul | default 5 dB | 20 | FedEx (D4): 257, 1.00 | Deadline-Heuristic (H1) (-16), MAX-AoI (D1) (-16), FedCS (D5) (-53) |
| Radio, time-varying backhaul | 11 dB | 20 | FedCS (D5): 288, 0.95 | none |
| Radio, time-varying backhaul | 17 dB | 20 | Value-Aware FL (D3): 264, 0.90 | none |
| Radio, time-varying backhaul | path loss 2.7 | 20 | FedEx (D4): 237, 0.90 | Deadline-Heuristic (H1) (-17), MAX-AoI (D1) (-16), Oort (D2) (-17), Value-Aware FL (D3) (-62), FedCS (D5) (-56) |
| Radio, time-varying backhaul | shadowing 8 dB | 20 | FedEx (D4): 262, 1.00 | Deadline-Heuristic (H1) (-19), MAX-AoI (D1) (-18), FedCS (D5) (-37) |
| Radio, mission backhaul | default 5 dB | 20 | FedEx (D4): 264, 1.00 | Deadline-Heuristic (H1) (-17), MAX-AoI (D1) (-17) |
| Radio, mission backhaul | 17 dB | 20 | FedEx (D4): 244, 1.00 | none |
| Radio, mission backhaul | path loss 2.7 | 20 | FedEx (D4): 243, 1.00 | Deadline-Heuristic (H1) (-26), MAX-AoI (D1) (-26), Oort (D2) (-19), Value-Aware FL (D3) (-32), FedCS (D5) (-19) |
| Payload, N = 6 knee | 2 MB | 20 | FedEx (D4): 271, 1.00 | Deadline-Heuristic (H1) (-31), MAX-AoI (D1) (-30), Oort (D2) (-34), Value-Aware FL (D3) (-54) |
| Payload, N = 6 knee | 5 MB | 20 | FedCS (D5): 276, 1.00 | none |
| Payload, N = 6 knee | 10 MB | 20 | FedCS (D5): 315, 1.00 | none |
| Layout, N = 6 knee | 25 % beyond 60 m | 20 | FedEx (D4): 221, 1.00 | Oort (D2) (-10), Value-Aware FL (D3) (-12) |
| Layout, N = 6 knee | 50 % beyond 60 m | 20 | Deadline-Heuristic (H1): 190, 1.00 | none |
| Layout, N = 6 knee | 75 % beyond 60 m | 20 | Value-Aware FL (D3): 245, 1.00 | none |
| Field, N = 12 knee | ±200 m, 1 MB | 20 | FedEx (D4): 1124, 0.95 | Deadline-Heuristic (H1) (-361), Value-Aware FL (D3) (-256), MAX-AoI (D1) (-361), Oort (D2) (-360), FedCS (D5) (-42) |
| Field, N = 12 knee | ±300 m, 1 MB | 20 | FedEx (D4): 1513, 0.90 | Deadline-Heuristic (H1) (-686), Value-Aware FL (D3) (-580), MAX-AoI (D1) (-715), Oort (D2) (-851) |
| Field, N = 12 knee | ±200 m, 10 MB | 20 | FedEx (D4): 1381, 0.90 | Deadline-Heuristic (H1) (-387), MAX-AoI (D1) (-396), Oort (D2) (-346), Value-Aware FL (D3) (-288), FedCS (D5) (-51) |
| Field, N = 12 knee | ±300 m, 10 MB | 20 | FedEx (D4): 1912, 0.90 | Deadline-Heuristic (H1) (-955), MAX-AoI (D1) (-963), Oort (D2) (-547), Value-Aware FL (D3) (-471) |
| Field | N = 24, ±300 m, 10 MB, 600 s | 20 | Deadline-Heuristic (H1): 2938, 0.90 | FedEx (D4) (-696) |
| Training time, 10 missions | spread | 40 | Deadline-Heuristic (H1): 276, 1.00 | none |
| Training time, 10 missions | stragglers | 40 | FedCS (D5): 270, 1.00 | Value-Aware FL (D3) (-65) |
| Data, N = 6 knee | IID | 20 | Value-Aware FL (D3): 280, 1.00 | none |
| Data, N = 6 knee | skewed shard sizes | 20 | Deadline-Heuristic (H1): 174, 1.00 | none |
| Data, N = 6 knee | Dirichlet 1 | 20 | Deadline-Heuristic (H1): 382, 0.65 | Value-Aware FL (D3) (-31) |
| Data, N = 6 knee | Dirichlet 0.1 | 20 | FedEx (D4): 471, 0.50 | Deadline-Heuristic (H1) (-34), Value-Aware FL (D3) (-49) |

## Budget, N = 6, 1 mule: knee 150 s

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 40 | 1.00 | 232.1 |  |  |  |  |  |  |  |  |
| Oort (D2) | 40 | 1.00 | 235.2 | -3.1 [-17.7, 15.7] | 40 | 0.0011 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| FedCS (D5) | 40 | 1.00 | 235.7 | -3.6 [-26.7, 19.2] | 40 | 0.1266 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Model-Aided RL (E3) | 40 | 1.00 | 252.1 | -20.0 [-50.0, 3.6] | 40 | 0.0007 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by E3 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 40 | 1.00 | 254.3 | -22.2 [-54.2, 0.4] | 40 | 0.0002 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 262.1 | -30.0 [-69.1, 0.4] | 40 | 0.0003 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 40 | 1.00 | 262.7 | -30.6 [-68.6, -3.7] | 40 | 0.0002 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 40 | 0.93 | 295.7 | -76.2 [-153.6, -1.6] | 37 | 0.1266 | no claim | 0.07 [0.00, 0.15] | 3 reached by D4 only, 0 by D4fedex only | 1.0000 | no claim |

## Budget, N = 6, 1 mule: stress 75 s

Fastest: Model-Aided RL (E3).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Model-Aided RL (E3) | 40 | 0.97 | 200.8 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 40 | 0.97 | 206.4 | -5.6 [-38.4, 23.1] | 39 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by E3 only, 0 by D5 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 40 | 0.93 | 209.0 | -14.2 [-60.2, 30.4] | 36 | 1.0000 | no claim | 0.05 [-0.05, 0.15] | 3 reached by E3 only, 1 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 212.8 | -9.1 [-55.2, 35.1] | 39 | 1.0000 | no claim | -0.03 [-0.07, 0.00] | 0 reached by E3 only, 1 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 40 | 1.00 | 231.7 | -27.3 [-65.6, 1.9] | 39 | 1.0000 | no claim | -0.03 [-0.07, 0.00] | 0 reached by E3 only, 1 by D1 only | 1.0000 | no claim |
| FedEx (D4) | 40 | 1.00 | 232.1 | -27.6 [-83.9, 20.6] | 39 | 1.0000 | no claim | -0.03 [-0.07, 0.00] | 0 reached by E3 only, 1 by D4 only | 1.0000 | no claim |
| Oort (D2) | 40 | 0.95 | 236.3 | -37.8 [-78.6, 0.7] | 37 | 0.5136 | no claim | 0.03 [-0.05, 0.10] | 2 reached by E3 only, 1 by D2 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 40 | 0.93 | 295.7 | -102.3 [-177.6, -29.4] | 36 | 0.0676 | no claim | 0.05 [-0.05, 0.15] | 3 reached by E3 only, 1 by D4fedex only | 1.0000 | no claim |

## Budget, N = 6, 1 mule: 60 s

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 20 | 0.95 | 175.0 |  |  |  |  |  |  |  |  |
| Oort (D2) | 20 | 0.85 | 175.3 | -12.7 [-59.5, 36.9] | 17 | 0.6686 | no claim | 0.10 [0.00, 0.25] | 2 reached by D5 only, 0 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.95 | 205.4 | -30.4 [-74.0, 9.9] | 19 | 0.6681 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| Model-Aided RL (E3) | 20 | 0.95 | 208.1 | -33.1 [-65.5, -6.7] | 19 | 0.1528 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by E3 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.90 | 209.7 | -43.3 [-132.3, 40.7] | 17 | 0.6686 | no claim | 0.05 [-0.10, 0.20] | 2 reached by D5 only, 1 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.95 | 232.4 | -57.4 [-134.3, 6.3] | 19 | 0.6536 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D3 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 264.5 | -83.6 [-183.2, -3.4] | 19 | 0.6681 | no claim | -0.05 [-0.15, 0.00] | 0 reached by D5 only, 1 by D4 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 20 | 0.95 | 280.6 | -102.0 [-206.3, -7.7] | 18 | 0.6681 | no claim | 0.00 [-0.15, 0.15] | 1 reached by D5 only, 1 by D4fedex only | 1.0000 | no claim |

## Budget, N = 6, 1 mule: 45 s

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 20 | 1.00 | 170.7 |  |  |  |  |  |  |  |  |
| Model-Aided RL (E3) | 20 | 1.00 | 174.3 | -3.6 [-35.3, 27.1] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by E3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.95 | 181.2 | -12.5 [-47.0, 22.4] | 19 | 1.0000 | no claim | 0.05 [0.00, 0.15] | 1 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.95 | 184.1 | -22.4 [-75.6, 25.0] | 19 | 0.8535 | no claim | 0.05 [0.00, 0.15] | 1 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.95 | 198.0 | -36.3 [-82.3, 7.7] | 19 | 0.3457 | no claim | 0.05 [0.00, 0.15] | 1 reached by D5 only, 0 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 210.7 | -49.0 [-102.8, -0.3] | 19 | 0.4667 | no claim | 0.05 [0.00, 0.15] | 1 reached by D5 only, 0 by D2 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 264.5 | -93.8 [-150.4, -43.7] | 20 | 0.0256 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D4 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 20 | 0.95 | 280.6 | -118.9 [-197.1, -47.4] | 19 | 0.0650 | no claim | 0.05 [0.00, 0.15] | 1 reached by D5 only, 0 by D4fedex only | 1.0000 | no claim |

## Budget, N = 6, 1 mule: 300 s

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 40 | 1.00 | 228.9 |  |  |  |  |  |  |  |  |
| FedEx (D4) | 40 | 1.00 | 232.1 | -3.2 [-21.6, 12.9] | 40 | 0.1715 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D4 only | 1.0000 | no claim |
| Oort (D2) | 40 | 1.00 | 238.3 | -9.4 [-21.8, 7.2] | 40 | 0.0009 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D2 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 244.1 | -15.2 [-32.2, 3.4] | 40 | 0.0005 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 40 | 1.00 | 246.7 | -17.8 [-32.6, -4.4] | 40 | 0.0004 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 40 | 1.00 | 246.7 | -17.8 [-32.8, -4.4] | 40 | 0.0003 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| Model-Aided RL (E3) | 40 | 1.00 | 248.3 | -19.4 [-36.9, -2.0] | 40 | 0.0005 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by E3 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 40 | 0.93 | 295.7 | -74.6 [-150.2, -3.4] | 37 | 0.1715 | no claim | 0.07 [0.00, 0.15] | 3 reached by D5 only, 0 by D4fedex only | 1.0000 | no claim |

## Budget, N = 6, 1 mule: 1800 s

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 20 | 1.00 | 261.0 |  |  |  |  |  |  |  |  |
| FedEx (D4) | 20 | 1.00 | 264.5 | -3.4 [-25.4, 9.9] | 20 | 0.3829 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D4 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 267.2 | -6.2 [-22.8, 20.4] | 20 | 0.0747 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D2 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 20 | 0.95 | 280.6 | -19.1 [-119.3, 71.1] | 19 | 0.9588 | no claim | 0.05 [0.00, 0.15] | 1 reached by D5 only, 0 by D4fedex only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 281.0 | -19.9 [-47.8, 12.6] | 20 | 0.0260 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 286.0 | -25.0 [-49.0, -9.6] | 20 | 0.0140 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 286.0 | -25.0 [-49.6, -9.8] | 20 | 0.0086 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| Model-Aided RL (E3) | 20 | 1.00 | 289.9 | -28.9 [-58.7, 5.3] | 20 | 0.0145 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by E3 only | 1.0000 | no claim |

## Mules: N = 6, 3 mules, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 40 | 1.00 | 59.6 |  |  |  |  |  |  |  |  |
| Deadline-Heuristic (H1) | 40 | 1.00 | 59.9 | -0.2 [-7.8, 7.9] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI (D1) | 40 | 1.00 | 59.9 | -0.2 [-7.8, 7.9] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| FedCS (D5) | 40 | 1.00 | 59.9 | -0.2 [-7.8, 7.9] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Model-Aided RL (E3) | 40 | 1.00 | 62.9 | -3.3 [-10.4, 3.0] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by E3 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 63.0 | -3.4 [-10.5, 3.0] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| Oort (D2) | 40 | 1.00 | 63.9 | -4.2 [-11.0, 1.6] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 40 | 1.00 | 110.8 | -51.1 [-69.7, -34.8] | 40 | 0.0000 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D4fedex only | 1.0000 | no claim |

## Mules: N = 6, 3 mules, stress

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 40 | 1.00 | 59.6 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 40 | 1.00 | 59.6 | 0.0 [0.0, 0.0] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D1 only | 1.0000 | no claim |
| FedCS (D5) | 40 | 1.00 | 59.6 | 0.0 [0.0, 0.0] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D5 only | 1.0000 | no claim |
| FedEx (D4) | 40 | 1.00 | 59.6 | -0.1 [-8.3, 7.6] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D4 only | 1.0000 | no claim |
| Model-Aided RL (E3) | 40 | 1.00 | 62.6 | -3.0 [-9.1, 0.0] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by E3 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 62.7 | -3.1 [-9.5, 0.1] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D3 only | 1.0000 | no claim |
| Oort (D2) | 40 | 1.00 | 63.6 | -4.0 [-11.9, 0.0] | 40 | 0.8281 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D2 only | 1.0000 | no claim |
| FedEx-Merge (D4fedex) | 40 | 1.00 | 110.8 | -51.2 [-68.6, -35.8] | 40 | 0.0000 | Deadline-Heuristic (H1) | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D4fedex only | 1.0000 | no claim |

## Mules: N = 6, 2 mules, knee

Fastest: Value-Aware FL (D3).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Value-Aware FL (D3) | 20 | 1.00 | 96.8 |  |  |  |  |  |  |  |  |
| Deadline-Heuristic (H1) | 20 | 1.00 | 96.8 | -0.0 [-2.6, 2.4] | 20 | 0.8785 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by H1 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 116.8 | -20.0 [-55.5, 14.1] | 20 | 0.5737 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by D4 only | 1.0000 | no claim |

## Mules: N = 12, 2 mules, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 221.6 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 1.00 | 255.4 | -33.8 [-66.8, -1.8] | 20 | 0.0189 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.95 | 262.8 | -66.1 [-128.8, -21.7] | 19 | 0.0189 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by H1 only | 1.0000 | no claim |

## Mules: N = 12, 3 mules, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 127.8 |  |  |  |  |  |  |  |  |
| Deadline-Heuristic (H1) | 20 | 1.00 | 164.8 | -37.0 [-104.3, 10.6] | 20 | 0.3139 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 164.9 | -37.1 [-108.7, 11.0] | 20 | 0.3139 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Mules: N = 24, 2 mules, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 319.2 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 1.00 | 360.2 | -40.9 [-64.2, -17.8] | 20 | 0.0146 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 362.0 | -42.8 [-73.1, -16.0] | 20 | 0.0146 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |

## Mules: N = 24, 3 mules, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 221.8 |  |  |  |  |  |  |  |  |
| Deadline-Heuristic (H1) | 20 | 1.00 | 221.9 | -0.1 [-20.9, 22.9] | 20 | 0.6742 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 248.4 | -26.6 [-80.9, 13.2] | 20 | 0.6236 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Scale, 1 mule: N = 12, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 40 | 0.93 | 462.9 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 0.95 | 499.9 | -41.3 [-120.7, 20.0] | 19 | 0.0721 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.95 | 506.7 | -58.0 [-126.7, 10.5] | 18 | 0.0097 | no claim | 0.00 [-0.15, 0.15] | 1 reached by D4 only, 1 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.90 | 529.9 | -61.6 [-170.8, 34.5] | 18 | 0.0637 | no claim | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.95 | 579.5 | -120.9 [-264.9, 3.8] | 19 | 0.0721 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.95 | 589.3 | -130.7 [-272.6, -2.1] | 19 | 0.0690 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |

## Scale, 1 mule: N = 12, stress

Fastest: Oort (D2).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Oort (D2) | 20 | 0.85 | 367.2 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 20 | 0.95 | 431.9 | -33.8 [-179.2, 101.4] | 16 | 1.0000 | no claim | -0.10 [-0.30, 0.10] | 1 reached by D2 only, 3 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.95 | 446.0 | -39.8 [-148.2, 66.7] | 16 | 1.0000 | no claim | -0.10 [-0.30, 0.10] | 1 reached by D2 only, 3 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.90 | 460.5 | -54.1 [-159.6, 46.1] | 16 | 1.0000 | no claim | -0.05 [-0.20, 0.10] | 1 reached by D2 only, 2 by D3 only | 1.0000 | no claim |
| FedEx (D4) | 40 | 0.93 | 462.9 | -82.1 [-184.2, 21.0] | 17 | 0.4433 | no claim | -0.10 [-0.25, 0.00] | 0 reached by D2 only, 2 by D4 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 0.90 | 515.1 | -141.4 [-298.9, 2.0] | 15 | 1.0000 | no claim | -0.05 [-0.25, 0.15] | 2 reached by D2 only, 3 by D5 only | 1.0000 | no claim |

## Scale, 1 mule: N = 24, knee

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 752.2 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 1.00 | 771.4 | -19.2 [-33.0, -4.5] | 20 | 0.0192 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 876.1 | -123.9 [-181.6, -58.2] | 20 | 0.0112 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 910.9 | -158.7 [-202.3, -124.7] | 20 | 0.0000 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 948.3 | -196.1 [-250.4, -155.0] | 20 | 0.0000 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 951.1 | -198.8 [-253.9, -157.4] | 20 | 0.0000 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |

## Scale, 1 mule: N = 24, stress

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 20 | 0.95 | 651.2 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 0.95 | 661.6 | -23.8 [-110.2, 56.1] | 18 | 1.0000 | no claim | 0.00 [-0.15, 0.15] | 1 reached by H1 only, 1 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 682.2 | -27.8 [-94.2, 37.1] | 19 | 1.0000 | no claim | -0.05 [-0.15, 0.00] | 0 reached by H1 only, 1 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.95 | 691.9 | -40.6 [-103.6, 15.2] | 19 | 0.3330 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D1 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 752.2 | -112.1 [-174.7, -45.8] | 19 | 0.0412 | Deadline-Heuristic (H1) | -0.05 [-0.15, 0.00] | 0 reached by H1 only, 1 by D4 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 0.85 | 754.2 | -115.7 [-209.1, -35.9] | 16 | 0.3330 | no claim | 0.10 [-0.10, 0.30] | 3 reached by H1 only, 1 by D5 only | 1.0000 | no claim |

## Radio, time-varying backhaul: default 5 dB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 257.2 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 20 | 1.00 | 273.0 | -15.7 [-22.6, -9.5] | 20 | 0.0042 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 273.6 | -16.4 [-23.1, -10.1] | 20 | 0.0042 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 281.5 | -24.6 [-68.3, 12.3] | 19 | 0.0126 | no claim | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 293.2 | -36.0 [-91.6, 6.5] | 20 | 0.0126 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 0.95 | 309.4 | -52.6 [-126.5, -1.7] | 19 | 0.0480 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D5 only | 1.0000 | no claim |

## Radio, time-varying backhaul: 11 dB

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 20 | 0.95 | 287.7 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 20 | 0.95 | 293.4 | -5.7 [-75.3, 57.4] | 19 | 0.5189 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 313.7 | -13.2 [-83.2, 48.7] | 19 | 0.6637 | no claim | -0.05 [-0.15, 0.00] | 0 reached by D5 only, 1 by D4 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 314.2 | -21.9 [-57.4, 14.1] | 19 | 0.1733 | no claim | -0.05 [-0.15, 0.00] | 0 reached by D5 only, 1 by D3 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.95 | 320.7 | -33.0 [-100.8, 27.4] | 19 | 0.1856 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 324.4 | -31.0 [-161.3, 72.9] | 18 | 0.6637 | no claim | 0.00 [-0.15, 0.15] | 1 reached by D5 only, 1 by D2 only | 1.0000 | no claim |

## Radio, time-varying backhaul: 17 dB

Fastest: Value-Aware FL (D3).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Value-Aware FL (D3) | 20 | 0.90 | 263.7 |  |  |  |  |  |  |  |  |
| FedEx (D4) | 20 | 1.00 | 276.3 | -24.7 [-111.3, 64.1] | 18 | 1.0000 | no claim | -0.10 [-0.25, 0.00] | 0 reached by D3 only, 2 by D4 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 295.1 | -42.1 [-102.0, 11.9] | 18 | 1.0000 | no claim | -0.05 [-0.15, 0.00] | 0 reached by D3 only, 1 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 310.5 | -54.2 [-142.3, 15.5] | 18 | 1.0000 | no claim | -0.10 [-0.25, 0.00] | 0 reached by D3 only, 2 by D1 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 0.95 | 314.8 | -63.7 [-139.1, 6.0] | 18 | 1.0000 | no claim | -0.05 [-0.15, 0.00] | 0 reached by D3 only, 1 by D5 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 332.1 | -78.1 [-183.4, 9.1] | 18 | 1.0000 | no claim | -0.10 [-0.25, 0.00] | 0 reached by D3 only, 2 by H1 only | 1.0000 | no claim |

## Radio, time-varying backhaul: path loss 2.7

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 0.90 | 236.6 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 20 | 1.00 | 260.4 | -16.4 [-23.3, -10.1] | 18 | 0.0032 | FedEx (D4) | -0.10 [-0.25, 0.00] | 0 reached by D4 only, 2 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 261.0 | -17.1 [-23.7, -11.1] | 18 | 0.0032 | FedEx (D4) | -0.10 [-0.25, 0.00] | 0 reached by D4 only, 2 by H1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 261.1 | -16.7 [-24.3, -10.1] | 18 | 0.0030 | FedEx (D4) | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by D2 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 0.95 | 296.3 | -56.0 [-137.5, -3.6] | 18 | 0.0071 | FedEx (D4) | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 303.4 | -61.9 [-123.3, -18.8] | 18 | 0.0032 | FedEx (D4) | -0.10 [-0.25, 0.00] | 0 reached by D4 only, 2 by D3 only | 1.0000 | no claim |

## Radio, time-varying backhaul: shadowing 8 dB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 262.0 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 1.00 | 264.2 | -2.2 [-25.4, 30.2] | 20 | 0.0442 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 280.2 | -18.2 [-25.1, -11.8] | 20 | 0.0018 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 280.9 | -18.9 [-25.5, -12.5] | 20 | 0.0018 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 0.95 | 291.6 | -36.5 [-98.7, -2.6] | 19 | 0.0371 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 304.9 | -42.9 [-118.3, 8.7] | 20 | 0.0067 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |

## Radio, mission backhaul: default 5 dB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 264.5 |  |  |  |  |  |  |  |  |
| Oort (D2) | 20 | 1.00 | 265.8 | -1.3 [-24.9, 29.1] | 20 | 0.0557 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 1.00 | 274.6 | -10.1 [-44.5, 20.0] | 20 | 0.1556 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 279.6 | -15.1 [-33.2, 12.2] | 20 | 0.0146 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 281.8 | -17.3 [-25.7, -10.0] | 20 | 0.0042 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 281.8 | -17.4 [-24.8, -10.5] | 20 | 0.0042 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |

## Radio, mission backhaul: 17 dB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 244.3 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 1.00 | 260.3 | -16.0 [-92.1, 60.6] | 20 | 0.9096 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 264.8 | -20.5 [-76.1, 49.9] | 20 | 0.1579 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 268.2 | -24.0 [-86.3, 32.8] | 20 | 0.4621 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 274.2 | -29.9 [-85.1, 40.7] | 20 | 0.0430 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 277.2 | -32.9 [-111.2, 51.5] | 20 | 0.1926 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Radio, mission backhaul: path loss 2.7

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 243.5 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 1.00 | 262.7 | -19.2 [-48.2, -2.3] | 20 | 0.0231 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 262.7 | -19.2 [-28.4, -11.4] | 20 | 0.0010 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 269.3 | -25.8 [-49.1, -11.4] | 20 | 0.0017 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 269.4 | -26.0 [-48.8, -11.1] | 20 | 0.0017 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 275.9 | -32.4 [-54.3, -16.2] | 20 | 0.0017 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Payload, N = 6 knee: 2 MB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 271.4 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 1.00 | 284.7 | -13.3 [-47.4, 17.0] | 20 | 0.1626 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 301.4 | -30.0 [-60.8, -7.4] | 20 | 0.0157 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 302.2 | -30.8 [-63.6, -8.3] | 20 | 0.0157 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 305.6 | -34.2 [-69.2, -11.2] | 20 | 0.0059 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 325.3 | -54.0 [-106.6, -14.8] | 20 | 0.0074 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Payload, N = 6 knee: 5 MB

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 20 | 1.00 | 276.4 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 1.00 | 325.7 | -49.3 [-106.3, 5.9] | 20 | 0.2105 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 330.3 | -53.9 [-123.9, -2.3] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 330.8 | -54.4 [-155.3, 2.6] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 331.3 | -54.9 [-165.8, 25.4] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D4 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 347.4 | -71.0 [-149.1, -2.7] | 20 | 0.0879 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D2 only | 1.0000 | no claim |

## Payload, N = 6 knee: 10 MB

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 20 | 1.00 | 315.0 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 20 | 1.00 | 320.3 | -5.3 [-53.3, 35.5] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 321.7 | -6.7 [-95.5, 65.0] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 337.0 | -22.0 [-116.0, 50.7] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 379.6 | -64.6 [-164.6, 24.7] | 20 | 0.5373 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D2 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 382.1 | -67.1 [-200.2, 25.2] | 20 | 0.9939 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D4 only | 1.0000 | no claim |

## Layout, N = 6 knee: 25 % beyond 60 m

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 1.00 | 220.5 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 1.00 | 221.2 | -0.7 [-11.6, 16.2] | 20 | 0.0747 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 226.6 | -6.1 [-17.5, 11.6] | 20 | 0.0225 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 226.6 | -6.1 [-17.5, 11.6] | 20 | 0.0225 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 230.5 | -10.0 [-15.8, -4.5] | 20 | 0.0225 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 232.9 | -12.3 [-17.4, -7.9] | 20 | 0.0021 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Layout, N = 6 knee: 50 % beyond 60 m

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 20 | 1.00 | 189.8 |  |  |  |  |  |  |  |  |
| MAX-AoI (D1) | 20 | 1.00 | 189.8 | 0.0 [0.0, 0.0] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D1 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 1.00 | 198.9 | -9.0 [-44.3, 10.9] | 20 | 0.1385 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 203.8 | -13.9 [-47.2, 5.7] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 1.00 | 206.2 | -16.3 [-49.1, 3.8] | 20 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D2 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 227.9 | -38.0 [-99.8, 8.6] | 20 | 0.7574 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D4 only | 1.0000 | no claim |

## Layout, N = 6 knee: 75 % beyond 60 m

Fastest: Value-Aware FL (D3).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Value-Aware FL (D3) | 20 | 1.00 | 245.2 |  |  |  |  |  |  |  |  |
| Oort (D2) | 20 | 1.00 | 249.7 | -4.5 [-29.4, 11.1] | 20 | 0.9264 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by D2 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 1.00 | 256.9 | -11.7 [-77.3, 29.4] | 20 | 0.1211 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by D4 only | 1.0000 | no claim |
| FedCS (D5) | 20 | 1.00 | 266.5 | -21.3 [-88.5, 21.5] | 20 | 0.1945 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by D5 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 1.00 | 277.2 | -32.0 [-91.3, 6.8] | 20 | 0.9264 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 281.3 | -36.1 [-95.3, 0.9] | 20 | 0.9264 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by D1 only | 1.0000 | no claim |

## Field, N = 12 knee: ±200 m, 1 MB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 0.95 | 1124.1 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 0.90 | 1111.4 | -42.3 [-64.0, -22.3] | 18 | 0.0048 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.90 | 1325.0 | -255.9 [-382.8, -85.6] | 18 | 0.0048 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.95 | 1368.3 | -360.9 [-530.3, -165.4] | 18 | 0.0048 | FedEx (D4) | 0.00 [-0.15, 0.15] | 1 reached by D4 only, 1 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.95 | 1368.4 | -361.0 [-534.9, -165.5] | 18 | 0.0048 | FedEx (D4) | 0.00 [-0.15, 0.15] | 1 reached by D4 only, 1 by H1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.90 | 1429.1 | -360.0 [-482.4, -234.4] | 18 | 0.0007 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D2 only | 1.0000 | no claim |

## Field, N = 12 knee: ±300 m, 1 MB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 0.90 | 1513.4 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 0.90 | 1593.9 | -80.5 [-227.1, 55.1] | 18 | 0.0168 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.90 | 2093.8 | -580.4 [-751.0, -429.7] | 18 | 0.0000 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.80 | 2159.6 | -685.8 [-861.9, -528.7] | 16 | 0.0001 | FedEx (D4) | 0.10 [0.00, 0.25] | 2 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.85 | 2240.8 | -715.2 [-883.9, -560.6] | 17 | 0.0001 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 2419.1 | -851.1 [-1259.3, -524.5] | 18 | 0.0003 | FedEx (D4) | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by D2 only | 1.0000 | no claim |

## Field, N = 12 knee: ±200 m, 10 MB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 0.90 | 1380.6 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 0.95 | 1415.0 | -51.0 [-83.0, -26.3] | 18 | 0.0020 | FedEx (D4) | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.95 | 1602.4 | -287.7 [-425.9, -146.2] | 18 | 0.0020 | FedEx (D4) | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.90 | 1726.8 | -346.3 [-440.2, -259.1] | 18 | 0.0001 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.85 | 1784.9 | -387.1 [-542.1, -256.8] | 17 | 0.0001 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by H1 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.85 | 1794.1 | -396.2 [-540.5, -266.7] | 17 | 0.0001 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by D1 only | 1.0000 | no claim |

## Field, N = 12 knee: ±300 m, 10 MB

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 0.90 | 1911.6 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 0.90 | 2031.3 | -119.8 [-434.1, 151.4] | 18 | 0.0312 | no claim | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D5 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.90 | 2383.0 | -471.4 [-627.0, -296.9] | 18 | 0.0010 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.90 | 2458.4 | -546.8 [-736.0, -355.6] | 18 | 0.0010 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 0.90 | 2874.9 | -963.4 [-1618.1, -446.5] | 18 | 0.0010 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D1 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.85 | 2889.2 | -955.4 [-1660.6, -421.2] | 17 | 0.0021 | FedEx (D4) | 0.05 [0.00, 0.15] | 1 reached by D4 only, 0 by H1 only | 1.0000 | no claim |

## Field: N = 24, ±300 m, 10 MB, 600 s

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 20 | 0.90 | 2937.8 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 1.00 | 3134.4 | -191.0 [-580.4, 260.3] | 18 | 0.3892 | no claim | -0.10 [-0.25, 0.00] | 0 reached by H1 only, 2 by D5 only | 1.0000 | no claim |
| Oort (D2) | 20 | 0.95 | 3141.7 | -287.6 [-626.8, 33.4] | 17 | 0.4494 | no claim | -0.05 [-0.20, 0.10] | 1 reached by H1 only, 2 by D2 only | 1.0000 | no claim |
| MAX-AoI (D1) | 20 | 1.00 | 3164.0 | -144.3 [-488.1, 218.5] | 18 | 0.4494 | no claim | -0.10 [-0.25, 0.00] | 0 reached by H1 only, 2 by D1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 1.00 | 3261.5 | -326.7 [-797.3, 64.4] | 18 | 0.1540 | no claim | -0.10 [-0.25, 0.00] | 0 reached by H1 only, 2 by D3 only | 1.0000 | no claim |
| FedEx (D4) | 20 | 0.95 | 3549.9 | -695.6 [-1008.7, -396.1] | 17 | 0.0105 | Deadline-Heuristic (H1) | -0.05 [-0.20, 0.10] | 1 reached by H1 only, 2 by D4 only | 1.0000 | no claim |

## Training time, 10 missions: spread

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 40 | 1.00 | 275.6 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 40 | 1.00 | 276.2 | -0.6 [-68.3, 60.5] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D5 only | 1.0000 | no claim |
| FedEx (D4) | 40 | 1.00 | 297.6 | -22.0 [-99.4, 40.9] | 40 | 0.4625 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D4 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 316.7 | -41.1 [-92.4, 1.4] | 40 | 1.0000 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D3 only | 1.0000 | no claim |

## Training time, 10 missions: stragglers

Fastest: FedCS (D5).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedCS (D5) | 40 | 1.00 | 269.6 |  |  |  |  |  |  |  |  |
| Deadline-Heuristic (H1) | 40 | 1.00 | 288.8 | -19.2 [-81.5, 51.3] | 40 | 0.1132 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by H1 only | 1.0000 | no claim |
| FedEx (D4) | 40 | 1.00 | 308.6 | -39.1 [-91.7, 12.5] | 40 | 0.1132 | no claim | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D4 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 40 | 1.00 | 334.3 | -64.8 [-118.1, -13.6] | 40 | 0.0248 | FedCS (D5) | 0.00 [0.00, 0.00] | 0 reached by D5 only, 0 by D3 only | 1.0000 | no claim |

## Data, N = 6 knee: IID

Fastest: Value-Aware FL (D3).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Value-Aware FL (D3) | 20 | 1.00 | 279.6 |  |  |  |  |  |  |  |  |
| Deadline-Heuristic (H1) | 20 | 1.00 | 281.8 | -2.3 [-26.6, 14.7] | 20 | 0.2274 | no claim | 0.00 [0.00, 0.00] | 0 reached by D3 only, 0 by H1 only | 1.0000 | no claim |

## Data, N = 6 knee: skewed shard sizes

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 20 | 1.00 | 174.1 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 1.00 | 179.1 | -5.0 [-12.0, 1.5] | 20 | 0.1626 | no claim | 0.00 [0.00, 0.00] | 0 reached by H1 only, 0 by D3 only | 1.0000 | no claim |

## Data, N = 6 knee: Dirichlet 1

Fastest: Deadline-Heuristic (H1).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Deadline-Heuristic (H1) | 20 | 0.65 | 381.5 |  |  |  |  |  |  |  |  |
| Value-Aware FL (D3) | 20 | 0.75 | 443.8 | -30.7 [-79.8, -4.2] | 13 | 0.0137 | Deadline-Heuristic (H1) | -0.10 [-0.25, 0.00] | 0 reached by H1 only, 2 by D3 only | 0.5000 | no claim |

## Data, N = 6 knee: Dirichlet 0.1

Fastest: FedEx (D4).

| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm [95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | discordant | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedEx (D4) | 20 | 0.50 | 470.8 |  |  |  |  |  |  |  |  |
| FedCS (D5) | 20 | 0.55 | 481.7 | -12.0 [-21.1, -3.0] | 10 | 0.0547 | no claim | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by D5 only | 1.0000 | no claim |
| Deadline-Heuristic (H1) | 20 | 0.55 | 506.9 | -34.3 [-54.7, -15.1] | 10 | 0.0234 | FedEx (D4) | -0.05 [-0.15, 0.00] | 0 reached by D4 only, 1 by H1 only | 1.0000 | no claim |
| Value-Aware FL (D3) | 20 | 0.50 | 519.5 | -48.6 [-72.4, -29.0] | 10 | 0.0117 | FedEx (D4) | 0.00 [0.00, 0.00] | 0 reached by D4 only, 0 by D3 only | 1.0000 | no claim |

## Reading it

- Means are over each arm's own reached trials, so an arm that reached tau less often can show a lower mean; the paired time test and the reach test are the check on that.
- Cells of 20 trials and of 40 are not the same trials: FedEx's route flies the same at every budget, and its 264 s in the 20-trial budget cells and 232 s in the 40-trial ones come from different trial sets.
- Holm runs within a scenario only, not across scenarios.

# DDIL1N: the baselines pinned to narrow under step 1's contact model

Exploratory, not pre-registered. Deadline-Heuristic and FedEx pinned to the narrow band (campaign.baseline_contact_band), beside step 1's FeRRy-Plan, FeRRy-Adapt and the same baselines on wide, N = 6, knee budget, 20 trials on step 1's seeds, τ = 0.71. Paired by trial and seed; each measure in a Holm family of its own per comparison group. Difference = the first arm − the second.

## default channel (n6k1_knee_d1base)

| arm | reach τ | time to τ (s) | time to τ, miss at the end (s) | hover per mission (s) | updates per mission |
|---|---|---|---|---|---|
| FeRRy-Plan | 1.00 | 167 | 167 | 76 | 3.55 |
| FeRRy-Adapt | 1.00 | 148 | 148 | 68 | 3.54 |
| Deadline-Heuristic on wide | 1.00 | 286 | 286 | 12 | 3.40 |
| Deadline-Heuristic on narrow | 1.00 | 167 | 167 | 80 | 3.55 |
| FedEx on wide | 1.00 | 246 | 246 | 11 | 3.54 |
| FedEx on narrow | 1.00 | 167 | 167 | 80 | 3.55 |

| first | second | measure | first − second [95% CI] | pairs | Holm p | verdict |
|---|---|---|---|---|---|---|
| FeRRy-Plan | Deadline-Heuristic on narrow | time to tau | -0.6 [-1.7, 0.1] | 20 | 0.7188 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | hover per mission | -4.4 [-9.0, -0.5] | 20 | 0.2748 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | updates per mission | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | time to tau, miss at the end | -0.6 [-1.7, 0.1] | 20 | 0.7188 | no claim |
| FeRRy-Plan | FedEx on narrow | time to tau | -0.6 [-1.7, 0.1] | 20 | 0.7188 | no claim |
| FeRRy-Plan | FedEx on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | FedEx on narrow | hover per mission | -4.4 [-9.0, -0.5] | 20 | 0.2748 | no claim |
| FeRRy-Plan | FedEx on narrow | updates per mission | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | FedEx on narrow | time to tau, miss at the end | -0.6 [-1.7, 0.1] | 20 | 0.7188 | no claim |
| FeRRy-Adapt | Deadline-Heuristic on narrow | time to tau | -19.2 [-30.5, -9.4] | 20 | 0.0018 | FeRRy-Adapt |
| FeRRy-Adapt | Deadline-Heuristic on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | Deadline-Heuristic on narrow | hover per mission | -12.8 [-17.1, -8.4] | 20 | 0.0012 | FeRRy-Adapt |
| FeRRy-Adapt | Deadline-Heuristic on narrow | updates per mission | -0.01 [-0.04, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | Deadline-Heuristic on narrow | time to tau, miss at the end | -19.2 [-30.5, -9.4] | 20 | 0.0018 | FeRRy-Adapt |
| FeRRy-Adapt | FedEx on narrow | time to tau | -19.2 [-30.5, -9.4] | 20 | 0.0018 | FeRRy-Adapt |
| FeRRy-Adapt | FedEx on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | FedEx on narrow | hover per mission | -12.8 [-17.1, -8.4] | 20 | 0.0012 | FeRRy-Adapt |
| FeRRy-Adapt | FedEx on narrow | updates per mission | -0.01 [-0.04, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | FedEx on narrow | time to tau, miss at the end | -19.2 [-30.5, -9.4] | 20 | 0.0018 | FeRRy-Adapt |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | time to tau | 119.1 [70.1, 179.7] | 20 | 0.0000 | Deadline-Heuristic on narrow |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | hover per mission | -68.6 [-75.1, -62.6] | 20 | 0.0000 | Deadline-Heuristic on wide |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | updates per mission | -0.15 [-0.31, -0.04] | 20 | 0.0471 | Deadline-Heuristic on narrow |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | time to tau, miss at the end | 119.1 [70.1, 179.7] | 20 | 0.0000 | Deadline-Heuristic on narrow |
| FedEx on wide | FedEx on narrow | time to tau | 78.9 [50.3, 115.9] | 20 | 0.0000 | FedEx on narrow |
| FedEx on wide | FedEx on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FedEx on wide | FedEx on narrow | hover per mission | -69.1 [-75.3, -63.5] | 20 | 0.0000 | FedEx on wide |
| FedEx on wide | FedEx on narrow | updates per mission | -0.01 [-0.04, 0.00] | 20 | 0.3173 | no claim |
| FedEx on wide | FedEx on narrow | time to tau, miss at the end | 78.9 [50.3, 115.9] | 20 | 0.0000 | FedEx on narrow |

## 17 dB (n6k1_knee_d1ampharsh)

| arm | reach τ | time to τ (s) | time to τ, miss at the end (s) | hover per mission (s) | updates per mission |
|---|---|---|---|---|---|
| FeRRy-Plan | 1.00 | 151 | 151 | 59 | 3.54 |
| FeRRy-Adapt | 1.00 | 125 | 125 | 47 | 3.55 |
| Deadline-Heuristic on wide | 1.00 | 277 | 277 | 15 | 2.73 |
| Deadline-Heuristic on narrow | 1.00 | 151 | 151 | 60 | 3.54 |
| FedEx on wide | 1.00 | 216 | 216 | 17 | 2.94 |
| FedEx on narrow | 1.00 | 151 | 151 | 60 | 3.54 |

| first | second | measure | first − second [95% CI] | pairs | Holm p | verdict |
|---|---|---|---|---|---|---|
| FeRRy-Plan | Deadline-Heuristic on narrow | time to tau | -0.5 [-1.5, 0.0] | 20 | 0.7188 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | hover per mission | -0.8 [-2.3, 0.2] | 20 | 0.5701 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | updates per mission | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | Deadline-Heuristic on narrow | time to tau, miss at the end | -0.5 [-1.5, 0.0] | 20 | 0.7188 | no claim |
| FeRRy-Plan | FedEx on narrow | time to tau | -0.5 [-1.5, 0.0] | 20 | 0.7188 | no claim |
| FeRRy-Plan | FedEx on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | FedEx on narrow | hover per mission | -0.8 [-2.3, 0.2] | 20 | 0.5701 | no claim |
| FeRRy-Plan | FedEx on narrow | updates per mission | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Plan | FedEx on narrow | time to tau, miss at the end | -0.5 [-1.5, 0.0] | 20 | 0.7188 | no claim |
| FeRRy-Adapt | Deadline-Heuristic on narrow | time to tau | -26.0 [-46.9, -10.4] | 20 | 0.0018 | FeRRy-Adapt |
| FeRRy-Adapt | Deadline-Heuristic on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | Deadline-Heuristic on narrow | hover per mission | -13.6 [-18.1, -9.1] | 20 | 0.0012 | FeRRy-Adapt |
| FeRRy-Adapt | Deadline-Heuristic on narrow | updates per mission | 0.01 [0.00, 0.04] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | Deadline-Heuristic on narrow | time to tau, miss at the end | -26.0 [-46.9, -10.4] | 20 | 0.0018 | FeRRy-Adapt |
| FeRRy-Adapt | FedEx on narrow | time to tau | -26.0 [-46.9, -10.4] | 20 | 0.0018 | FeRRy-Adapt |
| FeRRy-Adapt | FedEx on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | FedEx on narrow | hover per mission | -13.6 [-18.1, -9.2] | 20 | 0.0012 | FeRRy-Adapt |
| FeRRy-Adapt | FedEx on narrow | updates per mission | 0.01 [0.00, 0.04] | 20 | 1.0000 | no claim |
| FeRRy-Adapt | FedEx on narrow | time to tau, miss at the end | -26.0 [-46.9, -10.4] | 20 | 0.0018 | FeRRy-Adapt |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | time to tau | 125.7 [69.0, 182.5] | 20 | 0.0005 | Deadline-Heuristic on narrow |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | hover per mission | -45.1 [-48.8, -41.4] | 20 | 0.0000 | Deadline-Heuristic on wide |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | updates per mission | -0.81 [-1.01, -0.61] | 20 | 0.0005 | Deadline-Heuristic on narrow |
| Deadline-Heuristic on wide | Deadline-Heuristic on narrow | time to tau, miss at the end | 125.7 [69.0, 182.5] | 20 | 0.0005 | Deadline-Heuristic on narrow |
| FedEx on wide | FedEx on narrow | time to tau | 64.5 [10.7, 110.7] | 20 | 0.0107 | FedEx on narrow |
| FedEx on wide | FedEx on narrow | reach | 0.00 [0.00, 0.00] | 20 | 1.0000 | no claim |
| FedEx on wide | FedEx on narrow | hover per mission | -43.6 [-48.2, -38.8] | 20 | 0.0000 | FedEx on wide |
| FedEx on wide | FedEx on narrow | updates per mission | -0.60 [-0.82, -0.40] | 20 | 0.0005 | FedEx on narrow |
| FedEx on wide | FedEx on narrow | time to tau, miss at the end | 64.5 [10.7, 110.7] | 20 | 0.0107 | FedEx on narrow |


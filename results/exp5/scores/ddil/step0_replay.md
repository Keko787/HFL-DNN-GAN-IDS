# DDIL step 0: the trace replay

*Read-only, from P1's kept traces; nothing flown. Script: `scripts/exp5/ddil_replay.py`; plan: `DeveloperDocs/FeRRy_DDIL_Unified_Plan.md`, "Step 0, extended". Per-trial rows are in `step0_replay_trials.csv`, the rebuild checks in `step0_replay_checks.csv`.*

All values are per mission, averaged over a trial's 4 missions and then over the 20 trials. Hover is dwell only (both passes), as the ledger counts it. Updates are Pass-1 sessions that carry an update and complete. Stops keep their recorded arrival times, classes and gates; only time inside a stop moves.

## The rebuild

- 220 trials rebuilt. Channel parameters match the recorded ones in every trial: yes.
- Recorded arrival SNRs reproduced to within 0.0e+00 dB; uplink drops recomputed from the trial seed: 0 mismatches.
- Pricing (a) reproduces every stop's recorded dwell to within 2.6e-10 s, and each trial's ledger dwell per mission to within 3.4e-10 s.

## F, mission backhaul

| Pricing | Hover, 5 dB | Hover, 17 dB | Updates, 5 dB | Updates, 17 dB | Failed sessions, 17 dB | Updates lost, 17 dB |
|---|---|---|---|---|---|---|
| (a) as recorded | 107.2 | 133.0 | 3.54 | 3.31 | 0.00 | 0.00 |
| (b) rate follows the channel | 78.0 | 59.6 | 3.54 | 3.31 | 0.00 | 0.00 |
| (b) with the interference clipped at 0 | 96.7 | 114.3 | 3.54 | 3.31 | 0.00 | 0.00 |
| (b), shared interference | 78.0 | 59.6 | 3.54 | 3.31 | 0.00 | 0.00 |
| (c) fastest class, no deferral | 44.6 | 29.1 | 3.54 | 3.31 | 0.00 | 0.00 |
| (c), shared | 48.2 | 34.7 | 3.54 | 3.31 | 0.00 | 0.00 |
| (d) the rule on (a) | 44.6 | 27.9 | 3.54 | 3.31 | 0.00 | 0.00 |
| (s) step 1's model, t_out 10 s | 78.0 | 59.8 | 3.54 | 3.30 | 0.38 | 0.01 |
| (s) at t_out 2 s | 77.7 | 58.0 | 3.54 | 3.23 | 1.69 | 0.09 |
| (s), resumable | 77.9 | 59.1 | 3.54 | 3.30 | 0.38 | 0.01 |
| (s′) shared | 78.0 | 59.8 | 3.54 | 3.30 | 0.38 | 0.01 |
| (e) the rule on (s) | 44.2 | 26.6 | 3.54 | 3.31 | 0.35 | 0.00 |
| (e′) the rule on (s′) | 45.7 | 34.9 | 3.54 | 3.30 | 0.59 | 0.01 |
| (e) without the guard | 44.0 | 26.5 | 3.54 | 3.31 | 0.35 | 0.00 |
| (e′) without the guard | 45.7 | 34.9 | 3.54 | 3.30 | 0.59 | 0.01 |
| (e) deferral only | 76.1 | 59.3 | 3.54 | 3.30 | 0.38 | 0.01 |
| (e′) deferral only | 76.1 | 59.3 | 3.54 | 3.30 | 0.38 | 0.01 |
| (p) (s) + re-poll | 78.5 | 61.2 | 3.55 | 3.51 | 0.38 | 0.01 |
| (p) (s′) + re-poll | 78.5 | 61.2 | 3.55 | 3.51 | 0.38 | 0.01 |
| (o) foresight optimum on (s) | 30.8 | 19.5 | 3.54 | 3.31 | 0.00 | 0.00 |
| (o) foresight optimum on (s′) | 35.2 | 29.7 | 3.54 | 3.31 | 0.00 | 0.00 |
| foresight greedy on (s) | 37.9 | 23.4 | 3.54 | 3.31 | 0.00 | 0.00 |
| foresight greedy on (s′) | 41.8 | 33.6 | 3.54 | 3.30 | 0.00 | 0.01 |
| loose bound on (s): each session alone | 10.8 | 4.7 | 3.54 | 3.31 | 0.00 | 0.00 |
| loose bound on (s′) | 10.8 | 4.6 | 3.54 | 3.31 | 0.00 | 0.00 |

## FX, mission backhaul

| Pricing | Hover, 5 dB | Hover, 17 dB | Updates, 5 dB | Updates, 17 dB | Failed sessions, 17 dB | Updates lost, 17 dB |
|---|---|---|---|---|---|---|
| (a) as recorded | 88.2 | 112.1 | 3.54 | 3.45 | 0.00 | 0.00 |
| (b) rate follows the channel | 71.4 | 55.3 | 3.54 | 3.45 | 0.00 | 0.00 |
| (b) with the interference clipped at 0 | 90.9 | 115.4 | 3.54 | 3.45 | 0.00 | 0.00 |
| (b), shared interference | 71.4 | 55.3 | 3.54 | 3.45 | 0.00 | 0.00 |
| (c) fastest class, no deferral | 45.7 | 32.4 | 3.54 | 3.45 | 0.00 | 0.00 |
| (c), shared | 47.8 | 31.9 | 3.54 | 3.45 | 0.00 | 0.00 |
| (d) the rule on (a) | 46.9 | 30.4 | 3.54 | 3.45 | 0.00 | 0.00 |
| (s) step 1's model, t_out 10 s | 71.3 | 55.4 | 3.54 | 3.44 | 0.53 | 0.01 |
| (s) at t_out 2 s | 70.3 | 52.5 | 3.48 | 3.33 | 2.08 | 0.12 |
| (s), resumable | 71.3 | 54.7 | 3.54 | 3.44 | 0.53 | 0.01 |
| (s′) shared | 71.3 | 55.4 | 3.54 | 3.44 | 0.53 | 0.01 |
| (e) the rule on (s) | 44.5 | 31.0 | 3.54 | 3.44 | 0.45 | 0.01 |
| (e′) the rule on (s′) | 45.3 | 32.1 | 3.54 | 3.42 | 0.65 | 0.03 |
| (e) without the guard | 44.2 | 30.0 | 3.54 | 3.45 | 0.42 | 0.00 |
| (e′) without the guard | 45.3 | 32.1 | 3.54 | 3.42 | 0.62 | 0.03 |
| (e) deferral only | 68.8 | 55.1 | 3.54 | 3.44 | 0.51 | 0.01 |
| (e′) deferral only | 68.8 | 55.1 | 3.54 | 3.44 | 0.51 | 0.01 |
| (p) (s) + re-poll | 71.5 | 55.7 | 3.54 | 3.50 | 0.53 | 0.01 |
| (p) (s′) + re-poll | 71.5 | 55.7 | 3.54 | 3.50 | 0.53 | 0.01 |
| (o) foresight optimum on (s) | 31.1 | 21.2 | 3.54 | 3.45 | 0.00 | 0.00 |
| (o) foresight optimum on (s′) | 34.7 | 25.9 | 3.54 | 3.45 | 0.00 | 0.00 |
| foresight greedy on (s) | 38.2 | 25.5 | 3.54 | 3.45 | 0.00 | 0.00 |
| foresight greedy on (s′) | 40.9 | 30.2 | 3.54 | 3.44 | 0.00 | 0.01 |
| loose bound on (s): each session alone | 11.1 | 5.0 | 3.54 | 3.45 | 0.00 | 0.00 |
| loose bound on (s′) | 11.0 | 5.0 | 3.54 | 3.45 | 0.00 | 0.00 |

## How much of F's 17 dB hover rise is the fixed-rate charge

| | F | FX |
|---|---|---|
| Rise under (a), 17 dB − 5 dB | 25.8 | 24.0 |
| Rise under (b) | -18.5 | -16.2 |
| Rise under (b), interference clipped at 0 | 17.6 | 24.5 |
| Rise under (s) | -18.2 | -15.9 |
| Rise under (s′) | -18.2 | -15.9 |
| 17 dB hover, (a) − (b): the fixed-rate charge | 73.4 [54.1, 94.2] | 56.9 [38.3, 78.1] |
| 17 dB hover, (a) − (s) | 73.2 [54.0, 93.8] | 56.7 [38.2, 77.8] |
| 17 dB hover, (a) − (s′) | 73.2 [54.0, 93.8] | 56.7 [38.2, 77.8] |

## Go/no-go for step 3 (the per-transfer rule)

The bar (decided 16 Nov, before this was read): the rule saves at least 15 s of hover a mission at 17 dB on the mission backhaul, at `t_out` = 10 s, under shared interference ((e′) against (s′)), with no fewer sessions completed. The per-class number is the sensitivity. Differences are (without the rule) − (with it), so a saving is positive; 95% bootstrap CIs over the 20 trials.

| Arm | Comparison | Hover saved | Sessions completed, change | Updates, change | Meets the bar |
|---|---|---|---|---|---|
| F | (e′) vs (s′), shared (primary) | 24.9 [21.5, 28.3] | 0.00 [-0.04, 0.04] | 0.00 [-0.04, 0.04] | yes |
| F | (e) vs (s), per class | 33.2 [28.3, 38.2] | 0.01 [0.00, 0.04] | 0.01 [0.00, 0.04] | yes |
| F | (e′) without the guard vs (s′) | 24.9 [21.5, 28.3] | 0.00 [-0.04, 0.04] | 0.00 [-0.04, 0.04] | yes |
| F | (e′) deferral only vs (s′) | 0.4 [0.2, 0.7] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | no |
| FX | (e′) vs (s′), shared (primary) | 23.3 [19.8, 26.7] | 0.00 [-0.05, 0.05] | -0.01 [-0.05, 0.03] | yes |
| FX | (e) vs (s), per class | 24.4 [20.5, 28.3] | 0.01 [0.00, 0.04] | 0.00 [-0.04, 0.04] | yes |
| FX | (e′) without the guard vs (s′) | 23.3 [19.8, 26.7] | 0.00 [-0.05, 0.05] | -0.01 [-0.05, 0.03] | yes |
| FX | (e′) deferral only vs (s′) | 0.3 [0.1, 0.5] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | no |

**Verdict on the primary model:** F meets the bar; FX meets it.

## Where the rule's saving comes from

Per mission, mission backhaul. "Moved" counts sessions the rule served on a class other than the stop's; "saved" is hover without the rule minus hover with it, 95% CI over trials.

| Arm | Channel | Model | Sessions | Moved to another class | Deferrals | Hover saved |
|---|---|---|---|---|---|---|
| F | 5 dB | per class | 11.93 | 9.12 (77%) | 0.45 | 33.7 [28.3, 38.7] |
| F | 5 dB | shared | 11.94 | 10.93 (92%) | 1.14 | 32.3 [28.2, 35.9] |
| F | 17 dB | per class | 11.03 | 7.40 (67%) | 0.29 | 33.2 [28.3, 38.2] |
| F | 17 dB | shared | 11.25 | 10.03 (89%) | 3.51 | 24.9 [21.5, 28.3] |
| FX | 5 dB | per class | 11.93 | 7.69 (64%) | 0.59 | 26.8 [22.6, 31.1] |
| FX | 5 dB | shared | 11.97 | 9.66 (81%) | 1.21 | 26.1 [22.5, 29.6] |
| FX | 17 dB | per class | 11.64 | 6.15 (53%) | 0.35 | 24.4 [20.5, 28.3] |
| FX | 17 dB | shared | 11.82 | 9.46 (80%) | 2.45 | 23.3 [19.8, 26.7] |

## The rule's settings (reported, not used for the decision)

Hover saved a mission at 17 dB, mission backhaul: (s) − (e) per class, and (s′) − (e′) shared.

| θ | D_max | F, per class | F, shared | FX, per class | FX, shared |
|---|---|---|---|---|---|
| 1.5 | 1 | 33.5 | 24.8 | 24.9 | 23.4 |
| 1.5 | 2 | 33.5 | 24.8 | 24.9 | 23.4 |
| 2 | 1 | 33.2 | 24.9 | 24.4 | 23.3 |
| 2 | 2 | 33.2 | 24.9 | 24.4 | 23.3 |
| 3 | 1 | 33.2 | 24.9 | 24.3 | 23.3 |
| 3 | 2 | 33.2 | 24.9 | 24.3 | 23.3 |

## What step 1's model removes from the rule's recorded-physics gain

| Arm | Rule's saving on (a): (a) − (d) | Rule's saving on (s): (s) − (e) | (d) − (e) |
|---|---|---|---|
| F | 105.1 | 33.2 | 1.3 [-2.3, 4.6] |
| FX | 81.7 | 24.4 | -0.6 [-4.1, 2.6] |

## Step 1's session model: failures

| Arm | Pricing | Sessions failed | Retries failed | Updates lost | Updates |
|---|---|---|---|---|---|
| F | (s) step 1's model, t_out 10 s | 0.38 | 0.01 | 0.01 | 3.30 |
| F | (s) at t_out 2 s | 1.69 | 0.21 | 0.09 | 3.23 |
| F | (s), resumable | 0.38 | 0.01 | 0.01 | 3.30 |
| F | (s′) shared | 0.38 | 0.01 | 0.01 | 3.30 |
| FX | (s) step 1's model, t_out 10 s | 0.53 | 0.03 | 0.01 | 3.44 |
| FX | (s) at t_out 2 s | 2.08 | 0.40 | 0.12 | 3.33 |
| FX | (s), resumable | 0.53 | 0.03 | 0.01 | 3.44 |
| FX | (s′) shared | 0.53 | 0.03 | 0.01 | 3.44 |

## The re-poll, (p), at 17 dB on the mission backhaul

| Arm | Model | Members re-polled above the floor | Updates recovered | Updates, (s) | Updates, (p) | Hover added |
|---|---|---|---|---|---|---|
| F | per class | 0.46 | 0.21 | 3.30 | 3.51 | 1.4 |
| F | shared | 0.46 | 0.21 | 3.30 | 3.51 | 1.4 |
| FX | per class | 0.09 | 0.06 | 3.44 | 3.50 | 0.3 |
| FX | shared | 0.09 | 0.06 | 3.44 | 3.50 | 0.3 |
| H1 | per class | 0.10 | 0.09 | 2.65 | 2.74 | 0.2 |
| H1 | shared | 0.10 | 0.09 | 2.65 | 2.74 | 0.2 |
| D4 | per class | 0.20 | 0.11 | 3.00 | 3.11 | 0.3 |
| D4 | shared | 0.20 | 0.11 | 3.00 | 3.11 | 0.3 |

## The hindsight bound, (o), and step 3c's gate

Step 3c (a learned per-transfer decision) is built only if (o) is at least 10 s of hover a mission better than the rule at 17 dB, under shared interference ((o) against (e′)). (o) is the foresight optimum: the best order, class and wait (up to 30 s, on a 0.5 s grid) for each stop's sessions with the channel known in advance, found by a DP over the stop's subsets of sessions, which is exact up to that grid. No policy that reads only the present can beat it, learned or not. The loose bound (each session alone at its best class and start, ignoring that sessions run one after another) and the foresight greedy are shown for scale.

| Arm | Model | (e) hover | (o) hover | (e) − (o) | Greedy hover | Loose bound | Updates (e) / (o) | Gate |
|---|---|---|---|---|---|---|---|---|
| F | per class | 26.6 | 19.5 | 7.1 [4.9, 9.6] | 23.4 | 4.7 | 3.31 / 3.31 | fails |
| F | shared (primary) | 34.9 | 29.7 | 5.2 [3.1, 7.6] | 33.6 | 4.6 | 3.30 / 3.31 | fails |
| FX | per class | 31.0 | 21.2 | 9.8 [7.2, 12.9] | 25.5 | 5.0 | 3.44 / 3.45 | fails |
| FX | shared (primary) | 32.1 | 25.9 | 6.1 [4.4, 8.0] | 30.2 | 5.0 | 3.42 / 3.45 | fails |

## Below-floor stretches at 17 dB (for decision 11's burst length)

Runs of consecutive 1 s steps below the floor on the stop's class, for every Pass-1 target, over each stop and 60 s either side (runs that touch the stop).

| Backhaul, arm | Runs | Mean (s) | Median (s) | 90th percentile (s) | Longest (s) |
|---|---|---|---|---|---|
| mission, D4 | 56 | 9.1 | 9 | 17 | 21 |
| mission, F | 728 | 6.4 | 4 | 15 | 25 |
| mission, FX | 530 | 7.2 | 5 | 16 | 25 |
| mission, H1 | 49 | 10.6 | 10 | 21 | 28 |
| seconds, F | 753 | 6.5 | 4 | 15 | 24 |
| seconds, FX | 537 | 7.4 | 6 | 17 | 26 |
| **mission, F and FX pooled** | 1258 | **6.7** | 5 | 16 | 25 |

## The seconds backhaul (read, not a gate)

`results/exp5/b3/s515` keeps no default-channel F traces, so F is shown at 17 dB only.

| Arm, cell | (a) | (b) | (s) | (s′) | (e) | (e′) |
|---|---|---|---|---|---|---|
| F, 17 dB | 142.9 | 59.3 | 59.4 | 59.4 | 27.3 | 35.7 |
| FX, 17 dB | 107.4 | 53.8 | 53.7 | 53.7 | 28.8 | 32.0 |
| FX, 5 dB | 89.9 | 71.1 | 71.1 | 71.1 | 44.2 | 45.8 |


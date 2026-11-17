# Steps 1 and 2, and FQ-s1, in the NOMS paper: two versions

*17 Nov 2026. Both files are `main_review3.tex` with only the passages below changed (checked: braces, environments and references balance, no duplicate labels, CRLF kept). LaTeX is not installed on the host, so neither was compiled; compile on Overleaf.*

| | `main_review3_with_step2.tex` | `main_review3_without_step2.tex` |
|---|---|---|
| Step 1 (the stricter contact model) | in | in (identical) |
| FeRRy-RL under the stricter model (FQ-s1) | in | in (identical) |
| Step 2 (the risk-priced dwell) | in: an appendix paragraph and three clauses | not mentioned |
| The three channel-uncertainty passages | keep the promise, now with the first attempt reported | cut back: no promise, no unproven cause |
| Main text | about +150 words (9 two-column lines) | about +80 words (5 lines) |
| Appendix | step 1's paragraph and Table `tab:session`, then step 2's paragraph | step 1's paragraph and Table `tab:session` |

The main text was already about a third of a page over the 8-page limit before either.

## Step 1, in both versions

Sources: `results/exp5/scores/probes/DDIL1.{md,csv}` (Holm per family) and `results/exp5/scores/ddil/step1_beside_recorded.md` (each arm paired with itself on the same seeds). Mission backhaul, \(N=6\), knee, 20 trials, τ = 0.71.

1. **Planner's channel parameters (line ~1014)**, extending the blue oracle note:
   > …they take the simulated channel's values; in the two settings where the planner instead estimates them from the UAV's own readings, mean times to target change by less than 1~s (Appendix~\ref{app:robustness}).

   (The largest mean difference is 0.85 s; one or two trials of 20 differ, by up to 17 s.)
2. **Contact model (line ~1180)**, after the two simplifications:
   > Appendix~\ref{app:robustness} relaxes both simplifications.
3. **Main text, Robustness and Limitations (line ~1692)**, after "no baseline becomes significantly faster than \sysname{}-Plan.":
   > Under a stricter contact model, in which a transfer follows the channel and fails after 10~s below $\gamma_{\min}$, \sysname{}-Plan reaches the target at 17~dB in 151 instead of 261~s and stays within the budget, the baselines change little, and no baseline is significantly faster than \sysname{}-Plan in any of nine channel settings (Appendix~\ref{app:robustness}).
4. **Appendix, after the 17 dB paragraph (line ~2035)**: a paragraph headed **A stricter contact model** (what changes, the 17 dB result, the timeout alone, the nine settings, FedEx, FeRRy-Adapt, the estimated channel) and **Table `tab:session`**:

   | Condition | FeRRy-Plan | FeRRy-Adapt | Deadline-Heuristic | FedEx | MAX-AoI, Oort, Value-Aware FL, FedCS |
   |---|---|---|---|---|---|
   | Default | 194 / 167 | 178 / 148 | 282 / 286 | 264 / 246 | 266–282 / 266–290 |
   | A = 17 dB | 261 / 151 | 215 / 125 | 265 / 277 | 244 / 216 | 260–277 / 246–288* |
   | n = 2.7 | 221 / 184 | 211 / 173 | 269 / 273 | 243 / 243 | 263–276 / 262–272 |

   Time to target (s), recorded / stricter; *Oort and Value-Aware FL reach the target in 95% of trials under the stricter model.

5. **A correction:** FeRRy-Adapt's recorded 17 dB time in the existing appendix sentence changes from 216 to **215 s**. The trials give 215.46 s; P1's report rounds that to 215.5, and the draft rounded it again to 216. Without this the new table and the sentence would disagree.

6. **FeRRy-RL under the stricter model (FQ-s1; `results/exp5/scores/probes/DDILFQ1.{md,csv}`)**, in both versions:
   - closing step 1's appendix paragraph:
     > Under the stricter model, \sysname{}-RL, trained on the recorded model, behaves like \sysname{}-Adapt: at 17~dB it reaches the target 24~s sooner than \sysname{}-Plan, and it differs significantly from \sysname{}-Adapt in no setting. Retraining it on the stricter model, by the same procedure (ten seeds at discount 0.75, the median-validation seed), changes nothing measurable, and \sysname{}-Adapt remains 17~s faster than the retrained policy on the default channel. At \(N=12\), too few trials reach the target within four missions to compare.
   - extending the learning section's conclusion (line ~1682):
     > …the deterministic rule is sufficient, also under the stricter contact model of Section~\ref{sec:robustness}, with or without retraining the policy for it.

   Behind it (Holm per family): FeRRy-RL (FQ-g75) against FeRRy-Plan at 17 dB, 23.6 s [8.3, 44.4]; FQ-g75 against FeRRy-Adapt and FQ-s1 against FQ-g75, no claim in any cell or measure; FeRRy-Adapt against FQ-s1 on the default channel, 17.1 s [8.8, 26.8]. FeRRy-RL flew four \(N=6\) settings (default, 17 dB, per-class 17 dB, 40% bursts) and \(N=12\) at 17 dB, where 10–20% of trials reach the target. It has no recorded mission-backhaul cells, so Table `tab:session` does not show it.

What the step 1 text claims, and its limits:
- **Supported (Holm):** no baseline is significantly faster than FeRRy-Plan in any of the nine settings; FeRRy-Plan is significantly faster than five or six of the six baselines in six of them; FeRRy-Adapt is significantly faster than FeRRy-Plan in seven.
- **Not claimed:** in the three harshest settings (40% bursts, per-class 17 dB, 17 dB under stress) FeRRy-Plan's leads over the baselines have 95% CIs that exclude zero but do not survive Holm across the probe's 102 baseline comparisons, so the text says "no difference survives Holm's correction". At 17 dB, FedEx (216 s against 151 s) is the one baseline not significantly slower.
- **Recorded against stricter** (151 against 261 s, 95% CI of the paired difference 55–163 s) is a paired bootstrap, not a Holm-corrected claim.
- **The default-channel stress budget**, where the draft's 29% overrun comes from, was not flown under the stricter model, so the overrun passages keep the recorded figure.
- **`t_out` = 10 s still needs a platform citation** before publication (the DDIL plan's "Before publication" row).

## Step 2, in the "with" version only

Source: `results/exp5/scores/probes/DDIL2.{md,csv}`.

| | FeRRy-Plan | Step 2's variant | Significant? |
|---|---|---|---|
| Recorded model, 17 dB: missions within budget | 70% | 100% | — |
| Recorded model, 17 dB: hover per mission | 133 s | 10 s | yes |
| Recorded model, 17 dB: time to target | 261 s | 226 s | no (Holm p = 1.0) |
| Recorded model, 17 dB: updates per round | 3.31 | 3.00 | no (Holm p = 0.22) |
| Recorded model, default: time to target | 194 s | 233 s | no (Holm p = 0.16) |
| Stricter model: time to target | — | 40–85 s later in 7 of 9 settings | yes, in those 7 |
| Stricter model: updates per round, harshest settings | — | 0.5–1.2 fewer | yes |
| Without the leave-by time | — | same | no difference |

The passages:
- **Appendix**, a paragraph headed **Pricing channel risk in the plan**, after step 1's table: the variant, its recorded-model result, and its cost under the stricter contact model (which step 1's paragraph now defines).
- **Robustness limitation (line ~1706)**, after the overrun sentence:
  > Pricing that uncertainty into the plan is not a free fix: an exploratory variant that does so removes the overruns at 17~dB but reaches the target later under the stricter contact model (Appendix~\ref{app:robustness}).
- **Conclusion (line ~1737)**:
  > …or for channel uncertainty before takeoff; a first variant that prices this uncertainty removed overruns but delayed the target under a stricter contact model.
- **Closing limitation (line ~2093)**:
  > The planner uses expected channel conditions before takeoff. Pricing the channel's uncertainty removes the overruns in the 17~dB test above, but at a cost in time to target under the stricter contact model, so how best to account for it remains open.

## The "without" version's three passages

- **Robustness limitation:** "Under the stress budget, \sysname{}-Plan also exceeds the mission budget in 29\% of missions by 33~s on average. Modeling update readiness remains future work."
- **Conclusion:** "The current planner does not account for when local training will finish. Future work will incorporate update readiness into mission planning and evaluate \sysname{} with measured UAV wireless links and battery limits in a physical testbed."
- **Closing limitation:** "Finally, under the stress budget, \sysname{}-Plan exceeds the mission budget in 29\% of missions by 33~s on average."

Both versions keep line 1032 (the plan is built on expected conditions and adapted in flight) and line 1488 (what the evaluation covers): they describe the design, not a promise.

# Changes since main (21): the two marked versions

*17 Nov 2026. `main21.tex` is `main (21).tex`, the version the reviewers have. `main21_with_step2.tex` and `main21_without_step2.tex` are it with every change in **blue**, each followed by an **orange note** (`\changenote`) saying what changed and why; a removal gets an orange note at the spot, quoting what went. Not compiled (no LaTeX on the build host); checked that braces, environments and references balance, and that stripping every blue span, note and the new table leaves exactly `main (21)` apart from the removals the notes announce.*

For the final version: uncomment `\renewcommand{\changenote}[1]{}` in the preamble (it hides every note) and drop the `\textcolor{blue}{...}` wrappers.

## In both versions

| # | Where (section) | Change | Why (as the orange note says) |
|---|---|---|---|
| 1 | Mission planning, the outage estimate (after σ_eff ≈ 5.5 dB) | Added: the planner is given the simulator's channel parameters; estimating them itself changes mean times by less than 1 s | The recorded planner had an advantage a deployed one lacks; the stricter-model runs show it does not matter |
| 2 | Environment, the contact model | Added: the two simplifications (a device below γ_min at arrival is skipped; a transfer keeps its start rate), relaxed in the appendix | Sets up the appendix's stricter model |
| 3 | Learning and decisions during flight, the conclusion | Added: the conclusion also holds under the stricter contact model, with or without retraining | FeRRy-RL was flown, and retrained, under it |
| 4 | Robustness and limitations, main text | Added: strong interference narrows the advantage but no baseline is significantly faster; under the stricter model FeRRy-Plan's 17 dB time is 151 instead of 261 s, within budget, and no baseline is faster in any of nine settings | Summarises the 17 dB result and its re-run |
| 5 | Table `tab:robustness`, its note | Added: no baseline significantly faster in the channel rows | The significance result for those rows |
| 6 | Appendix, the 17 dB paragraph | Added: the result on the main-setting backhaul (261 s against 244–277 s; FeRRy-Adapt 215 s) and the trade-off (FeRRy-Plan 30% overruns but 3.31 updates per round; baselines 2.65–3.00) | Reports the 17 dB result and what lies behind it |
| 7 | Appendix, new paragraph **A stricter contact model** | Added: what changes, FeRRy-Plan's 17 dB gain and that the rate following the channel gives it (not the timeout), most of it from the periodic interference's crests, the nine settings and Holm, FedEx at 17 dB, that Deadline-Heuristic and FedEx pinned to narrow match FeRRy-Plan (so its lead is the band) while FeRRy-Adapt stays 19–26 s faster (probe DDIL1N), FeRRy-Adapt, the estimated channel, FeRRy-RL and its retraining | Tests whether the 17 dB weakness came from the simplifications, whether the comparison holds without them, and whether the policy needs retraining |
| 8 | Appendix, new Table `tab:session` | Added: time to target, recorded / stricter, default, 17 dB and n = 2.7; in parentheses, Deadline-Heuristic and FedEx pinned to narrow under the stricter model (167 and 151 s; not flown at n = 2.7) | Sets the two models side by side, and shows the baselines on FeRRy-Plan's band |
| 9 | End of the document | Removed: three stray backtick characters | They printed in the PDF |

Items 1, 2, 4, 5 and 6 are the harsh-radio additions of `main_review3.tex`; items 1, 2 and 4 are extended here with step 1's result, and item 6 gives FeRRy-Adapt's 17 dB time as 215 s (215.46 s, which `main_review3.tex` had rounded twice to 216). Item 3 is FQ-s1's; items 7 and 8 are step 1 of the DDIL plan and FQ-s1.

## Only in `main21_with_step2.tex`

| # | Where | Change | Why |
|---|---|---|---|
| 10 | Appendix, new paragraph **Pricing channel risk in the plan** | Added: the risk-priced variant; under the recorded model at 17 dB it ends the overruns (30% to none) with no significant change in time (226 against 261 s) or updates; under the stricter model it is 40–85 s slower in 7 of 9 settings and merges fewer updates in the harshest | Tests the remedy the limitations name |
| 11 | Robustness and limitations, main text, after the 29% overrun | Added: pricing that uncertainty is not a free fix | Reports what happened when it was priced in |
| 12 | Conclusion | Added: the first variant removed overruns but delayed the target | Future work on channel uncertainty now cites the first attempt |
| 13 | Appendix, closing limitation | Removed ", so accounting for channel uncertainty could reduce these overruns"; added that pricing it removes the overruns at a cost in time, so how to account for it remains open | Replaces the suggestion with the result of trying it |

## Only in `main21_without_step2.tex` (removals, each with an orange note)

| # | Where | Removed | Why |
|---|---|---|---|
| 10 | Robustness and limitations, main text | "because planning uses expected channel conditions before takeoff" | The experiments reported do not establish that cause |
| 11 | Same paragraph | "and channel uncertainty" (from future work) | No longer promised |
| 12 | Conclusion | "or for channel uncertainty before takeoff" | No longer promised |
| 13 | Conclusion | "and channel variation" (from future work) | As above |
| 14 | Appendix, closing limitation | "The planner uses expected channel conditions before takeoff, so accounting for channel uncertainty could reduce these overruns." | Cause not established; remedy no longer promised |

## Still open

- Whether step 2 goes in: the two versions differ only in items 10–14.
- A platform citation for the 10 s timeout.
- Length: the main text grows by about 150 words with step 2, 80 without, on a draft already about a third of a page over 8 pages; the orange notes add more until they are hidden.
- The sources: `results/exp5/scores/probes/{P1,DDIL1,DDILFQ1,DDIL2}.{md,csv}` and `results/exp5/scores/ddil/step1_beside_recorded.md`, on branch `ddil-d2`.

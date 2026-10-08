# Proposed abstract revision (for review)

**Status:** proposal only, written 8 Oct 2026. The paper still uses the current
abstract in `front.tex`. To adopt this one, replace the text between
`\begin{abstract}` and `\end{abstract}` in `front.tex` with the block below, then
rebuild:

```bash
py -3.11 DeveloperDocs/paper/assemble.py
```

**Why:** an audit of the current abstract against the paper's results (below)
found claims the results do not back, wording that contradicts Section III, and a
repeated sentence. The proposal keeps the current abstract's structure (problem,
FeRRy, three components, prototype, comparison, results) and states only what
the results support. It is about 290 words, against about 380 now; IEEE abstracts
usually stay under 250, so check the IPDPS limit before submitting.

## Proposed abstract (paste-ready LaTeX)

```latex
Federated learning (FL) lets devices train a shared model without sharing their
data, but it assumes every device can reach the server. In contested and disaster
zones that assumption fails: devices that are out of range or poorly connected are
skipped repeatedly, and rounds stall. UAV data mules can carry model updates to
and from such devices, but a mule has a finite mission budget, and its radio
decides how far it reaches. We present FeRRy, a Federated RF-aware Routing
framework that treats a mule's reach as a decision. Before each flight, a
band-aware mission planner chooses the contact band class, which trades range for
rate, together with the route, under one feasibility test of per-device deadlines
and the mission budget. In flight, the same test gates a per-stop choice of band
and next stop, and the mule re-plans the rest of its route when the channel uses
up the plan's slack. Updates are merged on each mule and then across mules at the
edge server, with stale updates weighted down, and a coverage term keeps devices
from being starved. In a prototype whose server, mules and devices run as
separate processes over TCP and train a real intrusion-detection model on
CICIoT2023, with flight time, the radio channel and energy simulated, we compare
FeRRy with seven baseline schedulers, including adaptations of FedEx, Oort, FedCS
and a Whittle-index age-of-update scheduler. With six devices, one mule and a
moderate budget, FeRRy reaches the target accuracy 27--31\% sooner than each,
because its plan chooses a long-reach narrow band that removes most of the
transit between stops. Under tight budgets it keeps updates fresher than
age-of-information schedulers, and without its coverage term the share of rounds
that close falls from 99\% to 74\%.
```

## Audit of the current abstract

Each claim in the current abstract, what the paper shows, and the verdict. Numbers
come from `results/exp5/scores/` (the CSV named in brackets) and the paper's
tables.

| Current abstract says | What the paper shows | Verdict | In the proposal |
|---|---|---|---|
| Out-of-range or poorly connected devices are skipped, and FL fails | H0 (no mule, live link) merges 0.5 updates per round and ends at accuracy 0.686, against F's 3.5 and 0.831 (Table `tab:exp5_headline`) | Backed | Kept |
| "the radio access network primarily determines the range and strength of signals by modulating time budget and channel quality allocated for the UAVs" | Motivation, not a result; the radio network does not set the UAV's time budget | Unclear wording | Replaced by "a mule has a finite mission budget, and its radio decides how far it reaches" |
| FeRRy "jointly makes decisions about radio channel selection, task routing, and resource aggregation" | FeRRy chooses a band class (occupied bandwidth on one shared carrier, Section III-B) jointly with the route. Aggregation is a separate step with a fixed rule. | Inaccurate | "chooses the contact band class ... together with the route" |
| Component 1: a band-aware planner selects band and route before each flight | Plan clock (Section III-D). F plans the narrow band on 89% of missions; pinned to the wide band it takes 265 s, D4's time (Fig. `fig:exp5_mechanism`) | Backed | Kept |
| Component 2: an online loop "picks the band and next stop to match it retrospectively, and triggers re-plans" | Every arm re-plans. Only FX and FQ pick the band and next stop per stop, and the headline numbers are F's, which does not. At the N=6 knee FX does not differ from F (s53x); FX is faster in some cells (3 mules at N=12, the default channel and 8 dB shadowing in 5.15, Dirichlet 0.1 in 5.13). "Retrospectively" is wrong: the choice uses the signal observed on arrival. | Partly backed | Describes the per-stop choice and the re-plan; no claimed gain |
| Component 3: merges on each UAV and across UAVs, weighting older updates lower | Section III-F. The merge rules tie (C3, s51): collected updates almost always have merge age zero, so the weighting rarely acts | Backed as design; no measured benefit | Kept as description only |
| "improves system robustness by penalizing configurations that skip devices" | The coverage term matters: without it the round-close rate falls from 99% to 74% at N=12 (C2, s57) and the age of updates rises from 0.78 to 1.10 under stress (C5, s58). The robustness section shows a weakness (training time, 5.12). | Backed as coverage, not robustness | "a coverage term keeps devices from being starved", plus the 99% to 74% result |
| "caps access times on devices" | The age cap bounds the missions a device goes unserved, not access time, and removing it changes nothing (0.78 vs 0.78, C5, s58) | Not backed | Dropped |
| "ensuring that remote devices are not left disconnected" | At the knee budgets for N=24 to 96, the plan serves 54-57% of devices in FerrySim (5.11 (c)) | Overclaim | Dropped |
| The prototype sentence (processes over TCP, real IDS model on CICIoT2023) | Section III-G and the testbed subsection | Backed | Kept, merged with the next row |
| "We evaluate FeRRy on a testbed consisting of edge servers and UAVs ..." | Repeats the prototype sentence; flight time, radio and energy are simulated, and the real-radio study (5.10) was not run, so "UAVs" suggests hardware that was not used | Misleading, duplicated | Merged into one sentence that says what is simulated |
| Compared against FedEx, Oort, FedCS and a Whittle-index scheduler | D4, D2, D5 and D3, plus MAX-AoI (D1) and Chen's DQN (E3). Each is adapted, with its departures stated in Table `tab:exp5_arms` | Backed | "seven baseline schedulers, including adaptations of ..." |
| Metrics "target accuracy, round completion, deadline misses, and network update frequency" | The paper's metrics are time to target accuracy, round-close rate, deadline-miss rate and age of updates | Names differ | The proposal reports results instead of listing metrics |
| 25-31% sooner "than every baseline" (results sentence, added 8 Oct) | 26.6% (D4) to 31.1% (H1, D1) at N=6, one mule, knee budget, every comparison a Holm claim (s53x). Under the stress budget only D4 is significantly slower; with three mules only D4_FedEx is. D4_FedEx at the one-mule knee is as slow (281 s) but not a claim. | Number wrong, qualifier missing; **fixed in the paper** to 27-31% with "one mule and a moderate mission budget" | Same fix |
| Because the narrow band removes most of the transit | F spends 15 s of a mission in transit, the baselines 134-146 s (s53x traces; Fig. `fig:exp5_mechanism`) | Backed (the pinned-band times are descriptive) | Kept |
| Fresher updates than age-of-information schedulers under tight budgets | Age of updates 0.78 against 1.06 (MAX-AoI) and 1.07 (Whittle), Holm p 0.009 and 0.007 (C5, s58) | Backed | Kept |

## Notes for the edit

- "Moderate budget" stands for the knee budget, which the abstract does not
  define. If the IPDPS limit allows, "the budget at which updates saturate" is
  more exact.
- If space is tight, the last sentence's coverage result (99% to 74%) is the
  first to cut; the 27-31% and the freshness result carry the headline.
- The introduction and conclusion already match the proposal's claims, so
  adopting it needs no other edits.

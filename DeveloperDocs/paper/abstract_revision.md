# Abstract revision

**Status (8 Oct 2026, latest):** the paper's abstract is the **short hybrid**
(341 words, section below). The 305-word abstract that follows was the
paper's abstract until then (`front.tex`, rebuilt into `ferry_paper.tex`). It is the proposal written earlier the same day,
with two corrections from the methods audit (`methods_audit.md`):

- **Flight sentence.** The proposal said "the same test gates a per-stop choice
  of band and next stop". FX's band is not gated by the feasibility test; only
  the reorder is (and FQ's whole pair). It now reads: "a per-stop rule can
  switch to a faster band from the observed channel and reorder the remaining
  stops, each reorder checked by the same test".
- **Coverage result.** The 99% to 74% round-close result is at N = 12, so it
  now says "at twelve devices".

**Why it was revised:** an audit of the previous abstract against the paper's
results (below) found claims the results do not back, wording that contradicts
Section III, and a repeated sentence. The new abstract keeps the previous one's
structure (problem, FeRRy, components, prototype, comparison, results) and states
only what the results support. It is 305 words, against about 380 before; IEEE
abstracts usually stay under 250, so check the IPDPS limit before submitting.
If it must shrink, the 99% to 74% clause is the first to cut.

## The abstract (as in the paper)

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
and the mission budget. In flight, a per-stop rule can switch to a faster band
from the observed channel and reorder the remaining stops, each reorder checked
by the same test, and the mule re-plans the rest of its route when the channel
uses up the plan's slack. Updates are merged on each mule and then across mules
at the edge server, with stale updates weighted down, and a coverage term keeps
devices from being starved. In a prototype whose server, mules and devices run as
separate processes over TCP and train a real intrusion-detection model on
CICIoT2023, with flight time, the radio channel and energy simulated, we compare
FeRRy with seven baseline schedulers, including adaptations of FedEx, Oort, FedCS
and a Whittle-index age-of-update scheduler. With six devices, one mule and a
moderate budget, FeRRy reaches the target accuracy 27--31\% sooner than each,
because its plan chooses a long-reach narrow band that removes most of the
transit between stops. Under tight budgets it keeps updates fresher than
age-of-information schedulers, and without its coverage term the share of rounds
that close at twelve devices falls from 99\% to 74\%.
```

## Short hybrid (341 words; adopted, in the paper)

Written 8 Oct 2026 when the user asked for a hybrid nearer 300–350 words. It is
the 441-word hybrid below, shortened. It keeps:
- the numbered components;
- the privacy framing;
- FedEx's full name;
- the headline result with its cause;
- the freshness result, as about a quarter lower age of updates (0.78 against 1.06
  and 1.07, C5);
- the N = 24 result against FedEx's route, as 12% sooner (661 against 752 s);
- the budget result: 70% of missions within a tight budget, and the rest overrun
  it by 28 s on average.

**Correction (8 Oct 2026).** The first adopted text said the overrunning missions
went over by "only 8.5 s on average". The scorer's `sim_budget_overrun_s_mean`
averages over every mission, including the 70% that overrun by zero. From the
mission traces of the 20 compared trials (80 missions, stress budget, N = 6, one
mule), 24 missions overrun, by 28.5 s on average (median 19.7 s, maximum
152.7 s); the FedEx route's overrunning missions go over by 22.3 s. The abstract
and the results text now give 28 s.

The training-time limit is left to the conclusion, which names a compute term
in the plan as future work.

It drops the analysis-plan clause, MAX-AoI by name, the 99% to 74% coverage
result, the planning time at 96 devices, and the learning null. The coverage term
moves into the first component. All of these remain in the paper's body.

```latex
Federated learning (FL) supports privacy by letting devices train a shared model
without sharing their raw data, but assumes every device can reach a central
server. In contested and disaster-zone deployments that assumption fails:
devices that are out of range or poorly connected are skipped repeatedly, and
federation rounds stall. Unmanned aerial vehicles (UAVs) acting as data mules can
carry model updates to and from such devices, but a mule has a finite mission
budget, and its radio decides how far it reaches. We present FeRRy, a
Federated RF-aware Routing framework that treats a mule's reach as a decision,
through three components. First, before each flight, a band-aware mission planner
chooses the contact band class, which trades range for rate, together with the
route, under one feasibility test of per-device deadlines and the mission budget,
with a coverage term that keeps devices from being starved. Second, in flight,
the mule re-plans the rest of its route when the observed channel uses up the
plan's slack, and a per-stop rule can switch to a faster band and reorder the
remaining stops, each reorder checked by the same test. Third, a hierarchical
aggregation layer merges updates on each mule, then across mules at the edge
server, weighting stale updates down. In a prototype whose server, mules and
devices run as separate processes over TCP and train a real intrusion-detection
model on CICIoT2023, with flight time, the radio channel and energy simulated, we
compare FeRRy with seven baseline schedulers, including adaptations of
mobile-transporter federated learning (FedEx), Oort, FedCS and a Whittle-index
age-of-update scheduler. With six devices, one mule and a moderate budget, FeRRy
reaches the target accuracy 27--31\% sooner than each baseline, because its plan
chooses a long-reach narrow band that removes most of the transit between stops.
Under tight budgets, it keeps the age of updates about a quarter lower than
age-of-information schedulers and, with 24 devices, reaches the target 12\%
sooner than FedEx's route. Its plans stay within a tight budget in 70\% of
missions; the rest overrun it by 28\,s on average.
```

## Hybrid candidate (441 words; not yet in the paper)

Written 8 Oct 2026 after the user set the abstract limit at 500 words. It keeps
the adopted abstract's claims and adds back the original's strengths: the
numbered components, FedEx's full name ("mobile-transporter federated
learning") and the privacy framing. It also reports what the extra room allows:
the N = 24 result, decision cost, the learning null and the two limitations.
Every number is in the paper:

- 27–31%: s53x, N = 6, K = 1, knee.
- 99% to 74%: C2, s57, N = 12.
- Fresher updates: C5, s58.
- N = 24 under the stress budget: 661 vs 752 s, s59x claim.
- 1.5 s per plan: 5.11 (a) rerun.
- 36% exploratory gap: FQ vs E3.
- 30% overruns: F under the stress budget.
- Reach under training time: 5.12.

The second component now separates the departure check and re-plan, which every
FeRRy arm runs, from the per-stop rule (FX and FQ).

```latex
Federated learning (FL) supports privacy by letting devices train a shared model
without sharing their raw data, but it assumes that every device can reach a
central server during federation. In contested and disaster-zone deployments that
assumption fails: devices that are out of range or poorly connected are skipped
repeatedly or never reached, and federation rounds stall. Unmanned aerial vehicles
(UAVs) acting as data mules can carry model updates to and from such devices, but
a mule has a finite mission budget, and its radio decides how far it reaches. This
paper presents FeRRy, a Federated RF-aware Routing framework that treats a mule's
reach as a decision, through three components. First, before each flight, a
band-aware mission planner chooses the contact band class, which trades range for
rate, together with the route, under one feasibility test of per-device deadlines
and the mission budget. Second, in flight, the mule checks the rest of its route
at every departure and re-plans it when the observed channel has used up the
plan's slack; a per-stop rule can also switch to a faster band and reorder the
remaining stops, each reorder checked by the same test. Third, a hierarchical
aggregation layer merges updates on each mule and then across mules at the edge
server, weighting stale updates down. A coverage term in the planner keeps
devices from being starved. The prototype runs the edge server, the mules and the
devices as separate processes that communicate over TCP, with flight time, the
radio channel and energy simulated, and the devices train a real
intrusion-detection model on the CICIoT2023 dataset. On paired trials, under an
analysis plan fixed in advance, we compare FeRRy with seven baseline schedulers,
including adaptations of mobile-transporter federated learning (FedEx), Oort,
FedCS, a maximum age-of-information scheduler and a Whittle-index age-of-update
scheduler. With six devices, one mule and a moderate budget, FeRRy reaches the
target accuracy 27--31\% sooner than each baseline, because its plan chooses a
long-reach narrow band that removes most of the transit between stops. Under
tight budgets it keeps updates fresher than the age-of-information schedulers,
and without its coverage term the share of rounds that close at twelve devices
falls from 99\% to 74\%. With 24 devices under a tight budget it is faster than
FedEx's route, and it plans in 1.5\,s for 96 devices. A learned per-stop score
performs like the fixed rule; in an exploratory comparison, its 36\% advantage
over a learned scheduler without a plan comes from the plan. Because FeRRy's plan
prices the mean channel and ignores training time, it overruns tight budgets in
30\% of missions and becomes less reliable when devices need time to train.
```

## Audit of the previous abstract

Each claim in the previous abstract, what the paper shows, and the verdict. Numbers
come from `results/exp5/scores/` (the CSV named in brackets) and the paper's
tables.

| Previous abstract says | What the paper shows | Verdict | In the new abstract |
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

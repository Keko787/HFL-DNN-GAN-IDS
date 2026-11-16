# FeRRy under DDIL conditions — the unified plan

*15 Nov 2026; revised 16 Nov (step 3's guard limited to moves onto narrow, at the user's choice, the rule stated as a second layer on top of the per-stop decision, and an N = 12 cell in step 3 that tests the limited guard); revised again 16 Nov after a review against `hermes/` and P1's kept traces, and to make FQ a full track at the user's request (see [What the 16 Nov review changed](#what-the-16-nov-review-changed)). Nothing here has been built or flown. This page combines [FeRRy_DDIL_Robustness_Plan.md](FeRRy_DDIL_Robustness_Plan.md) (the DDIL plan, revised 14 Nov) and [Experiment_5_Per_Transfer_Band_Plan.md](Experiment_5_Per_Transfer_Band_Plan.md) (the per-transfer plan, 13 Nov) into one plan, and where they disagree it decides. They keep the detail it does not repeat (see [What the two pages keep](#what-the-two-pages-keep)). Every build that changes `hermes/` still needs the user's go-ahead, and step 9's analysis plan is fixed before it flies, never after. In the Builds and Order tables, D-numbers are builds; elsewhere D1–D5 are the baselines, as in the DDIL plan.*

---

## The verdict

The DDIL plan's structure stands, and the per-transfer plan becomes its step 3. Taking that alone would leave the misalignments below open, so this page also changes five things:

1. **Step 0 prices the rule itself**, on the recorded physics, on the rate-following physics and, since 16 Nov, on step 1's own session model. The go/no-go then measures what the rule gains once the fixed-rate charge is gone, and how much of the gain P12 expected was that charge.
2. **The rule is re-specified for step 1's model.** It stays a second decision inside each stop, on top of the per-stop decision, which is unchanged. A guard against moves onto narrow is in, step 1's session model replaces the floor-rate fallback, and the main arms have no idle wait (a bounded wait flies as an arm, 16 Nov).
3. **The band-independence assumption gets a control.** The simulator spreads the three classes' interference a third of a period apart by construction, so when one class is in a trough the other two are not. Switching class gains from that. A shared-interference cell goes into steps 0, 1, 3 and 9.
4. **FQ gets a track** ([The FQ track](#the-fq-track)) and an arm in step 9. Since 16 Nov the track does not wait on the rule: FQ is retrained on step 1's model and on a DDIL family whatever step 0 says.
5. **A criterion for improvement** (C7): the combined arm against F, so step 9 says whether the fixes did anything, not only whether FeRRy keeps pace with the baselines.

P12 is not flown as a separate probe. Its tuning moves to step 3 and its fresh-seed test to step 9.

A review on 16 Nov changed more. The next section lists what and where.

---

## What the 16 Nov review changed

*A read-only review of this page against `hermes/` and P1's kept traces, 16 Nov 2026. The changes are written into the sections named in the last column.*

### What the kept traces show

Mission backhaul, N = 6, knee, 1 MB, per mission over 80 missions an arm (20 trials of 4), from `results/exp5_p1mbh/b3/s515/*_traces` (a scratch tally of the `pass_1_flown` records; D0's replay reproduces it):

| | F, 5 dB | F, 17 dB | FX, 17 dB | H1, 17 dB | FedEx route, 17 dB |
|---|---|---|---|---|---|
| Gated out at arrival, not served | 0.03 | 0.54 | 0.20 | 1.25 | 1.06 |
| Served, then lost to the availability draw | 2.44 | 2.14 | 2.34 | 1.90 | 1.94 |
| Updates merged | 3.54 | 3.31 | 3.45 | 2.65 | 3.00 |
| Hover | 107 s | 133 s | 112 s | 9 s | 10 s |

"Gated out" is the trace's `unreachable`: below the floor, or beyond the class's range. On the default channel it is 0.03 to 0.09 for every arm, so the rise at 17 dB is the floor.

- **F keeps the most updates at 17 dB.** Its deficit is 26 s of hover a mission. The baselines keep their time and lose updates at the gate.
- **FX's hover rises as much as F's** (88 to 112 s, against 107 to 133 s). Choosing the class per stop does not protect a stop from troughs.
- **The availability draw is the largest loss in every arm**, about 40% of the members served. It is a device-side draw keyed by (seed, device, mission round) (`FerryRuntime.uplink_drops`), so a retry or re-poll in the same mission cannot redraw it. A member recovered from the gate still faces it, so recovering all of F's 0.54 adds about 0.3 updates a mission.
- **Time to τ is a coarse measure of this.** τ = 0.71 is reached in about 1.7 missions, and at 20 trials FX − F at 17 dB was 45 s [8, 89] with Holm 0.20 (P1). Hover and updates per mission are the measures the fixes act on directly.

### The holes, and what changed

| | Hole | Change | Where |
|---|---|---|---|
| R1 | **Step 0 priced a model step 1 does not fly.** Pricing (b) lets a trough-started session wait the trough out. Step 1 fails it after `t_out` and retries it at the end of the queue, which is a deferral every arm gets | The replay prices step 1's session model itself, (s), at both `t_out` values, and the rule is sized against it | Step 0 |
| R2 | **A deferral does not leave a trough.** `ContactChannel.interference_db` is the same for every device on a class, so the next device starts in the same trough. Only a class change or a wait leaves it, and under shared interference only a wait | The shared-interference number gates the build on the primary model; a bounded wait flies as an arm; the shared cell joins the tuning grid | Steps 0, 3 |
| R3 | **The frequency plan is used three ways.** The shadowing is shared because the classes share one carrier; the interference phases are per class; step 7a's per-class jamming needs separate centre frequencies | One frequency plan, declared once (decision 9). The interference model and the jammed cells follow it | Step 1, step 7a |
| R4 | **C1 and C7 could not tell fixed from unfixed.** Recorded F already passes C1 at 17 dB. C7 needed a gain of about 50 s to τ, and the go/no-go asks for about 25 s. Step 3 tuned on mean time to τ, where the noise at 20 trials is wider than the differences between settings | C1 gains a margin; C7's family is its two deciding cells, at 80 trials; step 3 tunes and passes on hover and updates per mission; C8 (yield) is proposed | Criteria, step 3 |
| R5 | **C2 had no step that owned it.** F overruns 30% and FX 16% at 17 dB against a 10% bound, step 1's retries lengthen stops, and the rule ignored the budget | A leave-by time per stop, in build D2, that every FeRRy arm keeps | Step 2, step 3 |
| R6 | **Jammed cells against a 4-mission trial.** The jam lasts three missions of four (`n_missions = 4`), the dead-band rule changed only the next plan, and the time comparison drops trials that miss τ | Jammed trials fly 6 missions; the window is in seconds; the rule replans inside the mission; no trial is dropped | Step 7a, step 9 |
| R7 | **The combined arm first flew in step 9**, the pre-registered flight. Steps 2 and 3 chase the same 26 s and were tuned apart | A rehearsal on seen seeds, with each switch left out in turn | Step 9a |
| R8 | **The re-poll was FeRRy's alone**, though the baselines lose about twice as many members at the gate | A shared protocol switch in D1, for every arm, like the retry | Step 1, step 4 |
| R9 | **"Every switch on" mixed the simulator with the planner.** The causal estimate is a planner switch, so it was unclear whether F in step 9 knew the channel | Environment and planner switches are named apart; F-o and FX-o are the oracle references | Step 1 |
| R10 | **The payload.** At 18.8 KB a floor-rate session on narrow takes about 1.25 s, against 66.4 s at 1 MB, so the hover problem may be a 1 MB problem. Step 9 had no measured-payload cell | 5a is read before D1 is built; step 9 gains two measured-payload cells | Step 0, step 9 |
| R11 | **FQ was a gated side track.** It was retrained only if the rule helped it, trained on the default channel's family only, and reported as exploratory; its phase features read the interference period | FQ-s1 and FQ-d train whatever the rule does; a DDIL training family; a family of its own in step 9 (C9); a learned per-transfer decision, gated on a bound from step 0 (user's request, 16 Nov) | The FQ track, step 9 |
| R12 | **Smaller points.** Band-dependent loss on top of partway failure charges low SNR twice; the bursty process was unspecified; the sinusoid raises the SNR for half of each period; the seconds-backhaul default F traces are not where step 0 looks | Each is handled where it arises | Steps 0, 1 |

---

## What was misaligned, and what this page decides

| | Misalignment | Decision | Where |
|---|---|---|---|
| M1 | **Order.** The DDIL plan flies step 3 on step 1's corrected simulator. The per-transfer plan flies its arms on the recorded physics, where a session's rate is fixed at its start | Step 1 first. The rule flies only on step 1's model; its gain on the recorded physics is sized by step 0's replay, not flown | Step 0, step 3 |
| M2 | **Go/no-go.** Both use about 15 s a mission. The per-transfer plan measures it against the recorded pricing and for switching only; the DDIL plan measures switching against the rate-following pricing | One test, revised 16 Nov: the rule itself (switching and deferral) must save at least 15 s of hover a mission at 17 dB against step 1's session model, on the primary interference model | Step 0 |
| M3 | **The rule.** DDIL step 3 serves the best current SNR first, waits up to `w` s, and never moves narrower on one arrival reading. The per-transfer rule is first in, first out with deferral, has no wait, and may move narrower | The per-transfer rule, as a second layer under the unchanged per-stop decision, with the guard (no move onto narrow from a medium or wide stop; limited 16 Nov), no idle wait in the main arms and a bounded wait as an arm (16 Nov), and step 1's session model in place of the floor-rate fallback | Step 3 |
| M4 | **Time.** The DDIL plan budgets step 3 at about 320 trials, 0.8 h. The per-transfer plan's tuning and P12 are about 4,800 trials, 12 h, plus FQ's retraining | Step 3 is about 840 trials, 2.1 h; the FQ track is 380 to 620 trials and 3–6 h of training; step 9 grows to about 22 h. About 42 h in all (16 Nov) | [Order and time](#order-and-time) |
| M5 | **FQ.** The DDIL plan leaves it out; the per-transfer plan has FQ-tx and FQ-txR. Step 9's combined arm is built on F, while the rule is an option of FX and FQ | A full track since 16 Nov: FQ-s1 and FQ-d train after step 1 whatever the rule does, FQ-tx and FQ-dR follow the rule, and the best FQ arm flies in step 9 with a family of its own (C9). The rule stays a setting any FeRRy arm can carry, and the combined arm's base, FX or F, is chosen from step 1 | The FQ track, step 9 |
| M6 | **Status.** The per-transfer plan's stage table calls P12 confirmatory; its Risks section says it is exploratory whatever it shows | Exploratory: step 3 tunes on seeds already seen. Step 9 stays the only pre-registered flight | Step 3, step 9 |
| M7 | **Goal.** The DDIL plan asks for parity (C1: no baseline significantly faster). P12 asks for a lead (FX-tx beats FX and the FedEx route) | Parity is the bar for "works under DDIL". C1's tests are two-sided, so a lead shows as claims where there is one. The new C7 tests improvement over F | [Criteria](#criteria-c1-widened-c2-owned-c7-re-drawn-c8-and-c9-added) |
| M8 | **The independence assumption.** The per-transfer plan names it as a risk and the DDIL plan declares it for jamming, but no cell flies interference that hits every class at once | Declared once, for interference and jamming, as the frequency plan (decision 9). A shared-interference switch in step 1, sized in step 0 and flown in steps 3 and 9; it is the primary model if the plan is nested | Steps 0, 1, 3, 9 |
| M9 | **Backhaul.** P12 flies both backhauls; DDIL steps 1–9 fly the mission backhaul | The mission backhaul for the rule's flights: the rule acts on contacts, the mission backhaul is the paper's main setting, and step 6 is where the seconds backhaul matters. Step 0 reads both backhauls' traces | Step 0, step 3 |

---

## Why this way

- **The fixed-rate charge.** On the recorded physics, a session that starts in a trough on narrow is charged 66.4 s for 1 MB, longer than the 60 s interference period. The rule's main way of gaining is to avoid starting sessions in troughs, which dodges exactly that charge. Step 1's rate-following switch removes much of the same charge for every arm. Flown first on the recorded physics, P12 would probably credit the rule with time that step 1 gives back to F anyway, and its lead would shrink later. Step 0 can size both from the kept traces, so nothing is built until it has been read.
- **The phases.** `_contact_phases` in `hermes/l1/channel_model.py` gives the three classes a seeded shuffle of {0, 1/3, 2/3} of the period, and each class gets its own 1 s noise. The shadowing is shared because the classes share one carrier, and the `ContactChannel` docstring calls the interference a declared model, "not a propagation claim". So in every recorded cell, a trough on one class meets the other two 20 s and 40 s from theirs, the most favourable case for switching. If the classes share the 3.32 GHz carrier's centre frequency, narrow's 1.4 MHz and medium's 5 MHz lie inside wide's 20 MHz, and an interferer on narrow's channel would be in wide's too.
- **FQ.** The paper presents three FeRRy arms, and the learned one is central to it, so the DDIL work has to say what learning does under a harsh channel. On the default channel it added nothing: Study 5.5's verdict was flat (the best learned score at FX's level, `keep_learned = false`), and at N = 6 the slot decides about once a mission. A harsh channel is where a per-stop choice has more to learn, but only if FQ is trained on it. So since 16 Nov FQ is retrained on step 1's model and on a DDIL family whatever the rule does, and flies where it has decisions to make (N = 12, the jammed band).
- **Cost.** Taking P12 whole would add about 11 h of flights for a result step 9 then re-tests. This page keeps step 3 at development size and runs the fresh-seed test once, in step 9.

---

## Step 0, extended — size the charge, the rule and the re-poll (read-only)

*Replaces DDIL step 0 and the per-transfer plan's Stage 0. Revised 16 Nov: pricings (s), (p) and (o) added, the rule moved onto (s), and the go/no-go changed.*

A script over P1's kept traces: `results/exp5_p1mbh/b3/s515/*_traces` (mission backhaul) and `results/exp5/b3/s515/*_traces` (seconds backhaul). For every recorded F and FX session at 17 dB and on the default channel, re-walk each stop's sessions in order, each starting when the one before ends:

| | Pricing | What it tells |
|---|---|---|
| (a) | As recorded: the SNR at the session's start, the floor rate if below the floor | The reference |
| (b) | The rate follows the channel in 1 s steps (`NOISE_BIN_S`), nothing delivered below the floor, no failure | How much of F's 17 dB hover is the fixed-rate charge |
| (c) | (b), on the fastest class that reaches the device from the stop | Switching alone, on the corrected rate |
| (d) | `fastest_defer` with the guard (θ = 2, D_max = 1), on pricing (a) | What the rule would show on the recorded physics |
| (s) | Step 1's session model: (b), but a session fails after `t_out` seconds in a row below the floor and is tried once more at the end of the queue. At `t_out` = 2 s and 10 s, with and without resumable transfers | What every arm gets from step 1, and how much of a deferral's work the shared retry already does |
| (e) | The same rule as (d), on pricing (s) | What step 3 can gain on the model it will fly on |
| (s′), (e′) | (s) and (e), with every class on one interference phase and one noise sequence | The same under shared interference. (e′) is read against (s′) |
| (p) | On (s) and (s′): each member in range but below the floor at arrival is tried once more at the stop's end | What step 1's shared re-poll recovers, for F, FX, H1 and the FedEx route |
| (o) | On (s) and (s′): the best order, class and wait for each stop's sessions, with the channel known in advance | A bound on what any per-transfer policy can gain, hand-built or learned |

Partway failures and the retry are now in the replay, as (s). Step 1 still flies them.

**The traces carry what the rebuild needs** (checked 16 Nov).
- `mule_ready.channel_params.contact` has the salt, phases and epoch.
- Each `pass_1_flown` stop has its position, arrival and end times, class, targets, `unreachable` members and arrival SNRs. The device files have the positions.
- Per-session start times are not recorded, so they are re-walked from the stop's device order. `unreachable` holds members beyond the class's range as well as those below the floor; (p) separates them by distance.
- The replay first reproduces every recorded arrival SNR from the rebuilt channel. If it cannot, the go/no-go's last line applies.
- `results/exp5/b3/s515` has no default-channel F traces (no `n6k1_knee_base__F_traces`), so the seconds-backhaul reads use F's 1, 11 and 17 dB cells and FX's default cell. The go/no-go is on the mission backhaul, where every cell is present.

**Reads** (descriptive):
- hover and sessions completed per mission under each pricing;
- the share of sessions that start below the floor, and how long the below-floor stretches last;
- under (s): sessions that fail, and retries that fail, at each `t_out`;
- (b) with the interference clipped at 0 from above: the sinusoid raises the SNR for half of each period, and this shows how much of (b)'s gain is that half;
- (e) without the guard; (e) over the grid θ ∈ {1.5, 2, 3} × D_max ∈ {1, 2}, reported but not used for the decision;
- (p): members recovered per mission, and how many of them pass the availability draw (recomputed from the trial seed);
- (o) − (e) and (o) − (e′): what the hand-built rule leaves for a learned one.

**Go/no-go.** `t_out` and resumable transfers (decision 2) and the frequency plan (decision 9) are decided before step 0 is read, so the bar is not chosen after its numbers are seen.
- **Step 3's rule (builds B4–B8) is built only if it saves at least 15 s of hover a mission, with no fewer sessions completed, at 17 dB on the mission backhaul, at the primary `t_out`, on the primary interference model:** (e′) against (s′) if that model is shared, (e) against (s) if it is per class. The other model's number is reported beside it.
- If the primary model is per class and (e′) keeps less than half of (e), any gain the paper reports is stated as depending on the per-class phase model, and step 9's shared-interference cell is the test that counts.
- (d) − (e) is reported as the part of the recorded-physics gain that step 1's model removes.
- (p) is a read, not a gate: the re-poll is a small switch in D1 and flies in step 1 whatever (p) shows.
- (o) gates the learned per-transfer decision (step 3c in [The FQ track](#the-fq-track)), not the rule.
- If the channel cannot be rebuilt, (d) to (o) are skipped, (c) against (b) bounds switching alone, and step 3 is built if that is at least 15 s; step 3's development flight then decides.
- As in the DDIL plan: if (b) removes most of F's 17 dB hover rise, step 1's rate change is the main correction.

### Step 5a — read before D1 is built

5a is the DDIL plan's launcher-only flight of P1's radio cells at the measured payload (18.8 KB, knee 120 s; 1,280 trials, 1.6 h). It needs no build, so it flies beside step 0, and both are read before D1 is built.

- **Why first.** At 18.8 KB a floor-rate session on narrow takes about 1.25 s, against 66.4 s at 1 MB. The fixed-rate charge, the trough-started sessions and most of what step 3 acts on shrink by the same factor of 53. The gate does not: a member below the floor at arrival is lost at any payload.
- **Reads.** F's and FX's hover per mission at 5 and 17 dB; members gated out at arrival; P1's tests of F against each baseline at 17 dB.
- **The fork** (decision 10). If F's hover rises by less than 5 s a mission at 17 dB at the measured payload, the hover problem is a 1 MB problem. Recommended then: step 3 and FQ-dR are skipped, the 1 MB hover rise is reported as a measured limit for large payloads, and the plan keeps the steps that act at any payload (step 1 with its re-poll, step 2, step 4, step 7a, and the FQ track's FQ-s1 and FQ-d).

---

## Step 3, unified — decide per transfer

*Replaces DDIL step 3, and the per-transfer plan's Stages 0–5 and P12. Built only if step 0 says go and 5a keeps 1 MB in scope, and flown only on step 1's model (build D1).*

**A second layer, not a replacement.** The per-stop decision stays exactly as each arm makes it now; the rule adds a decision inside each stop, after that one is made.

| Layer | Made | Decides |
|---|---|---|
| Per stop (unchanged) | At arrival, as now: F's plan, FX's covering class, FQ's learned pair | Where the mule hovers and where it goes next, the stop's class, which members are served (the arrival gate), and the class Pass 2's route is flown on |
| Per transfer (added) | Before each session at that stop | Which class serves this device now, among those that reach it from the stop, read above the floor and pass the guard; and whether to serve it now or move it to the back of the queue |

The stop's class anchors the second layer: it is the reference price r_j, it wins ties, and it is always a candidate, so a session served without deferral is never slower than on the stop's class at that instant. With the rule off (θ = ∞, D_max = 0, the stop's class the only candidate) the arm flies as it does today, bit for bit. The per-stop decision does not yet account for the rule: F and FX choose their stops as if every session ran on the stop's class, and only FQ-dR (the FQ track) learns its per-stop choice with the rule on.

**The rule** is the per-transfer plan's `fastest_defer` and its invariants, with five changes for step 1's model:

1. **The guard.** c\* is narrow only if the stop's class is narrow. At a medium or wide stop, narrow is not a candidate; every other class that reaches j from the stop and reads above the floor is. Against "never narrower than the stop's class", the only move this adds is wide to medium at a wide stop. *Limited to moves onto narrow at the user's choice, 16 Nov 2026.*
   - **Why narrow.** In the 6 Nov paired traces' first band changes ([exploratory_band_energy.md](../results/exp5/scores/b2/exploratory_band_energy.md)), moves onto narrow were priced faster and mostly were not: medium to narrow was faster at 5 of 25 FX stops and 7 of 44 FQ stops, and wide to narrow at 0 of 6 FQ stops. Wide to medium won at 14 of 16 FX stops and 11 of 13 FQ stops, and moves to a wider class won at 52 of 55 FX stops and 39 of 48 FQ stops. With the rate following the channel, a long narrow session's cost depends on the channel over its whole length, not at its start; 1 MB on medium takes about 1–2 s at mean SNR, close to what the reading at its start prices.
   - **What the evidence is.** Per-stop arrival choices on the default channel and the recorded physics, not the rule at 17 dB on step 1's model. The wide-to-medium sample is small (29 stops, all at N = 12 and 24, about 1–2 s saved a stop). FX-tx-ng (no guard) measures what the guard costs, and FX-tx-sg (the strict guard) at N = 12 measures what limiting it adds.
   - **θ sets how often a wide stop uses medium.** Medium takes about 1.3 to 2.2 times wide's time at mean SNR (the configuration reference's worked numbers, 0–60 m at 25 m altitude), and the bar θ · r_j is priced on wide. So when wide reads in a trough and medium does not, medium clears the bar for every device at θ = 3, for all but the closest at θ = 2, and only for the farther ones at θ = 1.5; the rest are deferred as before.
2. **No floor-rate fallback.** When no class is eligible, j is served on the stop's class and step 1's session model runs: the session completes, or it fails after `t_out` below the floor and takes the shared retry. The invariant "physics unchanged" now reads: the same physics as every other arm in the cell, step 1's.
3. **A bounded wait, as an arm** (16 Nov). The interference is the same for every device on a class, so a deferral does not leave a trough: the next device starts in it too. A deferral gains from the devices' own shadowing (σ = 4 dB, 7.4 s) and from the seconds the other sessions take, and no more.
   - The main arms keep `w` = 0. Step 1's retry already gives a trough-started session a second try.
   - FX-tx-w adds a wait: when every remaining session is deferred, the mule idles up to `w` seconds before it serves the queue, inside the feasibility test and the stop's leave-by time. `w` = 10 s is proposed; step 0's below-floor stretches set it.
   - It flies where a class change cannot help: the shared-interference cell and the bursty cell whose bursts hit every class.
   - B8 records the sessions served because the whole queue was deferred (`tx_forced`) and the seconds waited (`tx_wait_s`).
4. **Any arm can carry it.** `transfer_rule` is a mule setting, so F, FX, FQ and, as a control, the FedEx route can fly it.
5. **The stop's leave-by time holds** (16 Nov). No deferral, wait or new session starts after it (step 2). This replaces the slack-aware variant, which was to be built only if FX-tx overran more than FX.

The rule's price d_c is still the dwell at the SNR at the decision instant. On step 1's model that is an estimate, not the charge; the guard covers the case where it misleads most. From the per-transfer plan's recommended answers: both passes are priced (`transfer_passes = both`); sensing and switching are free for the main arms, with one cell at a declared retune cost; deferrals are first in, first out. DDIL step 3's best-SNR-first order is dropped, because deferral does the same job in a bounded, deterministic walk. Rescuing members that were below the floor at arrival is step 1's shared re-poll.

**Cells.** Step 1's model with every environment switch on and `t_out` at its primary value. N = 6, one mule, 1 MB, mission backhaul, knee (150 s) unless noted. P1's seeds, already seen, so for tuning only. The references (F, FX, H1, D1–D5) come from step 1's cells on the same seeds and are not re-flown. FQ's cells are in the FQ track.

| Arms | Cells | Trials |
|---|---|---|
| FX-tx at θ ∈ {1.5, 2, 3} × D_max ∈ {1, 2} | default 5 dB, 17 dB, 17 dB with shared interference, bursty (the higher duty cycle) | 480 |
| At the chosen setting: FX-tx1 (Pass 1 only), FX-txd (defer only), FX-tx-ng (no guard), D4-tx (the FedEx route with the rule) | 17 dB, bursty | 160 |
| F-tx (the rule on F, so step 9 can build on F) | default, 17 dB, bursty | 60 |
| FX-tx at a declared retune cost | 17 dB | 20 |
| FX-txd and FX-tx-w (the bounded wait) | 17 dB with shared interference; FX-tx-w on the bursty process too | 60 |
| FX-tx | 17 dB at the stress budget (75 s) | 20 |
| At the chosen setting: FX-tx and FX-tx-sg (the strict guard: never narrower than the stop's class), paired by seed | N = 12, 17 dB, knee, on the recorded N = 12 cells' seeds (already seen) | 40 |
| | | **840, about 2.1 h** (an N = 12 trial costs about two N = 6 trials) |

**Choosing the setting.** θ and D_max by mean hover per mission over the 17 dB, shared-interference and bursty cells, among the settings whose updates merged per mission are no lower than FX's, and with the default cell's hover no higher than FX's. Time to τ = 0.71 is reported beside it and not used: at 20 trials its paired interval is about ±40 s (P1: FX − F at 17 dB, 45 s [8, 89]), wider than the differences between settings. Descriptive, no tests. The choice is written into this page before the remaining arms fly.

**Passing.** FX-tx (or F-tx) passes when, on these cells, its hover per mission is lower than its parent's at 17 dB on the primary interference model or on the bursty process, with the paired 95% interval excluding 0 and updates per mission no lower, and it meets C5's +15 s margin on the default channel. This is the DDIL plan's rule for step 9's switches, with hover per mission as the rule's target metric. Its overrun rate at 17 dB and at stress is reported beside FX's; with the leave-by time on it should not be higher.

**Reads that matter for the paper.**
- FX-txd against FX-tx, and the shared-interference cell: how much of the gain needs the per-class phases.
- FX-tx-w against FX-tx in the shared-interference and bursty cells: what a wait adds where a class change cannot help.
- D4-tx against D4: whether the gain is FeRRy's or the rule's.
- FX-tx-ng against FX-tx: what the guard costs.
- FX-tx against FX-tx-sg at N = 12, with B8's count of wide-to-medium sessions: what the move the limited guard adds is worth. The N = 6 cells cannot show it, because they have few wide stops: F made none and FX 14 of 174 (8%) in the 6 Nov traces (N = 6, knee, default channel, recorded physics), and in P1's 17 dB traces F made none and FX 5 of 83 (6%). FX's share is 22% at N = 12 and the knee in the 6 Nov traces. Without this cell the move would first fly in step 9, the pre-registered flight. If FX-tx is slower than FX-tx-sg here, the strict guard goes into step 9 instead.

---

## The FQ track

*Replaces step 3b. Revised 16 Nov at the user's request: the learned arm is central to the paper, so the track no longer waits on the rule.*

**Where FQ stands.**
- On the default channel, learning added nothing over FX. Study 5.5's verdict was flat: the best learned score was −0.0774, FX's −0.0786 and greedy_1's −0.0716, so `keep_learned = false`.
- At N = 6 the slot decides about once a mission (1.03 decisions a sortie on FQ-g75's held-out N = 6 cell), while the losses at 17 dB happen session by session.
- FQ-g75 was trained on the recorded physics and on `jittery56` (5 dB). On step 1's model, at 17 dB or under a jam, it flies outside the conditions it was trained in.
- Its phase features (`pair_v1`'s `prev_sin` to `arrival_cos`) read the interference period P_c from the channel's configuration. The bursty process has no period, and under C6 a DDIL claim cannot rest on a channel parameter.

So the track gives FQ what it needs to have something to learn: the physics it will fly on, a harsh-channel family to train on, cells where it makes more than one decision a mission, and a per-transfer decision if step 0 shows room for one.

**The arms.**

| Arm | Trained on | Rule | Trains when |
|---|---|---|---|
| FQ-g75 | Recorded physics, `jittery56` (the recorded checkpoint) | off | already trained |
| FQ-s1 | Step 1's physics, `jittery56` | off | after D1, always |
| FQ-d | Step 1's physics, the DDIL family, phase block off | off | after D1, always |
| FQ-tx | FQ-g75's checkpoint, no training | on, at step 3's setting | if step 3 is built |
| FQ-dR | As FQ-d, with the rule on in training and in flight | on | if step 3 is built and the rule passed there |

FQ-s1 changes the physics alone. FQ-d changes the family and drops the phase block together; if FQ-d trails FQ-s1 on the sinusoid cells, FQ-dp (the DDIL family with the phase block on) is trained to tell the two apart. FQ-dR replaces the per-transfer plan's FQ-txR.

**The DDIL family** (build B10; working name `ddil`).
- FerrySim cells at N = 6 and N = 12 at their knees, with step 1's environment switches on: the default channel, 11 and 17 dB, the bursty process at both duty cycles, shared interference, and a narrow-band jam window.
- `FerryCell` gains optional fields for the interference amplitude and step 1's switches. At None they are left out of the cell's JSON, as `field_radius_m` is, so the existing families' hashes do not move.
- n = 2.7 is left out of the family, so step 9's n = 2.7 cell tests the learned arm outside the conditions it trained on.
- Features stay `pair_v1`; FQ-d flies the schema with its `phase` flag off. The family, step 1's settings, the phase flag and the rule's settings go in the checkpoint header, so the loader refuses a checkpoint on an arm that flies other settings.
- FerrySim's headroom report is run on the family first. It is a read, not a gate: it says how far the best per-stop choice is from FX's before any training is spent.

**Training.** As FQ-g75's: γ = 0.75, 4,000 episodes, ten seeds, and the median-validation checkpoint flown. FQ-s1 and FQ-d are 20 trainings at 1–2 h each, 14 at a time: about 2–4 h of machine time. FQ-dR is 10 more, 1–2 h.

**Cells**, 20 trials, step 1's model, seeds already seen:
- N = 6, knee: default, 17 dB, 17 dB with shared interference, bursty (the higher duty cycle), and narrow jammed (6 missions);
- N = 12, knee: 17 dB, where the slot makes more than one decision a mission. FX flies this cell too, as the reference (step 1 has no N = 12 cell).

FQ-g75, FQ-s1 and FQ-d fly all six: 380 trials with FX's, about 1.2 h. FQ-tx and FQ-dR add 240 trials, about 0.7 h, if step 3 is built. The other references come from steps 1, 3 and 7a on the same seeds.

**Reading.**
- FQ-s1 against FQ-g75: what the physics change alone does to a learned arm.
- FQ-d against FQ-s1 and against FX: whether training on harsh channels gives the per-stop choice something FX's rule does not have. This is the track's main read.
- FQ-d in the narrow-jammed cell, against FX with and without the dead-band rule: whether FQ learns to leave a dead class from its own readings (`snr_here`, `offset`), with no rule written for it.
- FQ-dR against FQ-d and against FX-tx: whether the rule and the learned choice add up.
- Judged on hover and updates per mission, and on FerrySim's validation return over the family's held-out episodes (400 episodes, against 20 flown trials). Time to τ is reported.

**Step 9's FQ arm** is the best of these by validation return, with the flown cells as the check. It is recorded before step 9's analysis plan is written and is not changed after.

### Step 3c — a learned per-transfer decision (gated)

- The per-transfer plan left this out of scope: a new action space (class, defer or wait, before each session), a new feature schema (`pair_v2`) and a learner path of its own. It is where a learned policy has the most decisions to make: 6 to 12 a mission at N = 6, against about one for the slot.
- **Gate.** It is built only if step 0's bound leaves room: (o) at least 10 s of hover a mission better than the hand-built rule at 17 dB ((e), or (e′) on a shared primary model). The 10 s is proposed. If the rule already takes most of what the bound allows, a learned one cannot add to it.
- If built, it trains on the DDIL family and flies the FQ track's cells as FQ-t (working name), against FX-tx and FQ-dR. Its size and hours are fixed when the gate is read. It is a large build and is not in the time table below (decision 13).
- If the gate fails, the paper can still say why: the bound shows what was left to learn.

---

## Changes to the DDIL plan's other steps

**Step 1.**
- **Environment and planner switches are named apart** (16 Nov).
  - Environment, the same for every arm: the session model with its retry, the re-poll, resumable transfers, bursty interference, shared interference, the band outage. "Every switch on" means these.
  - Planner: the causal estimate. With the environment switches on, F and FX plan from it, as the Rules require of any arm a DDIL claim rests on.
  - F-o and FX-o (working names) are F and FX reading the channel's own σ and A, as recorded. They fly in the default and 17 dB cells only, to show what not knowing the channel costs.
- **The estimate's prior and its update are declared in D1.** Proposed: the default channel's values as the prior, and an update from every stop's arrival readings, inside a mission as well as between missions. Time to τ = 0.71 takes about 1.7 missions, so an estimate that updates only between missions would fly mostly on its prior.
- **One more environment switch: shared interference**, where every class takes one phase and one noise sequence, so `I_b(t)` is the same for all three (the shadowing is shared already). Under a nested frequency plan (decision 9) it is the primary model and the per-class phases are the sensitivity.
- **One more environment switch: a re-poll, for every arm** (16 Nov). A member in range but below the floor at arrival is solicited once more at the stop's end.
  - It is part of the contact protocol, like the retry, so every arm gets it. It was step 4's, for FeRRy arms only.
  - The baselines lose about twice as many members at the gate as F does at 17 dB, so a re-poll for FeRRy alone would not be a fair comparison.
  - F's stop outlasts the 60 s interference period and a baseline's does not, so F should gain more from it. That gain is F's own.
- **The availability draw is unchanged.** It is device-side and keyed by mission round, so neither the retry nor the re-poll redraws it.
- **Band-dependent loss is flown alone**, as a sensitivity cell, and left out of "every switch on". The session model already fails a session on sustained low SNR; with both on, low SNR is charged twice.
- **The bursty process is specified before D1 is built** (decision 11): the mean burst and gap lengths, the depth, and whether a burst hits one class or all. Proposed: bursts only lower the SNR; the mean burst length comes from step 0's below-floor stretches at 17 dB; the lower duty cycle hits one class at a time and the higher one hits all, so the shared case is not tested on a periodic pattern only.
- **Cells**, 20 trials, 8 arms (F, FX, H1, D1–D5) unless noted:
  - at the knee, every environment switch on: P1's channels (5, 11, 17 dB; n = 2.7; σ = 8 dB), the bursty process at two duty cycles, and 17 dB with shared interference (8 cells, 1,280 trials);
  - F-o and FX-o in the default and 17 dB cells (80 trials);
  - at stress, 17 dB (160 trials);
  - each switch alone at the 17 dB knee: the session model at each `t_out`, the re-poll, band-dependent loss, bursty interference, the causal estimate (6 cells, 960 trials).
- Step 1 becomes about 2,480 trials, 6.0 h.

**Step 2.** This is the main fix for the hover rise. It is the step that changes where the mule hovers, and it does not depend on the interference being periodic or per class.
- **A leave-by time per stop** (16 Nov; build D2), so that C2 has a step that owns it. Each stop gets the plan's predicted end plus its share of the mission's slack (the sharing rule is fixed in D2). After it no session, retry, re-poll, deferral or wait starts, and members not served count as timeouts.
- With the reserve and the re-check after the last Pass-1 stop, it is what has to hold C2's 10% bound. F overruns 30% and FX 16% at 17 dB today, before step 1's retries lengthen the stops.
- Step 2's arm flies with and without it at the 17 dB knee and at stress: 40 more trials, so about 360 trials, 0.9 h.

**Step 4.** The re-poll moved to step 1, for every arm. Step 4 keeps the move-closer stop, and prices it on every class that reaches the device from the new stop, not only a wider one. At one distance narrow keeps about 12 dB more margin than wide (their 233.5 m and 65 m slant ranges at n = 2.2), so under interference that hits every class the stop that rescues a device is closer and no wider. *This last point was added while revising; it was not in the review.*

**Step 7a.**
- **The jammed cells follow the frequency plan** (decision 9).
  - Separate centre frequencies: the cells as the DDIL plan drew them (wide jammed, narrow jammed, a wideband control).
  - One centre frequency, the classes nested: a jammer on narrow's 1.08 MHz takes narrow out, about a quarter of medium's 4.5 MHz and 6% of wide's 18 MHz. A jammer over wide's channel covers all three, so "wide jammed, narrow clear" cannot happen. The five cells are then redrawn in D8: narrow jammed for one and for three windows, a wideband jam at finite power for one and for three windows, and a wideband outage as the control.
  - A wideband jam at finite power is not a no-help case. Every class loses the same dB and narrow has the most margin at any distance, so the test there is whether FeRRy moves closer and narrower while the baselines on wide lose the window. *Added while revising.*
- **Jammed trials fly 6 missions**, not 4, so a three-mission window leaves missions to recover in.
- **The window is in seconds**, the same for every arm: [t0, t0 + 3·T_nom], with T_nom the trial's `t_nom_s` (about 203 s at the N = 6 knee) and t0 = 0 proposed. A window counted in missions would jam an arm with long missions for longer.
- **The dead-band rule acts inside the mission.** When a stop's arrival view finds every in-range member below the floor on the stop's class, the mule replans the rest of the mission without that class at once, through the in-flight replan path (`in_flight_response = replan`), and leaves the class out of later plans as before. Acting on the next plan only would lose the whole first mission, most of a 1.7-mission time to τ.
- **No trial is dropped from the time comparison.** A trial that misses τ is given the trial's end time; D0 adds this beside the paired mean over the trials both arms reach.
- **FQ-d flies the narrow-jammed cell**, in the FQ track.
- 1,300 trials at 6 missions: about 4.7 h (a 6-mission trial costs about 1.5 times a 4-mission one).

**Step 9.**
- **The combined arm's base** is FX if FX passes against F on step 1's model (17 dB or bursty, and C5 on the default channel), otherwise F. It carries the rule (FX-tx or F-tx at step 3's setting) if step 3 passed, and every other switch that passed its own step and stayed in at step 9a.
- **Arms:** the combined arm, F, FX, the FQ track's arm, H1 and D1–D5 (10 arms). In the jammed cells, H1 and the FedEx route on medium are added.
- **Suite:** the DDIL plan's 10 cells, plus 17 dB with shared interference at N = 6, plus two cells at the measured payload (17 dB and narrow jammed, N = 6, knee 120 s) if decision 10 keeps them: 13 cells, 10 at N = 6 and 3 at N = 12.
- **Jammed cells** fly 6 missions with the window in seconds, as step 7a.
- **80 trials** in the two cells C7 rests on, 17 dB and narrow jammed at N = 6 and 1 MB; 40 in the rest.
- **Trials:** about 6,400, about 22 h. Of that, the 6-mission jammed cells are about 2.3 h, the measured-payload cells 2.7 h and the 80-trial cells 2.7 h; each is a separate choice (decision 12). The dry run replaces this estimate.
- **Families:** C1, C3 and C5 as below; C7, C8 and C9 each their own. Hover and updates per mission are pre-registered secondary endpoints for C7, in a family of their own.

### Step 9a — rehearsal (new, 16 Nov)

The combined arm first flies here, on P1's seeds (already seen), not in the pre-registered flight.

- **Arms:** the combined arm, and the combined arm with each switch it carries left out in turn (step 2's pricing and leave-by time, the rule, the move-closer stop, the dead-band rule: up to four). The references come from steps 1 to 7a on the same seeds.
- **Cells**, 20 trials, N = 6, knee: default, 17 dB, 17 dB with shared interference, narrow jammed (6 missions).
- **The rule's setting is re-checked with step 2 on:** step 3's six settings in the 17 dB cell (120 trials). Step 2 moves plans off narrow, which changes what the rule sees.
- About 520 trials, 1.4 h.
- **Outcome**, descriptive, recorded before step 9's analysis plan is written: a switch leaves the combined arm if the arm is no worse without it, on hover and updates per mission, in every rehearsal cell. The arm that results is step 9's arm and is not changed after.

---

## Criteria: C1 widened, C2 owned, C7 re-drawn, C8 and C9 added

Proposed for the user to fix, with the DDIL plan's C1–C6, before step 9's analysis plan is written.

- **C1** adds the shared-interference cell to its disrupted links, and a margin (16 Nov):
  - (a) as before: no baseline significantly faster than FeRRy, and none reaches τ significantly more often;
  - (b) non-inferior: the upper end of the 95% CI of (FeRRy − baseline) time to τ is below +δ, for every baseline in every disrupted cell. δ = 30 s is proposed.
  - Why (b): recorded F at 17 dB passes (a), and it is the arm this plan sets out to fix. It fails (b): F − FedEx route was +16 s [−47, +86] at 20 trials (P1).
  - What (b) can show: at 40 trials that interval's half-width is about 50 s, by P1's intervals, so (b) passes only where FeRRy is about 20 s or more ahead on the mean. A miss is reported as not shown, not as a loss.
- **C2** is owned by step 2's leave-by time.
- **C3**'s jammed cells fly 6 missions with the window in seconds, and no trial is dropped from its time comparisons.
- **C6** covers FQ as well: the FQ arm a DDIL claim rests on is trained and flown with the phase block off, so it does not read P_c.
- **C7, improvement** (re-drawn 16 Nov). The combined arm against F.
  - Primary family, 2 comparisons at 80 trials: 17 dB and narrow jammed, N = 6, 1 MB. C7 holds if both are claims for the combined arm.
  - Secondary family, the other disrupted and jammed cells (both bursty duty cycles, shared interference, n = 2.7 and wide jammed at N = 6; 17 dB and narrow jammed at N = 12): 7 comparisons at 40 trials. C7 also needs no claim against the combined arm there.
  - Why: as one family of 9 at 40 trials, a claim needed a true gain of about 50 s to τ, by P1's intervals (FX − F at 17 dB: 45 s [8, 89], Holm 0.20, at 20 trials). The re-drawn family needs about 30 s. Both figures are rough; step 1's cells replace them before the analysis plan is written.
  - Secondary endpoints, pre-registered, a family of their own: hover per mission and updates merged per mission in the two primary cells.
- **C8, yield** (proposed 16 Nov). Updates merged per mission, FeRRy against each baseline at 17 dB, on the bursty process and with shared interference, at N = 6: one Holm family. C8 holds if FeRRy merges significantly more than every baseline at 17 dB and no cell is a claim against it. This is the claim the recorded runs already point to (F 3.31 a mission at 17 dB, against H1's 2.65 and the FedEx route's 3.00), read where time to τ is too coarse to show it.
- **C9, learning** (proposed 16 Nov). The FQ arm against FX in every disrupted and jammed cell: 9 comparisons, one Holm family, read as claims either way. C9 holds if the FQ arm is a claim over FX at 17 dB or with narrow jammed at N = 6, and no cell is a claim against it. The FQ arm's comparisons against the baselines and against the combined arm are a second family, also read as claims.
- **A lead over the baselines is not a criterion.** C1's tests are two-sided, so where FeRRy is significantly faster it shows as a claim. The per-transfer plan's rough lead (60–70 s on the mission backhaul) is replaced by step 0's numbers.

---

## Builds

The DDIL plan's builds D0–D9, with D3 replaced by the per-transfer plan's B4–B8, and B10 added:

| | Build | Change from the two pages | Size |
|---|---|---|---|
| D0 | Scorer entries and the replay | The replay adds pricings (d), (s), (e), their shared-interference twins, (p) and (o). The scorer adds hover and updates per mission as paired endpoints, and time to τ with a missed trial set to the trial's end | small to medium |
| D1 | Honest DDIL channel | Adds the shared-interference switch and the shared re-poll; names the environment switches apart from the causal estimate, with F-o and FX-o; declares the estimate's prior and updates it inside a mission; takes the bursty process's specification | large |
| D2 | Risk-priced dwell | Adds the leave-by time per stop | medium |
| B4–B8 | The per-transfer rule (replaces D3) | As the per-transfer plan, plus the guard (narrow a candidate only at a narrow stop, with `strict` and `off` modes for FX-tx-sg and FX-tx-ng), step 1's session model in place of the floor-rate fallback, the bounded wait for FX-tx-w, the leave-by time kept, the rule usable on F, FX, FQ and the FedEx route, and `tx_forced`, `tx_wait_s` and `tx_wide_to_medium` columns in B8 | medium in all (B5 medium, the rest small) |
| B9 | FerrySim with the rule | For FQ-dR; the rule's settings in the checkpoint header | small |
| B10 | The DDIL family | FerrySim cells with step 1's switches and harsher channels (optional `FerryCell` fields, existing hashes unchanged); the family, step 1's settings and the phase flag bound in the checkpoint header; the learned arms wired into batch 3 | small to medium |
| D4 | Mobility fallback | The move-closer stop only, priced on every class; the re-poll is D1's | small to medium |
| D8 | Dead-band rule | Replans inside the mission; jammed cells drawn from the frequency plan, 6 missions, the window in seconds | medium |
| D5–D7, D9 | As in the DDIL plan | D9's arm table takes the base chosen for step 9 and the rehearsal's leave-one-out arms | as there |

B4–B8 land as the per-transfer plan's §5o freeze entry (behind switches, no amendment). That holds only if every default is the recorded pipeline in behaviour and in trace output.

---

## Order and time

| Order | What | Needs | ~h of flights |
|---|---|---|---|
| 1 | Step 0 (replay, (a) to (o)) and step 5a; both read before D1 is built | D0; decisions 2, 9 and 10 | 1.6 |
| 2 | D1, then step 1 | go-ahead; decisions 2, 9 and 11 | 6.0 |
| 3 | B10, then FQ-s1 and FQ-d train and fly | D1 | 1.2, plus 2–4 h of training |
| 4 | D8, then step 7a | D1's band outage | 4.7 |
| 5 | D2, then step 2 | D1's causal estimate | 0.9 |
| 6 | B4–B8, then step 3, if step 0 says go and 5a keeps 1 MB in scope | D1; D2's leave-by time | 2.1 |
| 7 | B9, then FQ-tx and FQ-dR, if step 3 is built | step 3 | 0.7, plus 1–2 h of training |
| 8 | Build D4, then step 4 | D1 | 0.9 |
| 9 | D6, then step 6 (optional before step 9) | — | 0.7 |
| 10 | Step 9a, the rehearsal | the steps that passed | 1.4 |
| 11 | D9, then step 9 | the criteria fixed; step 9a's arm; the FQ arm chosen | ~22 |
| 12 | D7, then step 8; D5, then step 5b; step 3c if its gate passes | 5a's result, for 5b; step 0's (o), for 3c | later |

About 42 h of flights before steps 8, 5b and 3c, plus 3–6 h of FQ training.

- **Against 15 Nov.** This page stood at about 29 h. The 13 h more are step 9 (+8), step 7a's 6-mission trials (+1.6), the FQ track (+1.6), step 9a (+1.4), step 1 (+0.6) and step 2 (+0.1). For comparison, the DDIL plan as written needed about 25 h, and adding P12 as the per-transfer plan wrote it would have made about 36 h, plus the retraining.
- **Where it can come down.** Step 9's three additions are 2.3 to 2.7 h each (decision 12). If 5a shows the hover problem is a 1 MB one and step 3 is skipped, about 2.8 h of flights, 1–2 h of training and about 4 build days come off.

Builds take roughly 19 to 23 working days (D1 is 4–5 of them, B4–B8 about 3, B10 1–2). With time to re-fly, that is about 6 weeks on the one host. The DDIL plan's host notes stand: check the host's memory before long runs, and re-fly failed rows under a fresh out-root.

---

## Decisions

**Settled here.** These are the two pages' recommended answers or this page's resolutions; overrule any of them.

| Question | From | Answer |
|---|---|---|
| Fly the rule before or after step 1 | per-transfer 8, DDIL 2 | After, and only if step 0 says go on step 1's own session model and the primary interference model (16 Nov) |
| Step 3's wait, guard and order | DDIL step 3 against the per-transfer rule | A second layer under the unchanged per-stop decision. Guard in, limited to moves onto narrow (user's choice, 16 Nov); no wait in the main arms, a bounded wait as an arm (16 Nov); first in, first out with deferral |
| Pass 2 | per-transfer 1 | Both passes priced; FX-tx1 shows what Pass 2 adds |
| Sensing and switching cost | per-transfer 2 | Zero, plus one cell at a declared retune cost |
| FQ's training | per-transfer 3 | FQ-s1 on `jittery56` and FQ-d on the DDIL family, both on step 1's model, whatever the rule does; FQ-dR if the rule passes (user's request, 16 Nov) |
| Controls | per-transfer 4 | FX-txd and D4-tx both flown |
| The non-inferiority margin | per-transfer 5 | +15 s, as C5 |
| The budget inside a stop | per-transfer 6 | Step 2's leave-by time, kept by every FeRRy arm (16 Nov) |
| P12 | per-transfer Stages 2–5 | Not flown: tuning in step 3, the fresh-seed test in step 9 |
| The re-poll | DDIL step 4 | A shared protocol switch in D1, for every arm (16 Nov) |
| Step 3's tuning and passing metric | per-transfer Stage 2 | Hover and updates per mission; time to τ reported (16 Nov) |
| A rehearsal before step 9 | this page | Step 9a (16 Nov) |
| Jamming and the interference model | DDIL 5 | Both follow one frequency plan (decision 9), with the explicit-signal reference as proposed |

**Still yours.**
1. **The criteria**, C1–C9, before step 9's analysis plan is written: C1's margin δ, C7's two-cell family, C8 and C9 among them.
2. **Step 1's session model:** which `t_out` is primary (about 2 s or 10 s), and whether resumable transfers are on. Needed before step 0 is read.
3. **Re-baselining:** report step 1's numbers beside the recorded ones (recommended), or replace the recorded radio results.
4. **The paper:** which venue it is at, and whether the fixes go in its next version or a follow-up (DDIL decision 6, per-transfer decision 7).
5. **Compression (5b):** in scope, or out until 5a reads.
6. **Arm names.** FX-tx, F-tx, FQ-s1, FQ-d, F-o and the rest are working names.
7. **Scope:** losing the drone mid-mission and the dock losing its server link stay out unless the venue's framing needs them.
8. **The go-ahead** for each build that changes `hermes/`, and for each flight.
9. **The frequency plan** (16 Nov): the three classes on one centre frequency, nested, or on separate centre frequencies. Nested is recommended, because it is what the code states (one `carrier_hz`, shared shadowing). It makes shared interference the primary model and redraws step 7a's cells; separate keeps the per-class phases and the cells as drawn. Needed before step 0 is read.
10. **The DDIL payload** (16 Nov), when 5a is read: 1 MB as in every radio cell so far, the measured 18.8 KB, or both (recommended, with step 9's two measured-payload cells). And whether step 3 is skipped if the hover problem turns out to be a 1 MB one.
11. **The bursty process** (16 Nov): burst and gap lengths, depth, one class or all. Needed before D1 is built.
12. **Step 9's three additions** (16 Nov), each its own choice: 6-mission jammed cells (2.3 h), the measured-payload cells (2.7 h), and 80 trials in C7's two cells (2.7 h).
13. **Step 3c** (16 Nov): whether a learned per-transfer decision is built if step 0's bound leaves room for it. It is a large build.

---

## What the two pages keep

- **[The DDIL plan](FeRRy_DDIL_Robustness_Plan.md):** *Why*, the rules every step follows, C1–C6, the designs of steps 1, 2 and 4–8 (steps 1, 2, 4 and 7a as amended here), and builds D0–D9 apart from D3. Its step 3, its order and time table and its decisions are replaced here; its step 0 and step 9 are amended here.
- **[The per-transfer plan](Experiment_5_Per_Transfer_Band_Plan.md):** the rule's definition and invariants (with this page's five changes), the arms (with the FQ track's arms in place of FQ-txR), the FQ routes, builds B4–B9 and their tests. Its Stages, P12's analysis plan, *What to expect* and its decisions are replaced here.

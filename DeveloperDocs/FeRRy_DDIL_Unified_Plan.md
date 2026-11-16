# FeRRy under DDIL conditions — the unified plan

*15 Nov 2026. Nothing here has been built or flown. This page combines [FeRRy_DDIL_Robustness_Plan.md](FeRRy_DDIL_Robustness_Plan.md) (the DDIL plan, revised 14 Nov) and [Experiment_5_Per_Transfer_Band_Plan.md](Experiment_5_Per_Transfer_Band_Plan.md) (the per-transfer plan, 13 Nov) into one plan, and where they disagree it decides. They keep the detail it does not repeat (see [What the two pages keep](#what-the-two-pages-keep)). Every build that changes `hermes/` still needs the user's go-ahead, and step 9's analysis plan is fixed before it flies, never after. In the Builds and Order tables, D-numbers are builds; elsewhere D1–D5 are the baselines, as in the DDIL plan.*

---

## The verdict

The DDIL plan's structure stands, and the per-transfer plan becomes its step 3. Taking that alone would leave the misalignments below open, so this page also changes five things:

1. **Step 0 prices the rule itself**, on the recorded physics and on the rate-following physics. The go/no-go then measures what the rule gains once the fixed-rate charge is gone, and how much of the gain P12 expected was that charge.
2. **The rule is re-specified for step 1's model:** the narrower-band guard is in, step 1's session model replaces the floor-rate fallback, and there is no idle wait.
3. **The band-independence assumption gets a control.** The simulator spreads the three classes' interference a third of a period apart by construction, so when one class is in a trough the other two are not. Switching class gains from that. A shared-interference cell goes into steps 0, 1, 3 and 9.
4. **FQ gets a track** (step 3b) and an arm in step 9.
5. **A criterion for improvement** (C7): the combined arm against F, so step 9 says whether the fixes did anything, not only whether FeRRy keeps pace with the baselines.

P12 is not flown as a separate probe. Its tuning moves to step 3 and its fresh-seed test to step 9.

---

## What was misaligned, and what this page decides

| | Misalignment | Decision | Where |
|---|---|---|---|
| M1 | **Order.** The DDIL plan flies step 3 on step 1's corrected simulator. The per-transfer plan flies its arms on the recorded physics, where a session's rate is fixed at its start | Step 1 first. The rule flies only on step 1's model; its gain on the recorded physics is sized by step 0's replay, not flown | Step 0, step 3 |
| M2 | **Go/no-go.** Both use about 15 s a mission. The per-transfer plan measures it against the recorded pricing and for switching only; the DDIL plan measures switching against the rate-following pricing | One test: the rule itself (switching and deferral) must save at least 15 s a mission against the rate-following pricing at 17 dB | Step 0 |
| M3 | **The rule.** DDIL step 3 serves the best current SNR first, waits up to `w` s, and never moves narrower on one arrival reading. The per-transfer rule is first in, first out with deferral, has no wait, and may move narrower | The per-transfer rule, with the guard (never narrower than the stop's class), no idle wait (with a trigger to add one), and step 1's session model in place of the floor-rate fallback | Step 3 |
| M4 | **Time.** The DDIL plan budgets step 3 at about 320 trials, 0.8 h. The per-transfer plan's tuning and P12 are about 4,800 trials, 12 h, plus FQ's retraining | Step 3 is about 780 trials, 1.9 h; step 3b is gated (120 trials and 2–4 h of training); step 9 grows to about 14 h. About 29 h in all | [Order and time](#order-and-time) |
| M5 | **FQ.** The DDIL plan leaves it out; the per-transfer plan has FQ-tx and FQ-txR. Step 9's combined arm is built on F, while the rule is an option of FX and FQ | FQ-tx flies in step 3, retraining is gated (step 3b), and the best FQ arm flies in step 9. The rule becomes a setting any FeRRy arm can carry, and the combined arm's base, FX or F, is chosen from step 1 | Step 3b, step 9 |
| M6 | **Status.** The per-transfer plan's stage table calls P12 confirmatory; its Risks section says it is exploratory whatever it shows | Exploratory: step 3 tunes on seeds already seen. Step 9 stays the only pre-registered flight | Step 3, step 9 |
| M7 | **Goal.** The DDIL plan asks for parity (C1: no baseline significantly faster). P12 asks for a lead (FX-tx beats FX and the FedEx route) | Parity is the bar for "works under DDIL". C1's tests are two-sided, so a lead shows as claims where there is one. The new C7 tests improvement over F | [Criteria](#criteria-c1-widened-c7-added) |
| M8 | **The independence assumption.** The per-transfer plan names it as a risk and the DDIL plan declares it for jamming, but no cell flies interference that hits every class at once | Declared once, for interference and jamming. A shared-interference switch in step 1, sized in step 0 and flown in steps 3 and 9 | Steps 0, 1, 3, 9 |
| M9 | **Backhaul.** P12 flies both backhauls; DDIL steps 1–9 fly the mission backhaul | The mission backhaul for the rule's flights: the rule acts on contacts, the mission backhaul is the paper's main setting, and step 6 is where the seconds backhaul matters. Step 0 reads both backhauls' traces | Step 0, step 3 |

---

## Why this way

- **The fixed-rate charge.** On the recorded physics, a session that starts in a trough on narrow is charged 66.4 s for 1 MB, longer than the 60 s interference period. The rule's main way of gaining is to avoid starting sessions in troughs, which dodges exactly that charge. Step 1's rate-following switch removes much of the same charge for every arm. Flown first on the recorded physics, P12 would probably credit the rule with time that step 1 gives back to F anyway, and its lead would shrink later. Step 0 can size both from the kept traces, so nothing is built until it has been read.
- **The phases.** `_contact_phases` in `hermes/l1/channel_model.py` gives the three classes a seeded shuffle of {0, 1/3, 2/3} of the period, and each class gets its own 1 s noise. The shadowing is shared because the classes share one carrier, and the `ContactChannel` docstring calls the interference a declared model, "not a propagation claim". So in every recorded cell, a trough on one class meets the other two 20 s and 40 s from theirs, the most favourable case for switching. If the classes share the 3.32 GHz carrier's centre frequency, narrow's 1.4 MHz and medium's 5 MHz lie inside wide's 20 MHz, and an interferer on narrow's channel would be in wide's too.
- **FQ.** The paper presents three FeRRy arms, and a DDIL result without FeRRy-RL invites the question. FQ-tx needs no training, so it flies; retraining waits until FQ-tx shows that the rule helps FQ.
- **Cost.** Taking P12 whole would add about 11 h of flights for a result step 9 then re-tests. This page keeps step 3 at development size and runs the fresh-seed test once, in step 9.

---

## Step 0, extended — size the charge and the rule (read-only)

*Replaces DDIL step 0 and the per-transfer plan's Stage 0.*

A script over P1's kept traces: `results/exp5_p1mbh/b3/s515/*_traces` (mission backhaul) and `results/exp5/b3/s515/*_traces` (seconds backhaul). For every recorded F and FX session at 17 dB and on the default channel, re-walk each stop's sessions in order, each starting when the one before ends:

| | Pricing | What it tells |
|---|---|---|
| (a) | As recorded: the SNR at the session's start, the floor rate if below the floor | The reference |
| (b) | The rate follows the channel in 1 s steps (`NOISE_BIN_S`), nothing delivered below the floor | How much of F's 17 dB hover is the fixed-rate charge |
| (c) | (b), on the fastest class that reaches the device from the stop | Switching alone, on the corrected rate |
| (d) | `fastest_defer` with the guard (θ = 2, D_max = 1), on pricing (a) | What the rule would show on the recorded physics |
| (e) | The same rule, on pricing (b) | What step 3 can gain on step 1's rate half |
| (e′) | (e), with every class on one interference phase and one noise sequence | How much of (e) rests on the per-class phases |

(d) to (e′) move sessions in time, so they need the traces to carry enough to rebuild the contact channel (salt, distances, stop times); that is checked first. Partway failures and the retry are not in the replay; step 1 flies them.

**Reads** (descriptive): hover per mission under each pricing; the share of sessions that start below the floor, and how long the below-floor stretches last; (e) without the guard; (e) over the grid θ ∈ {1.5, 2, 3} × D_max ∈ {1, 2}, reported but not used for the decision.

**Go/no-go.**
- **Step 3's rule (builds B4–B8) is built only if (e) saves at least 15 s a mission against (b)** at 17 dB on the mission backhaul.
- (d) − (e) is reported as the part of the recorded-physics gain that was the fixed-rate charge.
- If (e′) keeps less than half of (e), step 3 still goes ahead, but any gain the paper reports is stated as depending on the per-class phase model, and step 9's shared-interference cell is the test that counts.
- If the channel cannot be rebuilt, (d) to (e′) are skipped, (c) against (b) bounds switching alone, and step 3 is built if that is at least 15 s; step 3's development flight then decides.
- As in the DDIL plan: if (b) removes most of F's 17 dB hover rise, step 1's rate change is the main correction.

---

## Step 3, unified — decide per transfer

*Replaces DDIL step 3, and the per-transfer plan's Stages 0–5 and P12. Built only if step 0 says go, and flown only on step 1's model (build D1).*

**The rule** is the per-transfer plan's `fastest_defer` and its invariants, with four changes for step 1's model:

1. **The guard.** c\* is never narrower than the stop's class. With the rate following the channel, a long narrow session's cost depends on the channel over its whole length, not at its start. In the 6 Nov paired traces, moves from medium to narrow looked faster at 25 of 25 stops and were faster at 5. FX-tx-ng (no guard) measures what the guard costs.
2. **No floor-rate fallback.** When no class is eligible, j is served on the stop's class and step 1's session model runs: the session completes, or it fails after `t_out` below the floor and takes the shared retry. The invariant "physics unchanged" now reads: the same physics as every other arm in the cell, step 1's.
3. **No idle wait** (`w` = 0). Step 1's retry already gives a trough-started session a second try. B8 records the sessions served because the whole queue was deferred (`tx_forced`). If those fail partway and cost more than 5 s a mission in the development cells, a bounded wait is built before step 9.
4. **Any arm can carry it.** `transfer_rule` is a mule setting, so F, FX, FQ and, as a control, the FedEx route can fly it.

The rule's price d_c is still the dwell at the SNR at the decision instant. On step 1's model that is an estimate, not the charge; the guard covers the case where it misleads most. From the per-transfer plan's recommended answers: both passes are priced (`transfer_passes = both`); sensing and switching are free for the main arms, with one cell at a declared retune cost; deferrals are first in, first out. DDIL step 3's best-SNR-first order is dropped, because deferral does the same job in a bounded, deterministic walk. Rescuing members that were below the floor at arrival stays with step 4's re-poll.

**Cells.** Step 1's model with every switch on and `t_out` at its primary value. N = 6, one mule, 1 MB, mission backhaul, knee (150 s) unless noted. P1's seeds, already seen, so for tuning only. The references (F, FX, H1, D1–D5) come from step 1's cells on the same seeds and are not re-flown.

| Arms | Cells | Trials |
|---|---|---|
| FX-tx at θ ∈ {1.5, 2, 3} × D_max ∈ {1, 2} | default 5 dB, 17 dB, bursty (the higher duty cycle) | 360 |
| At the chosen setting: FX-tx1 (Pass 1 only), FX-txd (defer only), FX-tx-ng (no guard), D4-tx (the FedEx route with the rule) | 17 dB, bursty | 160 |
| F-tx (the rule on F, so step 9 can build on F) | default, 17 dB, bursty | 60 |
| FX-tx at a declared retune cost | 17 dB | 20 |
| FX-tx, FX-txd | 17 dB with shared interference (step 1's new cell) | 40 |
| FX-tx | 17 dB at the stress budget (75 s) | 20 |
| FQ-g75 (not in step 1) and FQ-tx | default, 17 dB, bursty | 120 |
| | | **780, about 1.9 h** |

**Choosing the setting.** As in the per-transfer plan's Stage 2: θ and D_max by mean time to τ = 0.71 over the 17 dB and bursty cells, with the default cell no worse than FX. Descriptive, no tests. The choice is written into this page before the remaining arms fly.

**Passing.** FX-tx (or F-tx) passes when, on these cells, it improves time to τ against its parent at 17 dB or on the bursty process, and meets C5's +15 s margin on the default channel. This is the DDIL plan's rule for step 9's switches. Its overrun rate at 17 dB and at stress is reported beside FX's. If it is higher, the slack-aware variant (refuse a deferral when the stop's slack is short, the per-transfer plan's decision 6) is built before step 9.

**Reads that matter for the paper.**
- FX-txd against FX-tx, and the shared-interference cell: how much of the gain needs the per-class phases.
- D4-tx against D4: whether the gain is FeRRy's or the rule's.
- FX-tx-ng against FX-tx: what the guard costs.

### Step 3b — FQ (gated)

- FQ-g75 was trained on the recorded physics, so on step 1's model it flies outside the conditions it was trained in.
- If FQ-tx improves on FQ-g75 at 17 dB or on the bursty process in step 3 (the same test as above), FQ is retrained twice in FerrySim with step 1's switches on:
  - **FQ-s1**, with the rule off;
  - **FQ-txR**, with the rule on at step 3's setting.
- Both train as FQ-g75 did: `jittery56`, γ = 0.75, 4,000 episodes, ten seeds, and the median-validation checkpoint flown. Step 1's and the rule's settings go in the checkpoint header (build B9). FQ-s1 separates the physics change from the rule.
- 20 trainings at 1–2 h each, 14 at a time: about 2–4 h of machine time. Both then fly step 3's three main cells: 120 trials, 0.3 h.
- If FQ-tx does not pass, there is no retraining.
- Step 9's FQ arm is the best of FQ-g75, FQ-tx, FQ-s1 and FQ-txR on these cells, recorded before step 9's analysis plan is written.

---

## Changes to the DDIL plan's other steps

**Step 1.**
- One more switch in build D1: **shared interference**, where every class takes one phase and one noise sequence, so `I_b(t)` is the same for all three (the shadowing is shared already).
- One more cell: 17 dB at the knee with every switch on and shared interference, 8 arms, 160 trials.
- Step 1 becomes about 2,240 trials, 5.4 h.

**Step 7a.** Unchanged. Its per-class jamming assumption is the one in M8 and is declared with it: narrow jammed while wide stays clear needs the classes on separate centre frequencies.

**Step 9.**
- **The combined arm's base** is FX if FX passes against F on step 1's model (17 dB or bursty, and C5 on the default channel), otherwise F. It carries the rule (FX-tx or F-tx at step 3's setting) if step 3 passed, and every other switch that passed its own step.
- **Arms:** the combined arm, F, FX, step 3b's FQ arm, H1 and D1–D5 (10 arms). In the jammed cells, H1 and the FedEx route on medium are added.
- **Suite:** the DDIL plan's 10 cells, plus 17 dB with shared interference at N = 6 (11 cells: 8 at N = 6, 3 at N = 12).
- **Trials:** 11 × 10 × 40 = 4,400, plus 240 for the medium references: about 4,640, about 14 h, since N = 12 trials cost about twice what N = 6 trials do. The dry run replaces this estimate.
- **Families:** C1, C3 and C5 as in the DDIL plan; C7 its own family. The FQ arm is reported against the baselines and F as exploratory, not as claims.

---

## Criteria: C1 widened, C7 added

Proposed for the user to fix, with the DDIL plan's C1–C6, before step 9's analysis plan is written.

- **C1** adds the shared-interference cell to its disrupted links.
- **C7, improvement.** The combined arm against F in every disrupted and jammed cell: 17 dB, both bursty duty cycles, shared interference, n = 2.7, and wide and narrow jammed at N = 6; 17 dB and narrow jammed at N = 12. That is 9 comparisons, one Holm family. C7 holds if the 17 dB and narrow-jammed cells at N = 6 are claims for the combined arm and no cell is a claim against it.
- **A lead over the baselines is not a criterion.** C1's tests are two-sided, so where FeRRy is significantly faster it shows as a claim. The per-transfer plan's rough lead (60–70 s on the mission backhaul) is replaced by step 0's numbers.

---

## Builds

The DDIL plan's builds D0–D9, with D3 replaced by the per-transfer plan's B4–B8:

| | Build | Change from the two pages | Size |
|---|---|---|---|
| D0 | Scorer entries and the replay | The replay adds pricings (d), (e) and (e′) | small |
| D1 | Honest DDIL channel | Adds the shared-interference switch | large |
| B4–B8 | The per-transfer rule (replaces D3) | As the per-transfer plan, plus the guard, step 1's session model in place of the floor-rate fallback, the rule usable on F, FX, FQ and the FedEx route, and a `tx_forced` column in B8 | medium in all (B5 medium, the rest small) |
| B9 | FerrySim with the rule | Gated on step 3b; the checkpoint header also binds step 1's settings | small |
| D2, D4–D9 | As in the DDIL plan | D9's arm table takes the base chosen for step 9 | as there |

B4–B8 land as the per-transfer plan's §5o freeze entry (behind switches, no amendment). That holds only if every default is the recorded pipeline in behaviour and in trace output.

---

## Order and time

| Order | What | Needs | ~h of flights |
|---|---|---|---|
| 1 | Step 0 (replay, (a) to (e′)) and step 5a | D0 | 1.6 |
| 2 | D1, then step 1 | go-ahead; `t_out` and resumable transfers decided | 5.4 |
| 3 | D8, then step 7a | D1's band outage | 3.1 |
| 4 | D2, then step 2 | D1's causal estimate | 0.8 |
| 5 | B4–B8, then step 3, if step 0 says go | D1 | 1.9 |
| 6 | Build D4, then step 4 | D1 | 0.9 |
| 7 | B9, then FQ-s1 and FQ-txR, if FQ-tx passed | step 3 | 0.3, plus 2–4 h of training |
| 8 | D6, then step 6 (optional before step 9) | — | 0.7 |
| 9 | D9, then step 9 | the criteria fixed; the steps that passed; the bases chosen | ~14 |
| 10 | D7, then step 8; D5, then step 5b | 5a's result, for 5b | later |

About 29 h of flights before steps 8 and 5b, plus 2–4 h of FQ training if step 3b goes ahead. For comparison, the DDIL plan as written needed about 25 h, but its step 3 had no FQ, no controls and no shared-interference cell; adding P12 as the per-transfer plan wrote it would make about 36 h, plus the retraining.

Builds take roughly 14 to 17 working days (D1 is 3–4 of them, B4–B8 about 3). With time to re-fly, that is about 4 weeks on the one host. The DDIL plan's host notes stand: check the host's memory before long runs, and re-fly failed rows under a fresh out-root.

---

## Decisions

**Settled here.** These are the two pages' recommended answers or this page's resolutions; overrule any of them.

| Question | From | Answer |
|---|---|---|
| Fly the rule before or after step 1 | per-transfer 8, DDIL 2 | After, and only if step 0 says go |
| Step 3's wait, guard and order | DDIL step 3 against the per-transfer rule | Guard in; no wait, with a trigger; first in, first out with deferral |
| Pass 2 | per-transfer 1 | Both passes priced; FX-tx1 shows what Pass 2 adds |
| Sensing and switching cost | per-transfer 2 | Zero, plus one cell at a declared retune cost |
| FQ-txR's training | per-transfer 3 | `jittery56` on step 1's model, with FQ-s1 as its control; only if FQ-tx passes |
| Controls | per-transfer 4 | FX-txd and D4-tx both flown |
| The non-inferiority margin | per-transfer 5 | +15 s, as C5 |
| The budget inside a stop | per-transfer 6 | Ignored in v1; the slack-aware variant is built before step 9 if FX-tx overruns more than FX |
| P12 | per-transfer Stages 2–5 | Not flown: tuning in step 3, the fresh-seed test in step 9 |
| Jamming and the interference model | DDIL 5 | Per-class jamming with a wideband control and the explicit-signal reference, as proposed; plus the shared-interference cell |

**Still yours.**
1. **The criteria**, C1–C7, before step 9's analysis plan is written.
2. **Step 1's session model:** which `t_out` is primary (about 2 s or 10 s), and whether resumable transfers are on.
3. **Re-baselining:** report step 1's numbers beside the recorded ones (recommended), or replace the recorded radio results.
4. **The paper:** which venue it is at, and whether the fixes go in its next version or a follow-up (DDIL decision 6, per-transfer decision 7).
5. **Compression (5b):** in scope, or out until 5a reads.
6. **Arm names.** FX-tx, F-tx, FQ-s1 and the rest are working names.
7. **Scope:** losing the drone mid-mission and the dock losing its server link stay out unless the venue's framing needs them.
8. **The go-ahead** for each build that changes `hermes/`, and for each flight.

---

## What the two pages keep

- **[The DDIL plan](FeRRy_DDIL_Robustness_Plan.md):** *Why*, the rules every step follows, C1–C6, the designs of steps 1, 2 and 4–8, and builds D0–D9 apart from D3. Its step 3, its order and time table and its decisions are replaced here; its step 0 and step 9 are amended here.
- **[The per-transfer plan](Experiment_5_Per_Transfer_Band_Plan.md):** the rule's definition and invariants (with this page's four changes), the arms (with FQ-txR as step 3b changes it), the FQ routes, builds B4–B9 and their tests. Its Stages, P12's analysis plan, *What to expect* and its decisions are replaced here.

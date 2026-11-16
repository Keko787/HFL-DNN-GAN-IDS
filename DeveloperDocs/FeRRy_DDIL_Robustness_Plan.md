# FeRRy under DDIL conditions — plan

*Draft, 13 Nov 2026; revised 14 Nov 2026 (see [Revisions](#revisions) at the end). Nothing here has been built or flown. Every step that changes `hermes/` changes the planner or the simulator, so each needs the user's go-ahead before it is built, and its analysis plan below must be fixed before it flies, never after. Related: [Experiment_5_Loss_Probes_Plan.md](Experiment_5_Loss_Probes_Plan.md) (where the measurements below come from), [HERMES_Scheduler_Freeze.md](HERMES_Scheduler_Freeze.md) (Rules 1–3).*

---

## Why

A UAV data mule is meant for denied, disrupted, intermittent and limited (DDIL) networks: the devices have no working link to the server, so the drone carries the model and the updates. FeRRy's lead in Exp 5 comes from one move: it picks the narrow band (232 m reach) so one long hover near the dock serves the whole field, and spends about 15 s flying instead of about 140 s. Under a harsh channel that move stops paying. FeRRy has to keep working, and keep its lead where it can, when the radio is bad.

### What the runs show (Exp 5, P1, mission backhaul, N = 6, one mule, 1 MB, 150 s budget, 20 trials)

Interference amplitude A raised from 5 to 17 dB (`s515.harsher_amp_db`), from `results/exp5_p1mbh/b3/s515/*_scored.csv`:

| | F, 5 dB | F, 17 dB | FX, 17 dB | H1, 5 dB | H1, 17 dB |
|---|---|---|---|---|---|
| Hover (dwell) per mission | 107 s | **133 s** | 112 s | 11 s | 9 s |
| Mission length | 168 s | 193 s | 172 s | 223 s | 219 s |
| Missions over budget | 4% | **30%** | 16% | 3% | 0% |
| Empty missions per trial (mean count, `missions_empty`) | 0.05 | 0.20 | 0.10 | 0.05 | 0.25 |
| Updates merged per mission | 3.54 | 3.31 | 3.45 | 3.46 | 2.65 |
| Time to τ = 0.71 | 194 s | 261 s | 216 s | 282 s | 265 s |

- At 5 dB F beats all six baselines (194 s against 265–282 s, Holm 0.002–0.017). At 17 dB no comparison is a claim either way: F 261 s, baselines 244–277 s, FedEx 244 s. On the seconds backhaul: F 255 s, baselines 264–332 s, FX 209 s ([P1.md](../results/exp5/scores/probes/P1.md)).
- F loses time; the baselines lose updates. The CARE room's loss (baselines about 3× faster, 2 layouts, 1 seed) did not reproduce.
- Path-loss exponent 2.7: F's transit rises from 15 to 35 s on the mission backhaul (narrow reaches less far, so more stops), and on the seconds backhaul FedEx averages 237 s against F's 257 s (no claim).
- The adaptive backhaul carrier helps where the radio is bad: F+L1 beats F at 17 dB, n = 2.7 and 8 dB shadowing (Holm 0.029, 0.021, 0.045), but its controller sees the SNR that decides each loss, so it is close to an oracle.

### Why, in the code

- **One SNR reading per transfer.** `hermes/mission/contact_plan.py` gates each member at arrival (in range and SNR at or above the floor) and prices each session at the SNR at its own start. A session that starts below the floor runs at the floor rate (CQI 1) for its whole length. A narrow-band session to a far device has the least margin, so an interference trough turns it into the slowest transfer the link allows.
- **A member below the floor on arrival is not solicited** and counts as a timeout for that mission.
- **The plan prices the mean.** `FerryRuntime.member_dwell_s` in `hermes/mule/ferry.py` prices dwell at the mean SNR, so the plan cannot see the troughs coming; and nothing re-checks the budget after the last Pass-1 stop, so the longer hovers become overruns.
- **The simulator is wrong in both directions on long transfers.**
  - *Lenient:* a session that starts is never cut off, and the per-contact availability draw (`FerryRuntime.uplink_drops`) ignores the band. A real disrupted link would cut long narrow-band transfers off partway.
  - *Harsh:* a session's rate is held at its start for its whole length. A 1 MB session that starts in a trough on narrow is charged the floor rate (CQI 1, 0.12 Mb/s, 66.4 s), longer than the 60 s interference period. A real link adapts its rate frame by frame and would speed up once the trough passes.
  - Which way the two net out for F is not known. Step 0 sizes the harsh half from the recorded traces, and step 1 corrects both halves together.
- **The planner reads the true channel.** `FerryRuntime.outage_probability` takes σ_sh, σ_I and A from the simulator's `ContactChannel`; a fielded drone would have to estimate them.

### A related loss: devices that need training time (P11 at τ = 0.82, re-scored 9 Nov 2026)

Recorded P11 runs (`results/exp5_rd10`, 10 missions, 40 trials, training median 68 s), scored at τ = 0.82 with `loss_probes.py`'s own tests (scratch script, not in `results/`):

| | F | Baselines (H1, Whittle, FedEx, FedCS) |
|---|---|---|
| Reach 0.82 within 10 missions | 70% spread, 75% stragglers | 97–100% (Holm 0.004, 0.008) |
| Flight for the 10 missions | 896–925 s | 1,729–2,159 s |
| Reach 0.82 at equal flight time (427–436 s) | 28–35% | 5–23% (no claims) |
| Updates merged per mission | 0.89–0.99 | 1.89–2.25 |

The loss is per mission, not per second: F arrives before devices finish training. It matters when sorties, not flight time, are the limit (battery swaps, crewed launches). Step 8 addresses it.

---

## Rules every step follows

- **Behind a switch, default off (Rule 1).** Each change is a mode; the recorded pipeline stays the default, and the goldens and the gate must be unchanged with every switch off.
- **New arms, not changed ones.** Recorded F and FX stay as they are; each planner or flight change flies as an arm of its own, against F, FX and the baselines on the same seeds.
- **No oracle in a DDIL claim.** Any arm a DDIL claim rests on plans from what the mule has observed (step 1's causal estimate), never from `ContactChannel`'s parameters.
- **Exploratory unless pre-registered here.** Each step is scored on its own, Holm per family (the baselines, FeRRy's own arms apart), with `scripts/exp5/loss_probes.py`'s tests: paired time to τ = 0.71 and 0.82, exact McNemar on reach, plus overrun rate, hover time and updates per mission.
- **Trials.** 20 paired trials per cell; 40 for any cell the paper would headline.
- **Out-roots.** A step that changes a setting the job tag does not name gets its own short `--out-root` (kept traces under 250 characters; `launch.py plan` prints the deepest path).
- **Hours** are rough: about 2.4 h per 1,000 trials at N = 6 on this host (batches 1–3), and the launcher's own estimate where a command was dry-run.

---

## What "works under DDIL" means

*Proposed 14 Nov 2026, for the user to fix. Once fixed, these become step 9's analysis plan and are not changed after step 9 flies.* "FeRRy" here is step 9's combined arm.

| | Condition | Proposed criterion | How it is checked |
|---|---|---|---|
| C1 | Disrupted links: the 17 dB sinusoid, the bursty process at both duty cycles, n = 2.7 | No baseline significantly faster than FeRRy to τ = 0.71 or 0.82, and none reaches τ significantly more often | Holm over FeRRy against each baseline; any claim against FeRRy fails C1 |
| C2 | The budget | At most 10% of missions over the knee budget in every C1 and C3 cell (F today: 4% on the default channel, 30% at 17 dB) | Overrun rate per cell, against the bound |
| C3 | A denied band: one class jammed for three missions | Reaches τ in at least 90% of trials in every jammed cell. With wide jammed, significantly faster than the baselines flown on wide. In every jammed cell, no baseline significantly faster, including H1 and the FedEx route flown on medium, which the jam does not touch (P10: the FedEx route on medium took 190 s against F's 179 s at the N = 6 knee, no claim) | Holm; exact McNemar on reach |
| C4 | Degradation | Reported, not tested: time to τ and reach against A at 5, 11 and 17 dB, beside the baselines', so that a cliff shows | Step 1's cells, descriptive |
| C5 | No regression | On the default channel, non-inferior to F: the upper end of the 95% CI of (FeRRy − F) time to τ below +15 s | Paired bootstrap CI |
| C6 | No oracle | Plans only from what the mule observed | A unit test that, with the switch on, the planner never reads `ContactChannel`'s σ_sh, σ_I or A |

If they hold, the claim they support is: FeRRy keeps pace with the best baseline on disrupted links, and when a band is denied it keeps collecting ahead of schedulers tied to that band and level with those that were given a clear one, all without knowing the channel. If some fail, the paper reports which ones, as measured limits.

---

## The steps

| # | Step | Kind | Changes | Trials | ~h |
|---|---|---|---|---|---|
| 0 | Size the fixed-rate charge from the recorded traces | analysis, read-only | none (a script) | 0 | minutes |
| 1 | An honest DDIL channel in the simulator | sim fidelity | `hermes/l1`, `hermes/mission` | ~2,080 | 5.0 |
| 2 | Plan for the troughs, not the mean | planner | `hermes/mule/ferry.py`, feasibility | ~320 | 0.8 |
| 3 | Decide per transfer, not per stop | flight clock | `contact_plan.py`, FX rule | ~320 | 0.8 |
| 4 | Use the drone's mobility | flight clock | re-poll, move-closer stops | ~380 | 0.9 |
| 5 | Shrink the update | launcher, then FL layer | none, then compression | 1,280 + later | 1.6 + later |
| 6 | A causal backhaul controller, and no lost partials | L1, merge | `channel_utility.py`, upload path | ~300 | 0.7 |
| 7a | A jammed band, small version, early | sim model + planner | step 1's band outage, a dead-band rule | ~1,300 | 3.1 |
| 7 | A jammed band, full | — | folded into 7a and 9 | — | — |
| 8 | Price training time | planner | feasibility, plan score | ~80 | 0.5 |
| 9 | Combine and confirm | pre-registered | one combined arm | ~3,840 | ~12 |

Trial counts assume 20 trials a cell (40 for steps 8 and 9) and one new arm per step for steps 2–4, 6 and 8, flown on both channel models where step 1 applies; they are estimates, to be replaced by each command's dry run.

### Step 0 — Size the fixed-rate charge (no code, read-only)

**Question.** How much of F's extra 26 s of hover per mission at 17 dB is the fixed-rate charge, and how much would choosing the class per transfer save once that charge is gone? This decides how much step 1's rate change matters and whether step 3 is worth building.

**How.** P1's kept traces are on the host (`results/exp5_p1mbh/b3/s515/*_traces`, `results/exp5/b3/s515/*_traces`). For every recorded F and FX session at 17 dB and on the default channel, re-price it three ways:
- (a) as recorded, at the SNR at its start;
- (b) with the rate following the channel through the session, in 1 s steps, nothing delivered below the floor;
- (c) as (b), on the fastest class that reaches the device from the stop.

Re-walk each stop's sessions in order, each starting when the one before ends, so the knock-on timing is kept. That needs the traces to carry enough to rebuild the contact channel (salt, distances, stop times), which is checked first. If they do not, each session is priced at its recorded start and the report says the knock-on effects are missing. This is the [per-transfer plan](Experiment_5_Per_Transfer_Band_Plan.md)'s Stage 0 with (b) added.

**Reads** (descriptive only): seconds of hover per mission under (a), (b) and (c); the share of sessions that start below the floor; how long the below-floor stretches last.

**Go/no-go.**
- If (b) removes most of F's 17 dB hover rise, the harsh half dominates, and step 1's rate change is the main correction.
- Step 3 is worth building only if (c) still saves about 15 s a mission or more against (b).

### Step 1 — An honest DDIL channel in the simulator

**Question.** With a rate that follows the channel, transfers that can fail partway, band-dependent losses and a channel the planner must estimate, how far do F and the baselines move from the recorded runs, and in which direction? Every later step is measured on this model.

**Builds (one switch each, default off):**
- **The rate follows the channel, and a session can fail partway.** One switch, so the model is never corrected in one direction only (see *Why, in the code*).
  - A session is walked in 1 s steps (the contact channel's noise bin, `NOISE_BIN_S`) from its start. Each step delivers rate(b, SNR(t)) × Δt, and nothing below the floor.
  - The session ends when its bytes are delivered. It fails if the SNR stays below the floor for more than `t_out` seconds in a row. A failed session is a partial contact and costs the airtime it used.
  - `t_out` at two values: about 2 s, LTE's longest radio-link-failure timer T310 (3GPP TS 36.331; check before citing), and 10 s, a link layer that rides out fades with retransmissions. Either needs a platform citation before publication (Freeze §2). Which one is primary is a decision.
- **One retry, for every arm.** A member whose session fails is tried once more at the end of the stop's queue. If that fails too, it counts as a timeout for the mission, as today. This is part of the contact protocol, so every arm gets it. It is not step 4's re-poll, a FeRRy arm's retry of members that were below the floor at arrival.
- **Resumable transfers (optional, off by itself).** A transfer is sent in chunks with acknowledgements; bytes delivered survive a failure and resume on the retry or at the next contact, as delay-tolerant (custody) transfer would.
- **Band-dependent loss.** A per-session loss probability from the session's SNR, like the backhaul's `loss_probability` in `hermes/l1/channel_model.py`, beside the band-blind availability draw.
- **Bursty interference.** A two-state (Gilbert–Elliott) on/off interference process beside the recorded sinusoid, so no policy can gain by timing a periodic pattern it would never see in the field.
- **A causal channel estimate.** The planner's σ_eff and A come from the SNR the mule observed on earlier stops and missions, not from `ContactChannel` (`FerryRuntime.outage_probability`). Required, not optional, for any arm a DDIL claim rests on (Rules).
- **A band outage** (used by step 7a): band b carries nothing over a window [t0, t1].

**Cells.** N = 6, one mule, 1 MB, mission backhaul; F, FX, H1, D1–D5.
- At the knee, every switch on together, `t_out` at its primary value: P1's channels (5, 11, 17 dB; n = 2.7; σ = 8 dB) and the bursty process at two duty cycles (7 cells, 1,120 trials).
- At stress, 17 dB, every switch on (160 trials).
- Each switch alone at the 17 dB knee, to say which one moves the result: the session model at each `t_out`, band-dependent loss, bursty interference, the causal estimate (5 cells, 800 trials; the band outage waits for step 7a).

The recorded runs with the switches off are the reference; with all switches off, a re-flown cell must match them trial for trial (a check).

**Expected.** The direction is not known.
- The rate change shortens sessions that start in a trough, most of all F's long narrow-band ones.
- The failures and their retries lengthen those same stops, or lose the update.
- Step 0 sizes the first half. Whatever the result, it is the honest starting point for steps 2–9.

### Step 2 — Plan for the troughs, not the mean

**Question.** If the plan prices the dwell the channel actually produces, does F avoid the long narrow hover when the margin is thin, and stop overrunning?

**Change.**
- `member_dwell_s` prices the expected airtime over the SNR's spread (normal with σ_eff, the floor-rate tail included) instead of at the mean; a quantile version is the alternative.
- The feasibility test adds a reserve that grows with the predicted dwell's spread, and the budget is re-checked after the last Pass-1 stop.
- On step 1's causal estimate when that switch is on.

**Arm.** A new F arm (name at build). Cells: as step 1, both channel models (recorded and honest), plus the 5 dB default as a no-regression check.

**Expected.** At 5 dB the same plans as F (the spread is small); at 17 dB more medium and wide missions, fewer overruns (30% today), and a time to τ between F's and FX's. It may not beat the baselines; the aim is to stop losing to the channel.

### Step 3 — Decide per transfer, not per stop

**Question.** Within one stop, can the mule avoid starting transfers in a trough?

**Change (flight clock, a new FX arm):**
- Serve the members with the best current SNR first and defer those below a threshold to the end of the stop.
- Wait a bounded time (at most `w` seconds, inside the feasibility test) when every remaining member is in a trough.
- Pick the band per member, not per stop.
- Never move to a narrower band on one arrival reading: moves from medium to narrow looked faster at 25 of 25 such stops and were faster at 5 (6 Nov paired traces). The rule is `fastest_covering_class` in `hermes/scheduler/policies/cross_heuristic.py`.

**Cells.** As step 1, on both the sinusoid and the bursty process; the gain on the bursty process is the one that would carry to the field.

**Expected.** FX already recovers about 45 s at 17 dB; this should recover more of F's 26 s of extra hover per mission.

### Step 4 — Use the drone's mobility

**Question.** When a member is below the floor, is it cheaper to retry or to move than to write it off for the mission?

**Change.**
- Re-poll members that were below the floor at arrival once more at the end of the stop.
- If one still fails, add a one-device stop closer to it on a wider band when the feasibility test admits it, reusing the hover-stop machinery the age cap uses (`hermes/scheduler/plan/hover.py`).

**Cells.** As step 1, and P4's far-device cells (25, 50, 75% of devices beyond wide's 60 m).

**Expected.** Fewer empty missions at 17 dB (F 20% today) and fewer misses on far devices, at some transit cost.

### Step 5 — Shrink the update

**Question.** Shorter transfers sit through fewer troughs. At the measured model size (18.8 KB) does F keep its lead under harsh radio? Every radio cell so far priced 1 MB.

**5a, launcher only (no code).** P1's radio cells at the measured payload and its knee (120 s, `pilot_outputs.knee_meas_s`). Dry-run 13 Nov 2026: 800 + 480 trials, about 1.6 h, deepest path about 241 characters.

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s515 --out-root results/exp5_dds --set "s515.payload_bytes='measured'" --set "s515.budget='knee_meas'" --set "s515.arms=['F','FX','H1','D1','D2','D3','D4','D5']" --set "s515.amps_db=[11.0]" --set "s515.mission_backhaul_arms=[]"
```

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s515 --out-root results/exp5_ddm --set "s515.payload_bytes='measured'" --set "s515.budget='knee_meas'" --set "s515.arms=['F','FX','H1','D1','D2','D3','D4','D5']" --set "s515.mission_backhaul_arms=[]" --set "s515.backhaul='mission'" --set "s515.amps_db=[]" --set "s515.shadow_sigmas_db=[]"
```

The scorer needs a probe entry for these two flights in `loss_probes.py` (as P1's) before they are scored.

**5b, later (FL layer).** Update compression (8-bit quantization, top-k sparsification of the deltas), with its accuracy cost measured on the real model. Only worth building if 5a shows the gain holds at small payloads and the 1 MB assumption is kept for larger models.

**Expected.** At 18.8 KB F reaches τ in 59 s against 239–272 s with the default channel; the question is how much of that survives 17 dB.

### Step 6 — A causal backhaul controller, and no lost partials

**Question.** Does the backhaul gain (F+L1's three claims) survive a controller that cannot see the loss coming, and what does carrying a failed upload forward buy?

**Change.**
- `AdaptiveChannelController` (`hermes/l1/channel_utility.py`) picks the carrier from SNR observed on earlier uploads, not the SNR that decides this one.
- A failed upload is kept and sent at the next dock instead of being lost (today: lost without retry, about 2% on the fixed backhaul, about a fifth on the time-varying one).

**Cells.** 5.15's seconds-backhaul cells, F+L1 recorded against the causal controller, with and without carry-forward.

**Expected.** Some of the gain stays; carry-forward should help most on the time-varying backhaul.

### Step 7 — A jammed band

*Revised 14 Nov 2026: a small version, step 7a below, now runs right after step 1, and this step's cells move into step 9's suite. The text is kept as the original design.*

**Question.** In a denied-band scenario, does FeRRy's band choice keep collecting while baselines fixed on one band cannot? This is the DDIL case FeRRy's design should win.

**Needs** step 1's band outage. **Cells.** N = 6 and 12, one mule, 1 MB, knee: wide jammed, then narrow jammed, for windows of one and three missions; F, FX, step 2's and step 3's arms, H1, D1–D5 (the baselines fly wide). With wide jammed the baselines' only option is to wait; flying them on medium is the fair second reference (P10's `campaign.baseline_contact_band`).

**Expected.** A clear FeRRy lead with wide jammed, if the planner sees the outage in time (through step 1's causal estimate, or an explicit "band down" signal, which is itself a design choice).

### Step 7a — A jammed band, small version

**Question.** When one band carries nothing for a while, does FeRRy notice from its own contacts and move to another band? Does that keep it collecting where baselines tied to the jammed band cannot? This is the DDIL case FeRRy's band choice should win outright, so it is tested early, not last.

**Change.**
- Step 1's band outage: band b carries nothing over [t0, t1].
- **A dead-band rule (D8), a switch on F and FX.**
  - A band is marked down when every in-range member at a stop on it was below the floor at arrival.
  - The next plan leaves the band out.
  - The band is tried again after `k` missions, or as soon as an arrival view reads it above the floor.
  - The rule infers the outage from the mule's own contacts, never from the outage's schedule.
- A reference arm that gets an explicit "band down" signal at t0. It is an upper bound on what inference can get, not a candidate for step 9.

**Assumptions to declare.**
- The three classes (20, 5 and 1.4 MHz on the one 3.32 GHz contact carrier, `hermes/l1/contact_link.py`) are taken to sit on non-overlapping frequencies, so a jammer can deny one class and leave the others.
- A wideband jammer that hits every class is the control. There, band choice cannot help, and every arm should wait alike.

**Cells.** N = 6, one mule, 1 MB, knee, mission backhaul, step 1's switches on.
- Wide jammed and narrow jammed, each for one and for three missions (4 cells).
- The wideband control, for three missions (1 cell).

**Arms** (13, at 20 trials, so about 1,300 trials):
- F and FX, each with and without the dead-band rule;
- the explicit-signal reference;
- H1 and D1–D5 on wide;
- H1 and the FedEx route on medium (P10's `campaign.baseline_contact_band`), the fair second reference when wide is jammed.

**Expected.**
- **Wide jammed:** F should lead the wide-band baselines, but this tells us little. At N = 6 F rarely flies wide (it picks narrow 89% of the time at 1 MB), so the outage barely touches it, while the wide-band baselines lose the window. Against the medium references F should be about level, as in P10.
- **Narrow jammed:** this is the test FeRRy could lose, because its usual band goes and the baselines' does not. Without the rule, F loses missions until the window ends. With it, F should lose one mission and then fly medium or wide. The rule's worth shows in these cells.

### Step 8 — Price training time

**Question.** If the plan knows when each device's update will be ready, does F stop arriving early, and close the per-mission gap at τ = 0.82?

**Change.** The feasibility test and the plan score read each device's ready time, as FedCS does (`ready_at`, `hermes/mule/fit_clock.py`, used by `hermes/scheduler/policies/fedcs_degraded.py`): a device not ready by the predicted arrival is not worth a stop yet, and the plan may wait at the dock or serve it last.

**Cells.** P11's two compute cells (spread, stragglers), 10 missions, 40 trials: the new arm pairs with P11's recorded arms on the same cells and seeds (`results/exp5_rd10`).

**Expected.** More updates per mission, at some mission time; reach at 0.82 within 10 missions closer to the baselines'.

### Step 9 — Combine and confirm

**Question.** With the fixes that passed their own steps, does FeRRy meet the criteria in *What "works under DDIL" means* across a fixed set of DDIL scenarios, on seeds nobody has looked at?

**The arm.** One FeRRy arm (name at build, decision 8) is F with every switch that passed its step.
- A step passes when, on its own development cells, it improves its target metric at 17 dB or in the jammed cells and meets C5 on the default channel. This is judged descriptively, and recorded before step 9's analysis plan is written.
- The arm's switches are recorded in the manifest. It plans from step 1's causal estimate only (Rules).

**Scenario suite.** Step 1's switches on, mission backhaul, one mule, 1 MB, knee.
- N = 6: default; the 17 dB sinusoid; the bursty process at both duty cycles; n = 2.7; wide jammed and narrow jammed for three missions (7 cells).
- N = 12 at its knee: default; 17 dB; narrow jammed for three missions (3 cells).

**Arms.** The combined arm, F, FX, H1 and D1–D5 (9 arms). In the jammed cells, H1 and the FedEx route on medium are added.

**Trials.** 40 a cell, on a new base seed (`--seed`). Every compared arm is re-flown so all pairs share seeds.
- 10 × 9 × 40 = 3,600 trials, plus 240 for the medium references: about 3,840 in all.
- About 12 h, since N = 12 trials cost about twice what N = 6 trials do. The dry run replaces this estimate.

**Analysis plan.** It is written into this page and committed before the flight, never after.
- One Holm family per tested criterion (C1, C3, C5).
- Paired time to τ = 0.71 and 0.82, and exact McNemar on reach.
- Overrun rate per cell against C2's bound.
- C4 descriptive. C6 is checked by its unit test before the flight.

This is the plan's only pre-registered step; every other step stays exploratory.

---

## Builds

| | Build | What it is | Size |
|---|---|---|---|
| D0 | Scorer entries and the replay | `loss_probes.py` probes for steps 1–9 (and 5a), with τ = 0.82 and overrun, hover and updates per mission in each report; step 0's replay script | small |
| D1 | Honest DDIL channel | Step 1's switches (the rate following the channel with partway failure and the shared retry, resumable transfers, band-dependent loss, bursty interference, the causal estimate, the band outage), recorded in the manifest; tests that each is inert when off and that the goldens are unchanged | large |
| D2 | Risk-priced dwell | Step 2's dwell pricing, reserve and last-stop budget check, as a new arm | medium |
| D3 | Per-transfer flight rule | Step 3's ordering, bounded wait and per-member band, as a new FX arm | medium |
| D4 | Mobility fallback | Step 4's re-poll and move-closer stop | medium |
| D5 | Compression | Step 5b, FL layer | medium |
| D6 | Causal L1 and carry-forward | Step 6 | small to medium |
| D7 | Ready-time pricing | Step 8 | medium |
| D8 | Dead-band rule | Step 7a's rule on F and FX, and the explicit-signal reference arm | small to medium |
| D9 | The combined arm | Step 9's arm as one switch set in the driver's arm table and the manifest | small |

---

## Order and time

| Order | What | Needs | ~h of flights |
|---|---|---|---|
| 1 | Step 0 (replay) and step 5a (no code) | D0 | 1.6 |
| 2 | D1, then step 1 | go-ahead; `t_out` and the retry decided | 5.0 |
| 3 | D8, then step 7a | D1's band outage | 3.1 |
| 4 | D2, then step 2 | D1's causal estimate | 0.8 |
| 5 | D3, then step 3, if step 0 supports it; D4, then step 4 | D1 | 1.7 |
| 6 | D6, then step 6 (optional before step 9) | — | 0.7 |
| 7 | D9, then step 9 | the criteria fixed; the steps that passed | ~12 |
| 8 | D7, then step 8; D5, then step 5b | 5a's result for 5b | later |

About 25 h of flights before steps 8 and 5b, and about 12 to 15 working days of builds (D1 is 3–4 of them). With time to re-fly, that is about 3–4 weeks on the one host, queued with the remaining optional Exp 5 runs ([Experiment_5_Remaining_Runs.md](Experiment_5_Remaining_Runs.md)). Check the host's memory before long runs (the SteelSeries capture leak failed P6's first ±300 m pilot). Failed rows are not retried on resume, so re-fly them under a fresh out-root.

---

## Decisions for you

1. **What "works" means.** C1–C6 as proposed, or your own. They must be fixed before step 9's analysis plan is written.
2. **Order.** As in *Order and time*: the replay and 5a, the honest simulator, the small jammed band, the planner fixes, then step 9. Or the planner fix first (step 2) on the recorded channel, which is quicker but may tune for a model that is wrong in both directions.
3. **The session model's settings.** Which `t_out` is primary (about 2 s or 10 s), the shared single retry, and whether resumable transfers are on.
4. **Re-baselining.** With step 1's switches on, every arm's numbers change. Report them beside the recorded ones (recommended), or replace the recorded radio results in the paper.
5. **Jamming.** Per-class jamming, with the classes on non-overlapping frequencies and a wideband control (recommended). Whether a reference arm gets an explicit "band down" signal (recommended, as an upper bound only; the combined arm infers).
6. **The paper.** Where it stands decides what goes where: IPDPS (rebuttal 3 Dec, revision 18 Jan 2027) or NOMS (paper due 9 Nov 2026, camera-ready 28 Feb 2027). Put the measured harsh-radio limit and any fixes in the next version, or keep the fixes for a follow-up paper.
7. **Compression (5b).** In scope, or out until 5a reads.
8. **Arm names** for steps 2, 3, 4, 7a, 8 and 9.
9. **Scope.** Losing the drone mid-mission and the dock losing its server link are DDIL conditions this plan does not cover. Out, unless the venue's framing needs them.

---

## Revisions

**14 Nov 2026.**
- **Step 1:** the session's rate now follows the channel, in the same switch as partway failures. The recorded model is harsh on sessions that start in a trough as well as lenient about failures, so correcting only one side would bias the result. Also added: a single retry for every arm, `t_out` at two values, and the causal estimate made required for any DDIL claim.
- **New step 0:** a read-only replay of P1's traces that sizes the fixed-rate charge before step 1 is built, and is step 3's go/no-go.
- **New section:** what "works under DDIL" means (C1–C6), proposed.
- **New step 7a:** a small jammed-band test right after step 1, with a dead-band rule (D8). The full step 7's cells move into step 9.
- **New step 9:** one combined FeRRy arm (D9) on a fixed scenario suite with fresh seeds, the plan's only pre-registered step.
- Builds, order, times and decisions updated to match.

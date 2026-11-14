# FeRRy under DDIL conditions — plan

*Draft, 13 Nov 2026. Nothing here has been built or flown. Every step that changes `hermes/` changes the planner or the simulator, so each needs the user's go-ahead before it is built, and its analysis plan below must be fixed before it flies, never after. Related: [Experiment_5_Loss_Probes_Plan.md](Experiment_5_Loss_Probes_Plan.md) (where the measurements below come from), [HERMES_Scheduler_Freeze.md](HERMES_Scheduler_Freeze.md) (Rules 1–3).*

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
| Missions that collect nothing | 5% | 20% | 10% | 5% | 25% |
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
- **The simulator is lenient on long transfers.** A session that starts is never cut off, and the per-contact availability draw (`FerryRuntime.uplink_drops`) ignores the band. A real disrupted link would cut long narrow-band transfers off partway, so these results likely understate the harm.
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
- **Exploratory unless pre-registered here.** Each step is scored on its own, Holm per family (the baselines, FeRRy's own arms apart), with `scripts/exp5/loss_probes.py`'s tests: paired time to τ = 0.71 and 0.82, exact McNemar on reach, plus overrun rate, hover time and updates per mission.
- **Trials.** 20 paired trials per cell; 40 for any cell the paper would headline.
- **Out-roots.** A step that changes a setting the job tag does not name gets its own short `--out-root` (kept traces under 250 characters; `launch.py plan` prints the deepest path).
- **Hours** are rough: about 2.4 h per 1,000 trials at N = 6 on this host (batches 1–3), and the launcher's own estimate where a command was dry-run.

---

## The steps

| # | Step | Kind | Changes | Trials | ~h |
|---|---|---|---|---|---|
| 1 | An honest DDIL channel in the simulator | sim fidelity | `hermes/l1`, `hermes/mission` | ~1,920 | 4.6 |
| 2 | Plan for the troughs, not the mean | planner | `hermes/mule/ferry.py`, feasibility | ~320 | 0.8 |
| 3 | Decide per transfer, not per stop | flight clock | `contact_plan.py`, FX rule | ~320 | 0.8 |
| 4 | Use the drone's mobility | flight clock | re-poll, move-closer stops | ~380 | 0.9 |
| 5 | Shrink the update | launcher, then FL layer | none, then compression | 1,280 + later | 1.6 + later |
| 6 | A causal backhaul controller, and no lost partials | L1, merge | `channel_utility.py`, upload path | ~300 | 0.7 |
| 7 | A jammed band | sim model + probe | needs step 1's band outage | ~1,920 | 4.6 |
| 8 | Price training time | planner | feasibility, plan score | ~80 | 0.5 |

Trial counts assume 20 trials a cell (40 for step 8, as P11) and one new arm per step for steps 2–4, 6 and 8, flown on both channel models where step 1 applies; they are estimates, to be replaced by each command's dry run.

### Step 1 — An honest DDIL channel in the simulator

**Question.** With transfers that can fail partway, band-dependent losses and a channel the planner must estimate, how much worse are F and the baselines than the recorded runs show? Every later step is measured on this model.

**Builds (one switch each, default off):**
- **Mid-session outage.** A session fails if its SNR stays below the floor for more than `t_out` seconds while it runs (the SNR read through the session, not once at its start). A failed session is a partial contact and costs its airtime.
- **Resumable transfers (optional, off by itself).** A transfer is sent in chunks with acknowledgements; bytes delivered survive a failure and resume at the next contact, as delay-tolerant (custody) transfer would.
- **Band-dependent loss.** A per-session loss probability from the session's SNR, like the backhaul's `loss_probability` in `hermes/l1/channel_model.py`, beside the band-blind availability draw.
- **Bursty interference.** A two-state (Gilbert–Elliott) on/off interference process beside the recorded sinusoid, so no policy can gain by timing a periodic pattern it would never see in the field.
- **A causal channel estimate.** The planner's σ_eff and A come from the SNR the mule observed on earlier stops and missions, not from `ContactChannel` (`FerryRuntime.outage_probability`).
- **A band outage** (used by step 7): band b carries nothing over a window [t0, t1].

**Cells.** N = 6, one mule, 1 MB, mission backhaul; F, FX, H1, D1–D5.
- At the knee, every switch on together: P1's channels (5, 11, 17 dB; n = 2.7; σ = 8 dB) and the bursty process at two duty cycles (7 cells, 1,120 trials).
- At stress, 17 dB, every switch on (160 trials).
- Each switch alone at the 17 dB knee, to say which one moves the result (4 switches, 640 trials; the band outage waits for step 7).

The recorded runs with the switches off are the reference; with all switches off, a re-flown cell must match them trial for trial (a check).

**Expected.** F's 17 dB hover gets longer and some of its long sessions fail; the baselines' short sessions fail less. F may now lose at 17 dB, which is the honest starting point.

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

**Question.** In a denied-band scenario, does FeRRy's band choice keep collecting while baselines fixed on one band cannot? This is the DDIL case FeRRy's design should win.

**Needs** step 1's band outage. **Cells.** N = 6 and 12, one mule, 1 MB, knee: wide jammed, then narrow jammed, for windows of one and three missions; F, FX, step 2's and step 3's arms, H1, D1–D5 (the baselines fly wide). With wide jammed the baselines' only option is to wait; flying them on medium is the fair second reference (P10's `campaign.baseline_contact_band`).

**Expected.** A clear FeRRy lead with wide jammed, if the planner sees the outage in time (through step 1's causal estimate, or an explicit "band down" signal, which is itself a design choice).

### Step 8 — Price training time

**Question.** If the plan knows when each device's update will be ready, does F stop arriving early, and close the per-mission gap at τ = 0.82?

**Change.** The feasibility test and the plan score read each device's ready time, as FedCS does (`ready_at`, `hermes/mule/fit_clock.py`, used by `hermes/scheduler/policies/fedcs_degraded.py`): a device not ready by the predicted arrival is not worth a stop yet, and the plan may wait at the dock or serve it last.

**Cells.** P11's two compute cells (spread, stragglers), 10 missions, 40 trials: the new arm pairs with P11's recorded arms on the same cells and seeds (`results/exp5_rd10`).

**Expected.** More updates per mission, at some mission time; reach at 0.82 within 10 missions closer to the baselines'.

---

## Builds

| | Build | What it is | Size |
|---|---|---|---|
| D0 | Scorer entries | `loss_probes.py` probes for steps 1–8 (and 5a), with τ = 0.82 and overrun, hover and updates per mission in each report | small |
| D1 | Honest DDIL channel | Step 1's six switches, recorded in the manifest; tests that each is inert when off and that the goldens are unchanged | large |
| D2 | Risk-priced dwell | Step 2's dwell pricing, reserve and last-stop budget check, as a new arm | medium |
| D3 | Per-transfer flight rule | Step 3's ordering, bounded wait and per-member band, as a new FX arm | medium |
| D4 | Mobility fallback | Step 4's re-poll and move-closer stop | medium |
| D5 | Compression | Step 5b, FL layer | medium |
| D6 | Causal L1 and carry-forward | Step 6 | small to medium |
| D7 | Ready-time pricing | Step 8 | medium |

---

## Order and time

| Order | What | Needs | ~h of flights |
|---|---|---|---|
| 1 | Step 5a (no code) | D0 entry | 1.6 |
| 2 | D1, then step 1 | go-ahead | 4.6 |
| 3 | D2, then step 2 | D1 | 0.8 |
| 4 | D3, then step 3; D4, then step 4 | D1 | 1.7 |
| 5 | Step 7 | D1's band outage, D2, D3 | 4.6 |
| 6 | D6, then step 6 | — | 0.7 |
| 7 | D7, then step 8 | — | 0.5 |
| 8 | D5, then step 5b | 5a's result | later |

About 15 h of flights before 5b, on the one host, queued with the remaining optional Exp 5 runs ([Experiment_5_Remaining_Runs.md](Experiment_5_Remaining_Runs.md)). Check the host's memory before long runs (the SteelSeries capture leak failed P6's first ±300 m pilot).

---

## Decisions for you

1. **Order.** Honest simulator first (step 1), so every later gain is measured on it; or the planner fix first (step 2) on the recorded channel, which is quicker but may tune for a channel that is kinder than the field.
2. **Re-baselining.** With step 1's switches on, every arm's numbers change. Report them beside the recorded ones, or replace the recorded radio results in the paper.
3. **The paper.** Put the measured harsh-radio limit and any fixes in the revision (due 18 Jan 2027), or keep the fixes for a follow-up paper.
4. **Planner knowledge.** Whether step 7's planner may get an explicit "band down" signal, or must infer the outage from what it observes.
5. **Compression (5b).** In scope, or out until 5a reads.
6. **Arm names** for steps 2, 3, 4 and 8.

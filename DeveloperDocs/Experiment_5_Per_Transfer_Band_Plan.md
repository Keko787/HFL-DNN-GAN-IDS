# Per-transfer band adaptation for FX and FQ — plan

*Draft, 13 Nov 2026. Nothing here is built or flown. This page is the plan for the new arms and, once Stage 2 fixes their settings, the analysis plan of their probe (P12). As with the [loss probes](Experiment_5_Loss_Probes_Plan.md): change it before P12 flies, never after. **15 Nov 2026: [FeRRy_DDIL_Unified_Plan.md](FeRRy_DDIL_Unified_Plan.md) makes this page the DDIL plan's step 3. P12 is not flown; this page's Stages, P12 analysis plan and decisions are replaced there, and the rule takes four changes for the DDIL plan's step 1 model.***

---

## How this relates to the DDIL plan

[FeRRy_DDIL_Robustness_Plan.md](FeRRy_DDIL_Robustness_Plan.md), written the same day, has the same idea as its step 3, "Decide per transfer, not per stop", as a new FX arm. This page details that step for FX and FQ: the rule, the arms, the builds and P12's analysis plan. The two differ in three places, to reconcile before anything is built (decision 8):

- **Order.** The DDIL plan first makes the simulator harsher (step 1: sessions can fail partway, band-dependent losses), then flies step 3 on that model. This page flies the arms on the recorded physics, where a session's rate is fixed at its start.
- **A bounded wait.** Step 3 waits up to `w` seconds when every remaining member is in a trough. Here that is left out of v1 (*Not in this plan*).
- **Narrower bands.** Step 3 never moves to a narrower band on one arrival reading. In the 6 Nov paired traces, moves from medium to narrow looked faster at 25 of 25 stops and were faster at 5. That reading was taken at arrival, while each session is charged at its own start. This rule reads the SNR at the session's start, which on the recorded physics is the SNR the session is charged at, so its price is the charge. Under step 1's partway failures that no longer holds, and the guard may be needed here too.

**Revised DDIL plan, 14 Nov 2026.** The DDIL plan's step 1 now lets a session's rate follow the channel, in the same switch as partway failures. Its new step 0 replays P1's traces to size how much of F's 17 dB hover comes from holding a session at its start rate. This rule gains mostly by avoiding that charge, so step 0 is this plan's go/no-go as well: it extends Stage 0 below with the rate-following pricing. If P12 flies on the recorded physics first, its result has to be read with that in mind.

---

## Why

At 17 dB of interference no baseline beats F, but F's lead is gone (P1, `results/exp5/scores/probes/P1.md`). On the default mission-backhaul channel F is significantly faster than all six baselines (Holm p 0.002–0.017). At 17 dB every comparison is "no claim", and the FedEx route is 16 s ahead on the mean (244 against 260 s).

**Where the time goes.** At 17 dB F still flies narrow (94 % of Pass-1 stops) and its hover grows. The FedEx route's does not:

| Mission backhaul, N = 6, knee | Dwell per mission, default → 17 dB | Mission length, default → 17 dB |
|---|---|---|
| F | 107 → 133 s | 168 → 193 s |
| FX | 88 → 112 s | 149 → 172 s |
| FedEx route | 11 → 10 s | 215 → 211 s |

On the seconds backhaul FX's dwell goes from 90 to 107 s, and F's reaches 143 s at 17 dB. The cause is in the contact physics:

- **A session's rate is set at its own start and held to its end** (critic C2, `hermes/mission/host_mission.py`, the commit's step 2). If the session starts in an interference trough, it runs at the trough's rate all the way through.
- **A target below the floor at its session start is charged the floor rate** (critic B12, `ContactPlan.session_dwell_s`). For 1 MB that is 66.4 s on narrow, 15.9 s on medium and 3.9 s on wide.
- **On narrow the swing is large.** A device 50 m out takes 3.7 s at its mean SNR and 66.4 s 17 dB down.
- **F's one long hover sequences the sessions.** Six sessions in a row, over about two 60 s interference periods (P_c), so some start in a trough. The FedEx route's sessions are about 1 s on wide at 0 m. A trough either costs it nothing or drops the device at the gate, which charges no time. So it loses updates (yield 3.49 → 3.00) where F loses time.

**Why FX and FQ don't fix it.** Both choose the class once, at a Pass-1 arrival, and hold it for every session at the stop (`cross_heuristic.py`, `pair_slot.py`). Nothing decides between sessions, and Pass 2 always flies b̄. At N = 6 a collection flight makes one stop in 66–72 of 72 flights (the Phase 5 design, finding 1; `experiments/ferrysim/cells.py`). FQ made 1.03 decisions a sortie on its held-out N = 6 cell. So at N = 6 the slot decides about once a mission, while the losses happen session by session.

**The idea.** Decide before every session: on which class to serve this device now, or whether to serve another device first. Make it an option of FX and FQ.

---

## The rule (`fastest_defer`)

At each session the commit is about to price (a member classed REPLIED, or NO_REPLY, whose push airtime is spent), at the session's start t, for device j:

1. **Price every class.** A class c is eligible if j is within R_planar(c) of the stop and SNR_c(j, t) ≥ floor. Its price is d_c = `link.dwell_s(bytes, c, SNR_c(j, t))`, plus the switch cost s_sw if c is not the class of the previous session (the first session's previous class is the stop's class).
2. **Take the fastest.** c\* is the eligible class with the least d_c. Ties go to the stop's class, then the lower class index.
3. **Serve or defer.** The reference r_j is j's dwell on the stop's class at its mean SNR, the planner's own price.
   - If d\* ≤ θ · r_j, serve j now on c\*.
   - Otherwise, if j has been deferred fewer than D_max times at this stop and another session remains in the queue, move j to the queue's end.
   - Otherwise serve j now on c\*. If no class is eligible, charge the floor rate on the stop's class, as now.

**What it keeps to:**

| Invariant | Why |
|---|---|
| Serves exactly the stop's targets, as gated at arrival | Reach is unchanged and the scope guard holds: learning and rules only reorder within the admitted set (Freeze §1) |
| Never a class beyond j's reach from the stop | The stop was placed for the stop's class; nothing moves |
| A session served without deferral is never slower than on the stop's class at that instant | The stop's class is always a candidate, or the recorded floor-rate charge applies |
| At most D_max deferrals per device per stop; a queue that is all deferred is served | The walk ends; no idle hover in v1 |
| Reads only the SNR at the decision instant | The same reads FX's arrival view makes; no seeded phase, no oracle (`plan_score.py`, option (iii)) |
| Physics unchanged for every arm | The rate is still fixed at session start (critic C2). The arm gains only by choosing when, and on which class, each session starts |
| Off by default | Every recorded arm runs bit for bit as before |
| Deterministic | Fixed tie rules; deferrals are first in, first out |

**Not guaranteed:** a shorter stop overall. A deferred session can meet a worse channel later. The rule is myopic, and the departure check after the stop handles the budget as it does now.

**Assumptions to declare:**
- **Sensing.** Before each session the mule reads j's SNR on every class, at no charge, as `FerryRuntime.arrival_view` does at arrival.
- **Switching.** A class switch costs s_sw (0 s by default), and the device follows the switch. Signalling runs on the current class.
- Both need a platform citation before publication, like the D2 constants (Freeze §2).

**Pass 2.** The rule can also price the deliveries (`transfer_passes = both`). The Pass-2 route is still planned and flown on b̄, and only classes that reach j from the stop are eligible. This relaxes the Phase 4 spec's "Pass 2 flies b̄" for session pricing only; `FerryRuntime.contact_plan` keeps refusing a Pass-2 contact plan on another class.

---

## The arms

| Arm | Slot at arrival | Per-transfer rule | Passes | Role |
|---|---|---|---|---|
| **FX-tx** | FX's fastest covering class | `fastest_defer` | both | the FX arm |
| **FQ-tx** | FQ-g75's learned pair, recorded checkpoint | `fastest_defer` | both | the FQ arm, drop-in |
| **FQ-txR** | FQ-g75 retrained in FerrySim with the rule on | `fastest_defer` | both | the FQ arm, retrained |
| FX-tx1 | FX | `fastest_defer` | Pass 1 only | how much Pass 2 gives |
| FX-txd *(decision 4)* | FX | defer only: c\* is always the stop's class | both | how much comes from deferral alone, which still works if interference hits every class at once |

The names start with FX or FQ, so `loss_probes.is_ferry` keeps them in Ferry's family.

### FQ: two routes

- **FQ-tx: the rule under FQ, no retraining.** FQ picks the stop's class and next stop at arrival as now; the rule then works session by session from that class.
  - FQ-g75's Q values were learned without the rule. Its `dwell_here` feature is still priced at arrival, without the rule. So its band choice is tuned for the old stop, a train/test mismatch.
  - Cheap, and it shows whether the rule alone carries over.
- **FQ-txR: retrained with the rule on.** FerrySim runs the real mule, host mission and contact path (`experiments/ferrysim/inprocess.py`), so the rule is in training automatically once the switch is on.
  - Same family (`jittery56`), γ = 0.75, 4,000 episodes, same behaviour schedule and ten seeds as FQ-g75.
  - The flown checkpoint is picked as FQ-g75's was: the median-validation seed (FQ-g75 flies `g0.75_s3`, `[rl.checkpoints] best` in `params.toml`).
  - New checkpoint directory `results/exp5/checkpoints/5.x-tx/g75`.
  - Features stay `pair_v1`, so the only difference from FQ-g75 is the rule.
  - The rule's settings go in the checkpoint **header**, so the sha binds them and the runner refuses FQ-txR's weights on an arm without the rule, and the other way round (as `purpose` and `revision` are bound, resolution R2).

**Out of scope here:** a learned per-transfer decision, with a new action space (band or defer per session), a new feature schema (`pair_v2`) and a new learner path. That is a follow-up if FQ-txR shows the rule leaves gains on the table.

---

## Builds

Numbered after the probe plan's B0–B3.

| | Build | What | Where | Frozen? | Size |
|---|---|---|---|---|---|
| B4 | The rule | `fastest_defer` as a pure function over a `TransferView` (device, per-class SNR now, per-class eligibility and dwell, the stop's class, r_j, deferrals so far, queue left) returning serve-on-c or defer. Scheduler-side, like `cross_heuristic.py`: no `hermes.l1`, `hermes.mule` or numpy | new `hermes/scheduler/policies/transfer_rule.py`; types in `hermes/scheduler/plan/types.py` | no | small |
| B5 | The commit walks the queue | The step-2 loop prices sessions through the rule when the switch is on. It records each session's class, start, SNR, priced dwell, deferrals, and the stop-class dwell at the same start (for the saved-time column) | `hermes/mission/host_mission.py` (~L1482–1530); `hermes/mission/contact_plan.py` gains optional per-class `snr_at`, `dwell` and reach callables, None by default | not listed, but every arm runs it: the goldens must pin it | medium |
| B6 | Wiring | `FerryRuntime.contact_plan` binds the per-class callables when the switch is on. The supervisor passes the rule and records `pass_1_transfers` and `pass_2_transfers` in `mission_completed` | `hermes/mule/ferry.py`; `hermes/mule/mule_main.py` | **`mule_main.py` is**: behind a switch, as §5k and §5l did | small |
| B7 | Config and arms | `MuleConfig` fields `transfer_rule` (`off` \| `fastest_defer`), `transfer_passes` (`collect` \| `both`), `transfer_theta`, `transfer_max_defers`, `transfer_switch_s`; per-role JSON additive at the defaults. The driver's arm table gains the arms above, and FQ-txR's checkpoint under `[rl.checkpoints]`. The learned arms fly only in batch 2 today (`[rl] keep_learned`), so `s515` in batch 3 needs them wired in | `hermes/processes/config.py`; `experiments/exp4/driver.py`; `scripts/exp5/launch.py`, `params.toml` | no | small |
| B8 | Scoring | Columns `tx_sessions`, `tx_switch_share`, `tx_defers_mean`, `tx_saved_s_mean`, `tx_floor_charges`. P12's flights and its references (each tx arm against its parent, not only F) in the probe scorer | `experiments/analysis/traces_scorer.py`; `scripts/exp5/loss_probes.py` | no | small |
| B9 | FerrySim with the rule | A trainer flag that flies the rule; the header field above; the loader's refusal | `experiments/ferrysim/`; `hermes/scheduler/selector/pair_q.py` (loader check only) | `selector/` is, for its listed files; the check is additive | small |

**The freeze.** Land it as a §5o entry, "behind switches, no amendment" (Rule 3), as Phase 4 (§5k), Phase 5 (§5l) and U11 (§5n) did. That holds only if every default is the recorded pipeline in behaviour and in trace output. If a default has to change, it is an amendment instead.

**Tests:**
- **Defaults:** the 144 `9147211` goldens, UG4's Phase 3 oracles and the FerrySim parity test against `tests/golden/data/p4_plan.json` are all unchanged.
- **Degenerate equivalence:** with θ = ∞, D_max = 0 and the stop's class as the only candidate, FX-tx flies FX bit for bit.
- **The rule:**
  - no device outside the stop's targets;
  - no class beyond reach;
  - a session served without deferral is never slower than on the stop's class at that instant;
  - D_max holds;
  - an all-deferred queue ends;
  - ties are deterministic.
- **The commit:** the clock still lands on the contact's end, charged once for dwell and at most once for listen, with reordered stamps.
- **Pass 2:** with `transfer_passes = collect`, every Pass-2 session is on b̄.
- **FQ-txR:** its checkpoint is refused on an arm without the rule; a FerrySim training smoke test passes with the rule on.

---

## Stages

| Stage | What | Machine | Gate to the next |
|---|---|---|---|
| 0 | **Replay** (optional). If P1's kept traces are still on the host (`results/exp5_p1mbh/b3/s515/*_traces`, `results/exp5/b3/s515/*_traces`; they are not in the release archives, which hold b1, knee and sens only), reprice every recorded F and FX session at 17 dB on each class at its recorded start, and at its mean SNR | minutes | Go if trough-started sessions cost about 15 s a mission or more. A replay bounds the switching part only, not deferral's knock-on effects. No traces: skip to Stage 2 |
| 1 | B4–B9 and their tests | — | All tests pass; goldens unchanged |
| 2 | **Development** on the recorded seeds (P1's trials 0–19, already seen, so for tuning only): FX-tx at θ ∈ {1.5, 2, 3} × D_max ∈ {1, 2}, in the 17 dB cell on both backhauls and the default mission-backhaul cell. 6 × 3 × 20 = 360 trials | ~1 h | Pick θ and D_max by mean time to τ at 17 dB, with the default cell no worse than FX (descriptive, no tests). Write them and P12's trial counts into this page |
| 3 | **Retrain FQ** as FQ-txR with the chosen θ and D_max | as FQ-g75's run (not timed in these docs) | Held-out return on `jit-n6-150` and `jit-n12-180` reported beside FQ-g75's |
| 4 | **P12, confirmatory**, below, on a new base seed | ~11 h | — |
| 5 | Score, report `results/exp5/scores/probes/P12.md`, update the runs list | — | — |

---

## P12 — the analysis plan

**Question.** Does adapting each transfer give back the lead F has on the default channel, at 17 dB and at n_pl = 2.7? Does it cost anything where F already wins?

**Fresh seeds.** P1's 20 seeds have been studied, and Stage 2 tunes on them. P12 flies a new `campaign.base_seed` (`--seed`) and re-flies every compared arm, so all pairs share seeds. The job tag names neither the base seed nor the backhaul, so each backhaul gets its own out-root, as P1's did: `results/exp5_p12s` (seconds) and `results/exp5_p12m` (mission). Both are shorter than P1's `exp5_p1mbh`, but check the deepest trace path with `launch.py plan` (the path limit, probe plan rules).

**Cells.** N = 6, K = 1, knee 150 s, 1 MB, unless stated. The radio cells fly together in `s515`, as P1's did, so they all take the primary block's 40 trials. The blocks differ in how they are analysed, not in trials.

| Block | Cells | Trials |
|---|---|---|
| Primary | 17 dB, and n_pl = 2.7, each on the seconds and the mission backhaul (4 cells) | 40 |
| Secondary | default 5 dB on both backhauls; 11 dB and σ = 8 dB on the seconds backhaul (4 cells) | 40 |
| No regression | N = 6 at the stress budget (75 s), mission backhaul; N = 12 at its knee (180 s), jittery | 20 |

**Arms.**
- **Primary and secondary cells:** F, FX, FQ-g75, FX-tx, FX-tx1, FQ-tx, FQ-txR, H1, D1–D5 (13 arms; 14 with FX-txd).
- **No-regression cells:** F, FX, FQ-g75, FX-tx, FQ-tx, FQ-txR and the FedEx route (7 arms).

**Metrics.** As every probe: time to τ = 0.71, paired by seed over the trials both arms reach it, with a 95 % bootstrap CI; reach at τ by exact McNemar.

**Families and claims.** A claim is CI excluding 0 and Holm p < 0.05.

| Family | Comparisons | Count |
|---|---|---|
| **Primary** (one Holm family) | in each primary cell: FX-tx against FX, and FX-tx against the FedEx route | 8 |
| FX-tx against the rest | FX-tx against H1, D1, D2, D3 and D5 in the primary cells | 20 |
| FQ | FQ-tx and FQ-txR against FQ-g75, and against the FedEx route, in the primary cells | 16 |
| Attribution | FX-tx1 against FX-tx; FX-txd against FX-tx, if flown | 4–8 |
| Reach | each time comparison's reach test, its own family | — |

**No regression.** In the default and no-regression cells, FX-tx against FX and FQ-tx against FQ-g75. An arm is non-inferior when the upper end of the 95 % CI of (tx − parent) time to τ is below +15 s (decision 5).

**Descriptive.** Dwell and length per mission, the switch share per session, deferrals, saved seconds, floor-rate charges, yield, overrun rate and band shares.

**Trials and hours.** 8 × 13 × 40 + 2 × 7 × 20 = 4,440 trials (4,760 with FX-txd), about 11 h at the probes' ~9 s per N = 6 trial on this host (P1: 1,040 trials in 2.5 h). The two N = 12 cells cost more each.

**Sketch.** Arm names exist only after B7. The commands are given without `--yes`, so the launcher lists the jobs first. They are P1's two commands with the new arms, seed and out-roots.

The seconds backhaul (default, 11 dB, 17 dB, n_pl = 2.7, σ = 8 dB):

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s515 --out-root results/exp5_p12s --seed NEW_SEED --trials 40 --set "s515.arms=['F','FX','FQ-g75','FX-tx','FX-tx1','FQ-tx','FQ-txR','H1','D1','D2','D3','D4','D5']" --set "s515.amps_db=[11.0]" --set "s515.mission_backhaul_arms=[]"
```

The mission backhaul (default, 17 dB, n_pl = 2.7):

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s515 --out-root results/exp5_p12m --seed NEW_SEED --trials 40 --set "s515.backhaul='mission'" --set "s515.arms=['F','FX','FQ-g75','FX-tx','FX-tx1','FQ-tx','FQ-txR','H1','D1','D2','D3','D4','D5']" --set "s515.amps_db=[]" --set "s515.shadow_sigmas_db=[]" --set "s515.mission_backhaul_arms=[]"
```

The no-regression cells come from `s53x` (stress) and `s59x` (N = 12), with `--seed NEW_SEED` and out-roots of their own. Their exact commands go here at Stage 2's end.

---

## What to expect, roughly

If the rule removed FX's whole dwell rise at 17 dB (24 s a mission on the mission backhaul, 17.5 s on the seconds backhaul), then at 1.6–1.75 missions to τ FX-tx would gain roughly 30–40 s:

| Backhaul | FX-tx, rough | FedEx route | Lead, rough |
|---|---|---|---|
| Mission | ~175–185 s | 244 s | ~60–70 s |
| Seconds | ~170–180 s | 276 s | ~95–105 s |

P1's leads that cleared Holm at 20 trials were 70–88 s, hence 40 trials in the primary block. This is an estimate from aggregate dwell, not a prediction: Stage 0 or Stage 2 replaces it, and fixes the trial count before P12 is written down.

---

## Risks

- **Power.** A real 30–40 s gain may not clear an 8-comparison Holm family even at 40 trials.
- **Model dependence.**
  - The switching half gains from each class having its own interference phase (`channel_model._contact_phases`). A wideband interferer would remove that; FX-txd shows what is left.
  - Free sensing and switching is generous. A reviewer will ask for a cost (decision 2).
- **Myopia.** A deferral can land in a worse trough and lengthen the stop; at the stress budget that means more overruns.
- **Post hoc.** The rule was designed after seeing P1, so P12 is exploratory, whatever it shows. Fresh seeds keep it honest; they do not make it pre-registered.
- **FQ-tx's mismatch.** Its band choice was learned without the rule, so FQ-tx may understate what FQ can do with it. FQ-txR is the fair test.
- **Host.** P6's ±300 m pilot failed on low memory; the N = 12 cells should not run beside other N = 12 work.

---

## Decisions for you

1. **Pass 2.** Price Pass-2 deliveries with the rule too (recommended), with FX-tx1 to show what Pass 2 adds; or Pass 1 only, which keeps the Phase 4 spec as written.
2. **Sensing and switching cost.** Zero, as FX's arrival view (recommended for the main arms), plus one sensitivity cell at a declared retune cost; or a cost in every arm.
3. **FQ-txR's training family.** `jittery56` only (recommended: it differs from FQ-g75 by the rule alone); or add harsher cells, which suits P12's cells better but confounds the comparison.
4. **Controls.**
   - FX-txd, deferral only (recommended, about 7 % more trials).
   - The rule given to the FedEx route (D4-tx), to show whether the gain is Ferry's or the rule's: optional, about 7 % more.
5. **The non-inferiority margin.** +15 s suggested.
6. **Budget inside a stop.** v1 ignores the budget between sessions, as the commit does today. A variant would refuse a deferral when the stop's slack is short.
7. **The paper.**
   - The rebuttal is due 3 Dec and the revision 18 Jan 2027.
   - P12 could go into the revision as an exploratory result that motivates a limitation's fix, or into a follow-up paper.
   - Building and flying it takes Stages 1–4: about 12 h of machine time, plus the build and FQ's retraining.
8. **The DDIL plan's step 3.** Fly these arms on the recorded physics first, as this page does, which isolates the rule's own effect; or only after the DDIL plan's step 1 changes the simulator, as that plan orders it, which tests the rule where it would have to work. Then decide whether to adopt step 3's bounded wait and its narrower-band guard.

---

## Not in this plan

- A learned per-transfer decision (new action space, `pair_v2`).
- Integrating the rate over a session's 1 s noise bins instead of fixing it at the session's start. That is a physics change for every arm; it needs its own justification and a re-fly of everything it touches. It is now in the DDIL plan's step 1 (revised 14 Nov 2026).
- Rescuing a member that is below the floor on the stop's class at arrival but above it on another class. That changes the arrival gate, so reach as well as time.
- Waiting in the hover for a trough to pass when every remaining session is deferred (an idle time W > 0).

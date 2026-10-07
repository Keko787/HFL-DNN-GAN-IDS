# The age and staleness weight (Layer 3) — how an update gets an age, and what the age is worth

*7 Oct 2026. A reading document in the Joint RL Methods series: it restates the code and the records and decides nothing. It is the detail behind the "merge weight" and "age cap" rows of J4 in [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) §4; the companion on the plan score and the reward is [05_Objective_and_Reward.md](05_Objective_and_Reward.md). Code references are file:line as of commit `39b20b84`; numbers are copied from the named record, none was re-run for this document except where a line says so. Anything I could not check is marked "unverified".*

---

## 0. The short version

Layer 3 weights a collected update by how old it is, and uses the same notion of age to protect devices that have gone too long unserved. Two things were built for that and they are easy to confuse, because they count age in different units for different jobs:

| Age | Unit | Defined as | Used by | Defined in |
|---|---|---|---|---|
| **Merge age** a_i of an update | cluster rounds | version of the θ the mule carried (v) minus the version of the θ the device trained from (b_i) | the staleness factor s(a_i), the per-device cutoff a_max_j, the trace's `ages` | `aggregation_rules.update_age`, line 311 |
| **Partial age** of a partial | cluster rounds | cluster's current version V minus the version the mule carried (v_m) | the cluster's staleness factor s_m | `aggregation_rules.partial_age`, line 491 |
| **Plan age** a_j of a device | the device's own mule's missions | m − U_j: the mission being planned minus the mission whose merge last used j's update (0 if none) | age cap S, coverage weights in V, `age_next` feature, Network AoU | `s3d_age_cap.device_age`, line 148 |

The merge weight is w ∝ n·v·s(age) with an exact zero past a_max. The step it takes is shrunk, not renormalised, when updates are stale. Four rules reuse the machinery (`agg:cutoff`, `agg:asynchfl`, `agg:fedbuff`, `agg:fedex`); `agg:plain` is the untouched legacy path.

**Where it stands.** Built and unit-tested; **not yet flown as a comparison.** Study 5.1 (the rules against each other) is in batch 2 and has not run. The age weights are FedAsync's hinge with constants nobody derived; "derived from the bound" is not done (§13). The one place the merge rules have been contrasted in a scored run is indirect: D4 (FeRRy's merge) against D4fedex (FedEx's merge) on the same tour (§12).

---

## 1. Decisions and quantities

"Hand-set" means a chosen constant with no derivation in the repo; "derived" means computed by a stated rule from other quantities.

| Quantity | Defined where | Inputs | Output | Hand-set or derived | Code path |
|---|---|---|---|---|---|
| Version of θ | cluster round counter at DOWN | `HFLHostCluster._cluster_round` | `mission_slice.issued_round` | derived (a counter) | `host_cluster.py:667`, `:766` |
| basis_version b_i | device | the version on the last `DiscPush` it adopted | echoed on `GradientSubmission` | derived | `client_mission.py:400`, `:510`, `:538` |
| Merge age a_i | mule at close | v (carried), b_i | integer ≥ 0 (None if either unknown, counted as 0) | derived | `aggregation_rules.py:311` |
| a_max_j | mule, once per mission after planning | Φ_j (deadline window), S3c scale, period T | integer cutoff, or None | derived: ⌊Φ_j·s / T⌋, min with fixed `a_max` | `aggregation_rules.py:293`, `mule_main.py:3094` |
| s(a) | per rule | age | factor in (0, 1] | **hand-set** shape and constants (§6) | `aggregation_rules.py:277` |
| Mass n_i·v_i | mule | examples, value proxy | normaliser term M_m | derived; v_i = 1 or raw loss | `aggregation_rules.py:372` |
| Raw weight w_i | mule | mass, s(a_i), cutoff | w_i = mass·s, exactly 0 past a_max | derived | `aggregation_rules.py:401` |
| Δ_m | mule | w_i, Δθ_i, M_m | Σ w_i Δθ_i / M_m | derived | `partial_fedavg.py:109` |
| η, server rate | cluster | Δ_m, s_m | θ ← θ + η·Σ M_m s_m Δ_m / Σ M_m | **hand-set**, 1.0 | `cross_mule_fedavg.py:83` |
| FedProx ρ | device | received θ | (ρ/2)‖θ − θ_recv‖² added to the loss | **hand-set**, 0.01 in 5.1 | `model_task.py:192`, `:262` |
| Plan age a_j | scheduler | mission m, last merged round U_j | m − U_j | derived | `s3d_age_cap.py:148` |
| Cap S | `AgeCapSpec` | S* tool | device capped when age ≥ S − L | derived by a rule from layouts; S = 2 | `plan/types.py:276`, `age_cap_s_star.py` |
| Coverage weight w_j | plan score | plan age, miss streak | max(a_j, 1)·(1 + m_j) | **hand-set** form | `plan_score.py:334` |
| Network AoU | scorer | merged-update history | Σ ω_i · age_i(m) | derived (a metric) | `traces_scorer.py:808` |

---

## 2. How an age is defined and made observable

Before Phase 1 no version travelled with an update, so neither merge could know an update's age (build plan, "Age is not observable yet").

### 2.1 The chain of versions (merge age)

1. **The cluster issues a version with every θ.** The DOWN bundle's `MissionSlice.issued_round` is `_cluster_round` at dispatch (`host_cluster.py:667`); the counter moves only when a cluster round closes (`:766`). A fold that leaves θ unchanged (`expired`, or a FedBuff buffer still filling) does not advance it.
2. **The mule keeps it.** `client_cluster` hands the version to the supervisor just before θ arrives (`client_cluster.py:584` → `mule_main.py:1040`, `_on_model_version`), which stages it with θ. `HFLHostMission.open_round(theta, theta_version=...)` stores it as `_current_theta_version` (`host_mission.py:343-355`); Pass 2 stores the new one via `open_pass_2` (`:623-648`).
3. **It rides the push.** `DiscPush.basis_version` is "the version of `theta_disc`: the cluster round that produced it" (`fl_messages.py:126`); the mule sets it from `_current_theta_version` (`host_mission.py:505`).
4. **The device stores it with its basis, and with each prepared update.** `_set_theta_basis(..., version=)` keeps it (`client_mission.py:606-616`). `train_offline` stamps the prepared update with the basis it trained on (`:400`), because that update may ship after a newer basis arrived; the collect handler echoes the prepared version, not the push's (`:480-510`; the in-session fallback, which principle 14 forbids, uses the push's).
5. **The update names its form.** `GradientSubmission.basis_version` and `update_form` (`fl_messages.py:187-188`); under any age-aware rule the device sends Δθ = θ_after − basis, computed on the device (`client_mission.py:522-527`), so the mule never keeps old models.
6. **The mule computes the age.** `update_age(base_version, basis_version) = max(0, v − b)`; None if either is unknown (`aggregation_rules.py:311`). A basis newer than the mule's θ (another mule delivered a later one) counts as age 0.
7. **The age is recorded.** `PartialAggregate` carries `base_version`, `device_basis_versions`, `device_ages`, `device_weights` (= w_i / M_m), `excluded_devices`, `weight_mass` (M_m) (`types/aggregate.py:32-73`); `MissionRoundCloseLine` carries `basis_version` and `age` (`round_report.py:57`, `host_mission.py:1819`); the mule's trace event `pass_1_merge` writes the rule, base version, devices, ages, weights and exclusions (`processes/mule.py:410-431`).

### 2.2 The plan-side age (own-mule missions)

`FLScheduler.record_merged(device_ids, mission_round)` sets `last_merged_round` for the devices the mission's merge used (`fl_scheduler.py:553`); a CLEAN update the merge cutoff excluded is not passed in, so it does not reset the age. At planning, `device_age = m − U_j` with m the mission being planned (`FLScheduler.mission_round`, first mission 1) and U_j = 0 when never merged (`s3d_age_cap.py:148-179`). It is refused if no round is set, and if a merge is recorded after the mission being planned. It equals the scorer's age of j after mission m if m does not merge j, so a device capped at L = 0 is exactly one the scorer will count at age ≥ S unless the mission serves it.

### 2.3 Why two ages

Decision D5 (build plan, accepted in Phase 1) counted age in cluster rounds so it is global and comparable across mules. Decision D4 (30 Sep, Phase 4) counted the *cap's* age in the device's own mule's missions, "the scorer's unit, not cluster rounds", and the Phase 4 deviations list says so. The two are not the same number:

- With an unbudgeted Pass 2 the mule delivers the current θ to **every** slice device, including those S1 left out (`fl_scheduler.build_pass_2_queue`, `:1367-1400`). A device Pass 1 skipped for three missions has plan age 3, yet its next update carries the current basis: merge age 0.
- With several mules, cluster rounds advance about K times per mission period while a plan age advances by one per own-mule mission.
- The two ages act in opposite directions on purpose: a high *plan* age raises a device's claim to be served (cap, coverage weight); a high *merge* age shrinks what its update is worth.

The build plan's "One mission, end to end" still says a device's age "in cluster rounds" enters demand; that sentence predates D4 and is stale (§13).

### 2.4 Why Pass 2 had to be budgeted for ages to spread

If Pass 2 re-delivers θ to the whole slice, every basis is current and merge ages are 0, so the age-aware rules have nothing to weigh (build plan Phase 1: "Without it ages never spread, because Pass 2 re-delivers θ to the whole slice"). The fix has three parts:

- `pass_2_budget` walks Pass 2 as a second sortie against the mission budget, skipping rather than stopping (`mule_main.py:3118-3146`); skipped devices get a `SKIPPED` delivery line and keep their older basis.
- `train_ahead`: the Pass-1 push asks the device to train on the basis it adopts, on a background thread (`DiscPush.train_ahead`, set from `pass_2_budget` at `mule_main.py:722`; `client_mission._start_train_ahead`, `:404`). Without it a device collected in Pass 1 and skipped in Pass 2 came back at age 0 (the 28 Sep audit finding). A delivery that arrives meanwhile replaces the basis and the stale result is discarded.
- **Plan mode refuses it.** `plan_mode = "ferry"` raises on `pass_2_budget` (`mule_main.py:914`) because a whole-stop walk would deliver nothing at narrow bands (critic B8). So on **F's route** (every plan arm) Pass 2 stays unbudgeted and merge ages are 0 except after a failed or unreached delivery; budgeted Pass 2 exists only on the H1 route. Study 5.1 is built around this: both routes, budgeted Pass 2 on H1's route only (`params.toml [s51]`, `launch.py:934`). My reading, not a measurement: on F's route the age-aware rules should sit very near `agg:plain`, which is also the stated sanity expectation (§12).

---

## 3. The cutoff a_max_j from the device's deadline (decision D5)

```
a_max_j = ⌊ Φ_j · s / T ⌋            Φ_j = effective window,  s = S3c scale (1.0 unless on),  T = period_s
cap_j   = min(a_max_j, fixed a_max)   (either alone if only one is set; None if neither)
```

- `age_cap` (`aggregation_rules.py:293-308`): only `agg:cutoff` has a cutoff; an infinite window (the Oort-style `pref` law, unit U11) cuts nothing. `uses_age_cap` is true only when `a_max` or `period_s` is set (`:195`), so **without a period the cutoff never applies**; Exp 5 passes `--agg-period-t-nom` (T = the cell's T_nom) to every job whose merge is `agg:cutoff`, from every stage after the knee pilot (`launch.py:679-686`, `params.toml campaign.merge_period_t_nom`).
- `effective_window` (`s3_deadline.py:404-420`) returns the window the deadline itself uses, "so the deadline and the cutoff never disagree about Φ": the additive law's floor, the multiplicative law's clamp, `round_s` under F-round, infinity under F-pref. Note it is the *window* Φ_j, not the absolute Deadline(j) = Time + Φ − Idle; the build plan's wording ("converts Deadline(j)") is loose.
- **Snapshot at planning.** `MuleSupervisor._age_caps` (`mule_main.py:3094-3116`) is called once after the plan and the merge uses that snapshot; otherwise the mission's own CLEANs would already have tightened Φ_j (a CLEAN tightens it) and S3c may have moved its scale, cutting an update with a window it was never planned under (28 Sep audit).
- **An illustration of the arithmetic, not a trace reading.** Under the multiplicative law with Φ₀ about six missions' worth of window (build plan, pilot plan of 29 Sep) and T = T_nom, an always-on-time device has Φ multiplied by 0.8 (`beta_on`, `s3_deadline.py:165`) per mission, so ⌊6·0.8^k⌋ gives a_max = 6, 4, 3, 3, 2, 1, 1, 1, 1, 0 for k = 0..9 on-time deliveries (clamped at Φ_min = 5 s in the law's unit). A reliable device therefore gets a *tighter* cutoff. In a 4-mission trial a_max stays ≥ 2. The same point at Exp 4's wall clock: Φ = 60 s over a ~10 s mission gives about 6 rounds and "does not bind in a 4-mission trial" (build plan Phase 1 status; Configuration Reference §14).
- **Cluster side uses only the fixed cap.** `partial_staleness` zeroes a partial only when `spec.a_max` is set and the partial is older (`:498-505`); the per-device cap exists at the mule only. Exp 5 sets a period, not `a_max`, so the cluster never cuts a whole partial by age there; it only discounts it.

This is the deadline's third role in contribution C3: it admits (S3b), orders (S3), and cuts off the merge weight.

---

## 4. The merge on the mule

For the age-aware rules the mule forms (`merge_on_mule`, `aggregation_rules.py:409-484`; docstring `:19-27`):

```
w_i = n_i · v_i · s(a_i)           exactly 0.0 if a_i > cap_j        (update_weights, :372-402)
M_m = Σ_{admitted} n_i · v_i       staleness-free, admitted updates only
Δ_m = Σ_{admitted} w_i · Δθ_i / M_m        (partial_fedavg_delta, partial_fedavg.py:109-190)
```

- **Why the normaliser is staleness-free.** Dividing by Σ w_i would cancel a common staleness factor and send a uniformly stale mission's full mean. Dividing by M_m makes two updates both of age 3 move θ by s(3) times their mean (`:464-466` comment; tests `test_a_uniformly_stale_mission_shrinks_the_step`, `test_an_excluded_update_does_not_move_the_admitted_weights`). An excluded update adds nothing to M_m either, so it neither moves nor dilutes the admitted ones.
- **Value proxy v_i** (`:119-121`, `:330-348`): `uniform` (1, the default and what Exp 5 flies: no `--agg-value` is passed anywhere in `scripts/exp5`) or `loss` (the update's raw local loss, taken after the cutoff; a missing or non-positive loss counts as the mean of the admitted updates' known losses, 1.0 if none). Raw, not divided by a mean, so partials from different mules combine exactly as one flat merge over all their devices.
- **Zero-example updates** are dropped silently; if every update is past its cutoff the merge raises `PartialFedAvgError`, which the mule treats like a mission that collected nothing (`host_mission.py:409-420`).
- **The result is an update, not a model.** It carries `update_form = "delta"`, `weight_mass = M_m`, `n_updates`. `agg:plain` stays on `partial_fedavg` and is only annotated with versions and ages (`:428-441`), so choosing it changes nothing.
- **Identity check.** With every basis current (a_i = 0), v_i = 1 and η = 1: θ_v + Σ (n_i/Σn)(θ_i − θ_v) = Σ (n_i/Σn)·θ_i, the plain mean. `test_every_basis_current_reduces_to_todays_mean` pins it for `agg:cutoff` and `agg:asynchfl`.

---

## 5. The cluster merge (`cross_mule_fedavg.py`, `host_cluster.py`)

```
s_m = s(V − v_m)        live partial: s_m > 0 and not empty
θ ← θ + η · Σ_m M_m · s_m · Δ_m / Σ_{m live} M_m          (apply_weighted_deltas, :83-146)
```

- A single partial gives θ + η·s_m·Δ_m, the FedAsync / Async-HFL mixing form (`aggregation_rules.py:29-32`). The caller passes `normalizer = Σ M_m` over *live* partials (`host_cluster.py:590-593`); `normalizer=None` would divide by Σ W_m and "normalise staleness away".
- **Expired fold.** If no partial is live the cluster takes no step, **keeps the round open** (`cluster_round` unchanged), clears the pending partials so those mules' next UPs are not refused as duplicates, and reports outcome `expired` (`_expire_pending`, `host_cluster.py:615-640`). The service still sends the waiting mules their DOWN (tests `test_service_sends_down_on_expired_but_not_on_quorum`). Raising instead would drop the round as malformed.
- **Gating.** `min_participation` gates every rule except `agg:fedbuff` (its buffer size K is its quorum, `host_cluster.py:459-471`).
- **One mule is exact:** the mule's θ is always the cluster's current θ, so V − v_m = 0 and the cluster level is the identity. The two-level form (device staleness at the mule, partial staleness at the cluster) only has content with several mules. I did not check how often V − v_m > 0 in the K = 3 cells of 5.3 / 5.9 / 5.11 (b); unverified.
- **η = 1** everywhere in Exp 5 (no `--agg-server-lr` is passed). A server rate below 1 is available and untested here.

---

## 6. The rule registry (`aggregation_rules.py`)

`IMPLEMENTED_RULES = (agg:plain, agg:cutoff, agg:asynchfl, agg:fedbuff, agg:fedex)` (`:109`). One flag sets mule and cluster so both run the same rule (`check_partial_form` refuses a mismatch, `:508`).

| Rule | s(a) | Mule weight and normaliser | Cluster step | Source | Cutoff |
|---|---|---|---|---|---|
| `agg:plain` | none | n_i-weighted mean of full models | overwrite with the n-weighted mean over partials | HERMES as recorded | none |
| **`agg:cutoff`** (FeRRy) | FedAsync hinge: 1 for a ≤ b, else 1/(a_h·(a − b) + 1); defaults **a_h = 1, b = 0** (`:152-153`), so s = 1, ½, ⅓, ¼ | w_i = n_i·v_i·s; M_m = Σ n_i·v_i | θ + η·Σ M_m s_m Δ_m / Σ M_m | Xie et al. (hinge); Yang JSAC 2025 (cutoff); reference pending | exact 0 past a_max_j |
| **`agg:asynchfl`** | **(a + 1)^−q, q = 0.5** (polynomial, default since 7 Oct); `asynchfl_form = "exponential"` gives exp(−λ·a), λ = 0.5, this module's form before then | same as cutoff, no cutoff | same, with s at both tiers | Yu et al., IoTDI 2023, who adopt the polynomial from FedAsync | none |
| **`agg:fedbuff`** | 1/√(1 + a) | unweighted by n: w_i = s(a_i), M_m = update count | buffer K updates, then θ + η·(Σ s_m·M_m·Δ_m)/count | Nguyen et al., AISTATS 2022 | none |
| **`agg:fedex`** | none | mule sends the SUM Σ Δθ_i (weights 1, normaliser 1, M_m = count) | θ + η·Σ_m Δ_m / N, N = `fedex_n` or the registered devices | Bian, Shen, Chen, Xu, TMC 24(6), 2025 | none |
| `agg:seq` | — | — | — | after Cui et al. | **out** |

Notes on each, since several are decisions of the last two weeks:

- **`agg:asynchfl` switched to the polynomial on 7 Oct 2026** (commit `aedae8b`). The earlier form, exp(−λ·a), is not Async-HFL's: the paper's "exponential decay factors" (its Table 2) are its mixing weights, not its staleness function. Async-HFL does not report its q; **0.5 is FedAsync's chosen polynomial**, and FedBuff's (1 + τ)^−0.5. A recorded `agg:asynchfl` dict that names `decay` and no form reads back as the exponential it flew (`from_config`, `:262-265`); the exponential and `poly_q` are written to `aggregation_params` only under this rule, so every other rule's rows read as before (`to_params`, `:241-249`). Related Work Notes §6a item 10 records the decision; "no recorded run used it".
- **`agg:seq` is out** (decided 5 Oct 2026, `params.toml [s51]`; Readiness "Decided already"). The code still lists it in `PLANNED_RULES` and refuses it by name: carrying the model device to device needs in-session training, a protocol change (principle 14).
- **`agg:fedbuff` sets K to the opening mule's slice size**, not FedBuff's default of 10 (a declared deviation, Related Work §6a item 11). Exp 5 passes `--agg-buffer-k 6` = N explicitly (`launch.py:930`). The mean is not n-weighted and a buffer still filling at the end of a trial never reaches θ, so FedBuff does not tie with `agg:plain` even with every basis current; it ties only at K = 1 (build plan Phase 1 exit gate; Configuration Reference §14).
- **`agg:fedex` is faithful at η = 1 and `min_participation` = 1.** Port note: a FedEx client's m_i accumulates every local step since its last visit; ours is the update against the basis it trained from, equal when the device trains once between visits (principle 14). It is D4's merge, not a Study 5.1 arm.
- FedBuff and the cluster: `FedBuffBuffer.add` scales each partial by its cluster staleness × its mass (`:535-553`), so FedBuff applies staleness at both levels, like `agg:asynchfl`.

---

## 7. The proximal term on the device (`exp4/model_task.py`)

`DeviceConfig.fedprox_rho` (ρ, default 0.0) adds (ρ/2)·Σ‖w − w₀‖² over the trainable variables, w₀ the model *as received this round* (anchors are re-assigned at the start of each fit), in a custom training loop with the same optimiser, epochs and batch size; ρ = 0 keeps `model.fit` exactly as every recorded run used it (`model_task.py:192-205`, `:245-292`). The loss is the binary cross-entropy plus the model's regularisation losses plus the proximal term. It pulls local training back toward the basis it was handed, which matters most for an update that will be merged late.

Provenance, as the Related Work Notes (§5a, §6a item 9) correct it: FedAsync's own local objective already has this term and FedProx is the usual origin; the build plan cites Shen et al. (IoTJ 2024) as the source used (reference pending). Study 5.1 flies `agg:cutoff+fedprox` at **ρ = 0.01** (decided 5 Oct, "from the middle of the FedProx paper's grid {0.001, 0.01, 0.1, 1}"; 5.1's data are IID, where FedProx stays close to plain averaging). Hand-set.

---

## 8. The age cap S and S\*

**The mechanism** (`stages/s3d_age_cap.py`, decision D4; the user's decision 1 of 30 Sep). A device is capped when its plan age reaches S − L (`AgeCapSpec.caps`, `plan/types.py:311`; L, the lookahead, defaults to 0; S None switches the cap off, arm F-cap). Capping does five things:

- an **exempt** stop (every member capped) skips its own deadline clause (the predicate's `protected`); a **mixed** stop carries the earliest deadline among its uncapped members, so only the capped members' lateness is excused (critic B2); a **priority** stop (any member capped) goes first in a trim (`:215-330`);
- the **cap key** (`cap_key`, `:361`) is the ages of the capped devices a plan leaves out, largest first; the plan key compares it before the served share and before V, so a plan that drops an older capped device loses to any feasible plan that keeps it. (4, 4, 4) beats (5, 3), minimising the oldest unserved age, not the violation count (critic C5);
- a capped device its S3a stop cannot serve alone is offered a one-device **hover stop** (`plan/hover.py`);
- every capped device a mission fails is logged **by cause**: `unplannable`, `crowded` (at plan time, `plan_violations`, `:453`) and `dropped_in_flight`, `not_merged` (at close, `close_violations`, `:490`). There is no pass mark: device availability alone makes about 15 % of device-missions miss at S = 3 (critic A2);
- only the plan path calls this module (Freeze Rule 1), so the recorded pipelines never reach it.

**S\* and how S is set** (`experiments/analysis/age_cap_s_star.py`). For one layout, budget and arm family, S\* is the set-cover number of the devices a mission can serve at all: the fewest Pass-1 missions that together serve every servable device, priced by the S3b predicate at the mean SNR, deadlines left out. The cell's S is the ⌈0.9 n⌉-th smallest S\* over 30 layouts, the largest over the two pilot budgets (knee and stress), **never below 2** because S = 1 caps every device at every mission (critic A1). Exact up to 6 devices, a greedy upper bound above (never more than 2 above exact in U9's probe).

**What it gave.** `results/exp5/sstar/report.txt`: **S = 2 at N = 6, 12, 18 and 24** (budgets 150 s / 75 s, 180 / 90, 240 / 120, 262 / 130). Per the Readiness record: at every N one mission covers 90 % of layouts at the knee and two at the stress budget (N ≥ 12 by the greedy bound). At the knee the 2 is therefore the floor, not a measured need. At 1 MB with the earlier prior budgets of 90 and 45 s the tool printed F S = 2, FB+wide 4, FB+medium 3, FB+narrow 3.

**What S + 1 gives, as measured in the Phase 4 tests** (30 layouts, 1 MB, 3S missions): no `unplannable` at 45, 60 or 90 s in any family and no violation at all for F and FB+narrow there; a pinned class can still crowd a capped device at the stress budget (FB+medium 7 in 291 missions at 45 s, FB+wide twice in 360), and at 30 s every family crowds (F 20 in 309). S + 1 is no guarantee outside 45 to 90 s, for a pinned class, or under whole admission.

---

## 9. Age in the plan score, the features and the reward

- **Coverage weights in V** (`plan_score.coverage_weight`, `:334-366`): mode `age` gives max(a_j, 1)·(1 + m_j) with the arm's `miss_priority` on (F), max(a_j, 1) alone with it off (F-prio); `uniform` gives 1·(1 + m_j) or 1. Every device a plan leaves out is widened as a miss and a clean contact clears the streak, so under a plan m_j = a_j − 1 and **F's weight is about a_j²**: the objective is declared quadratic in age (critic A5). The floor at 1 is in the weight only (a weight of 0 would drop the device from U); the cap reads the raw age. Arm F-pref multiplies each weight by Oort's speed factor (`oort_speed_factors`, α = 2.0, `driver.py _PLAN_ARM`). V and the rank are detailed in [05](05_Objective_and_Reward.md).
- **Learned-score features.** `age_next` (the next stop's mean plan age, divided by S) and `capped_next`, `weight_share` (`pair_features.py:36`, `:48`). The per-stop **value proxy was dropped** from the features (critic C5): under equal data it equalled members ÷ N.
- **Reward.** G_k uses the *merge* weight (n_i·v_i·s(a_i), zero past the cutoff) while U uses the *plan's* coverage weights. See [05](05_Objective_and_Reward.md) §3.
- **The value proxy v_j, as it stands.** The merge has a `loss` mode (raw local loss) that Exp 5 never selects; the plan score has no v_j at all (weights are age × (1 + streak), with no n or value); the learned score dropped it. The build plan's "demand per device: deadline, value, age" is therefore realised on the merge side only, and there as uniform.

---

## 10. Network AoU (`analysis/traces_scorer.py:808`)

After mission m a device's age is m − U_i(m), U_i(m) the last mission up to m that merged its update, 0 if none, so a never-merged device ages from the start of the trial and an empty mission still makes every device one mission older. Network AoU is Σ_i ω_i·age_i(m), ω normalised to 1; uniform weights make it the mean age, reported as its mean over the trial and at the end, with the maximum and 95th-percentile per-device age and Jain's index of merged counts (blank when nothing merged). After Cui et al., TMC 2024 (weighted AoU; ω_i their Shapley data value, here uniform, or shard-weighted in Study 5.13).

"Merged" means the update survived the mule's merge **and reached θ**: an update excluded past its cutoff never does, a lost backhaul upload does not, and a FedBuff-deferred one counts at the mission whose merge flushed it (`merged_devices`, `:781`). With several mules the age counts the device's own mule's missions. This is the same unit as the plan age, which is why the scorer can count a cap violation without a plan (`cap_violations` scores every arm alike). It is the primary metric of 5.4, 5.8 and 5.14.

---

## 11. Layer interfaces

| Edge | What crosses | Where |
|---|---|---|
| **L3 → L2, age** | `last_merged_round` → plan age → cap set, cap key, coverage weights, `age_next` | `record_merged`, `device_age`, `plan_score.demand_weights` |
| **L3 → L2, miss streak** | consecutive misses → (1 + m_j) factor and S3b priority key | `DeviceSchedulerState.miss_streak` |
| **L2 → L3, window** | Φ_j (and S3c scale) → a_max_j, snapshotted after planning | `MuleSupervisor._age_caps` |
| **L2 → L3, who is collected** | the plan's stops and in-flight trims decide which updates exist to merge; Pass 2 composition decides every basis | `mule_main.py:914`, `:3118` |
| **L2 → L3, misses widen windows** | a device left out is widened as a synthetic TIMEOUT, so ages and windows feed the next mission's demand | `_widen_abandoned`, `mule_main.py:3149` |
| **L1 → L3** | contact reliability from the channel: the keyed availability draw drops an uplink (`DiscPush.uplink_drop`); the device still adopts the basis and trains ahead, but no update reaches the merge, so its plan age keeps growing | `fl_messages.py:152`, `client_mission.py:437-449` |
| **L3 → L1** | none directly; L3 sets payload via the model's bytes only through the declared payload | — |

---

## 12. Evidence

| Question | Evidence | Status |
|---|---|---|
| Is the weight arithmetic right? | 440 tests pass in the five age modules (`test_aggregation_rules`, `test_cluster_age_aware_merge`, `test_p4_plan_score`, `test_p4_age_cap`, `test_mule_age_caps`), run 7 Oct at `39b20b84` for this document. They pin the hinge, the polynomial, exact zero past the cutoff, the staleness-free normaliser, the plain-mean identity at age 0, expired folds, FedBuff buffering and FedEx. | Verified here |
| Does the age-aware merge reach τ sooner than a plain mean, and against the async rules? | **Study 5.1.** 5 rules × 2 routes × 2 budgets × Pass 2 {unbudgeted, budgeted on H1}, N = 6, 40 trials per cell, 1 MB, FedProx ρ = 0.01, FedBuff K = 6; metric time to τ = 0.71, scored against `agg:cutoff` in each cell (`params.toml [score.s51]`) | **Not run** (batch 2, ready) |
| The sanity tie | With every basis current, `agg:cutoff` and `agg:asynchfl` (value uniform, η = 1) should equal `agg:plain`; FedBuff only at K = 1 | Pinned at unit level; **not yet seen in a trial**. In the study FedBuff runs at K = 6 and is judged on its own terms; FedProx changes local training and does not tie |
| Do the rules separate once ages spread? | Needs budgeted Pass 2, available on H1's route only | Not run |
| Indirect: FeRRy's merge against FedEx's merge | Same D4 tour, batch 1, time to τ = 0.71, n = 20: **3 mules** D4 55.0 s (cutoff) against D4fedex 101 s; against F (52.3 s) D4fedex is a claim (Holm p 9.8e-4), D4 is not. **1 mule, knee:** 264 against 281 (F 194; D4fedex not a claim). Stress 264 against 281 | Scored; D4 and D4fedex were not compared head to head, and the two merges differ in several ways (n-weighting, staleness, normaliser N, whether partials can be applied together) |
| Does the cap bind? | Study 5.14 at N = 6, Network AoU: cap at S\*, S\* + 1 and off give **identical** results in every column (knee 0.715, stress 0.810; same mean cap violations 4.25 and 4.85; same time to τ) | Scored. At N = 6 the cap changed nothing; consistent with one stop serving all six devices in most missions (Readiness). It says the cap is inert there, not that it is useless |
| Weighted rank versus served-share rank (V alone against the lexicographic default) | 5.14 `wcov` against `capS`: knee 0.717 against 0.715, stress 0.805 against 0.810, neither a claim | Scored; null |
| Whole stops against member subsets under a tight budget | Stress: Network AoU 0.897 against 0.810 (diff −0.0865, CI [−0.149, −0.033], Holm p 0.0775) | Not a claim after Holm |
| Do age-weighted coverage weights beat uniform ones? | F against F-prio (age alone) is in 5.8; `uniform` mode has no study | Not run |

---

## 13. Failure modes, open items, and doc/code discrepancies

**Failure modes and edge cases (all from the code).**

- **Unknown age counts as 0 and is never cut.** A device with no basis version (a device that never received a version, or an old-style push) is merged at full weight (`update_weights`, `:386-392`).
- **A uniformly stale mission shrinks, it does not vanish.** Only a hard cutoff or the loss of every update stops a step; an all-expired fold stalls the round open by design.
- **A late, high-age update from a reliable device is cut hardest** (the a_max arithmetic in §3), and a CLEAN the cutoff excluded does not reset the plan age, so, on my reading of the code, the device can be capped, planned for, collected and excluded again; the `not_merged` cause logs exactly that case (Configuration Reference §18.2).
- **The mule credits merges the scorer may not.** The mule cannot see backhaul loss or cluster-side deferral, so `record_merged` can reset a plan age that the scorer leaves standing (critic B10; design R5).
- **Cluster cutoff applies only to a fixed `a_max`**, not to the per-device D5 cap (§3).
- **FedBuff holds updates outside θ.** A buffer unfilled at trial end never reaches θ; the scorer follows deferrals.

**Open items.**

- [ ] **Bound-derived weights.** The hinge constants (a_h = 1, b = 0) and q = 0.5 are hand-set; FedAsync's own CIFAR setting was a = 10, b = 4 (Related Work §5a). The theory track (substitute Δ(b̄, π) into FedEx Thm 2, eq. 24) is not done, and the weight "derived from the bound" is not a thing that exists. The decision is open (Readiness, "Decisions still open"): drop the derivation and add a ~240-trial sensitivity check on the constants; drop it with nothing added; or do the derivation, after which it becomes a weight mode and a 5.1 variant. 5.7 cites it too. This must be decided before batch 2.
- [ ] **Study 5.1.** Run it; read the sanity tie first.
- [ ] **A merge-age distribution is not scored.** The build plan lists "the age distribution at merge" as reported by 5.1; the trace records `pass_1_merge.ages`, but I found no scorer column that summarises it (`traces_scorer.py`, `scripts/exp5/scoring.py` have none; `score.s51 also` lists accuracy, AUC, closure and Network AoU, which is the *plan-age* unit). Without it, "ages spread" cannot be shown for F's route. Unverified that no other tool reads it.
- [ ] **Yang et al. (JSAC 2025)** and **Shen et al. (IoTJ 2024)** are cited by the build plan and still have no full reference in the repository.
- [ ] A reliable-device cutoff of 0 (§3) is arithmetic from the law's constants and has not been read from a trace.

**Doc and code discrepancies found while writing.**

1. Build plan, "One mission, end to end", step "Build demand": says a device's age a_j is "in cluster rounds". The code, and the plan's own D4 and Phase 4 deviation, count the cap and coverage age in own-mule missions. Stale text.
2. Build plan Study 5.1 (arms list), the aggregation-arms table and the Phase 1 test list still name `agg:seq`; the decision of 5 Oct 2026 and `params.toml` drop it. The code still lists it as planned (refused by name), which is consistent with "out".
3. Build plan: the baseline "every rule must beat" is `agg:plain`; the launcher scores every 5.1 rule **against `agg:cutoff`** in its cell (`[score.s51] reference = "cutoff"`), so the plain-versus-cutoff contrast appears as plain's difference from cutoff.
4. Configuration Reference §14 writes the hinge as "1/(a·(a − b) + 1)", using *a* for both the slope and the age; the code is 1/(hinge_a·(age − hinge_b) + 1) (`:283`).
5. Build plan Phase 1 says the cutoff converts "Deadline(j)"; the code converts the window Φ_j.
6. Joint RL Methods §1.3 (overview) and the Configuration Reference agree with the code on the age definitions; no contradiction with the overview was found. The overview's one-liner "age weights are FedAsync's hinge with hand-set constants" holds for `agg:cutoff` at defaults (a_h = 1, b = 0), not FedAsync's own CIFAR values.

---

## 14. Sources

[FeRRy Build Plan](../FeRRy_Build_Plan.html) (Phase 1, Phase 4, decisions D4 and D5, Study 5.1, metrics) · [Configuration Reference](../HERMES_Configuration_Reference.md) §14, §15, §18.2, §18.7 · [Related Work Notes](../HERMES_Related_Work_Notes.md) §3.1, §5a, §6a · [Experiment 5 Readiness](../Experiment_5_Readiness.md) · [Joint RL Methods overview](../HERMES_Joint_RL_Methods.md) · `hermes/mission/{aggregation_rules,partial_fedavg,host_mission,client_mission}.py` · `hermes/cluster/{cross_mule_fedavg,host_cluster}.py` · `hermes/types/{fl_messages,aggregate,round_report}.py` · `hermes/mule/mule_main.py` · `hermes/scheduler/stages/{s3d_age_cap,s3_deadline}.py` · `hermes/scheduler/plan/{plan_score,types}.py` · `experiments/exp4/model_task.py` · `experiments/analysis/{age_cap_s_star,traces_scorer}.py` · `scripts/exp5/{params.toml,launch.py}` · `results/exp5/{sstar,scores/b1}`.

*Numbers are copied from those records as of 7 Oct 2026. The 440-test run in §12 is the one thing executed for this document.*

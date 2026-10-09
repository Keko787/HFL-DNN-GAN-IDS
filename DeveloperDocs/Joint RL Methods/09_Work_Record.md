# HERMES / FeRRy — work record for the joint-RL line

*7 Oct 2026. A ledger of the work: how it started, what was built phase by phase, what each study has and has not produced, the defects found on the way, the decisions taken, and what is left. It restates git, the repository's records and the committed result files; it decides nothing. Commit hashes are the short hashes `git log` prints and were each checked against the log on 7 Oct 2026. Counts of tests and lines are the records' own and were not re-run unless a row says so. Companion to [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) (what is optimised jointly) and [08_Decision_Register.md](08_Decision_Register.md) (every runtime decision).*

---

## 0. The short version

- **What the line is.** A plan to justify, or refuse to justify, learning in HERMES's mule scheduler by first building the coupling that would make learning necessary: band class and route chosen together at the dock, band and next stop chosen at every arrival, one objective across scheduling and aggregation. The system was built as **FeRRy** (HERMES's Path A) in phases 0 to 5 between 28 Sep and 2 Oct 2026 (the addendum builds run to 3 Oct), and is being run as Experiment 5 (fifteen studies), which is phase 6.
- **What it started from.** The SEC'26 reviews and the code audit that followed them, and the team's own diagnosis that a trajectory layer alone would be too simple a method to justify reinforcement learning.
- **Where it stands (7 Oct 2026, morning).** Phases 0 to 5 are built; phase 6, the campaign, is under way. Pilots, the age-cap tool, the FerrySim re-pin, batch 1 (1,760 trials) and the RL campaign are done and committed. Batch 2 (about 9,220 trials, roughly 50 h) is built and ready; batch 3 waits for its own pilot. The full paper is due **8 Oct 2026 (AoE)**; batch 2 is for the revision window, to **18 Jan 2027**.
- **The result on learning is a null.** A learned in-flight score (FQ) matched the fixed rule FX and fell short of the one-step rule `greedy_1`; looking ahead (γ > 0) added nothing. The pre-registered verdict is "flat", FX stays as FeRRy's in-flight rule, and no learned component is part of the system the paper evaluates (section 5.3).
- **What did earn something.** On the real-process stack at N = 6 and one mule, F beat H1, D1, D2, D3 and D4 on time to τ = 0.71 at the knee budget; at N = 24 it beat H1 and D3. The lead is smaller or absent at the stress budget, with three mules, at N = 12, and against D4 and FX (section 5.1). None of it comes from a learned policy.

---

## 1. How the work started

### 1.1 The SEC'26 reviews and the audit (July 2026)

HERMES, as submitted to SEC 2026 (submission 74), had four reviews (74A to 74D). Their points, as the build plan records them: no recent UAV federated-learning baselines; a scheduler that looks like plain EDF; scale too small; no new algorithm and a deadline function printed with its sign reversed; layers only integration-tested, with the benefit of RL unclear; real versus simulated unclear; little real-world validation and no sensitivity analysis.

An 18-agent audit of every implementation claim against the code was generated on **21 Jul 2026**, inside the rebuttal window of 18 to 22 Jul ([SEC26_Code_Audit.md](../SEC26_Code_Audit.md)). It found the following, in the order the audit ranks them:

| Finding | Why it matters here |
|---|---|
| Table V ran on Chameleon Cloud, not the AERPAW digital twin; Table II and Table VI have no producing code in the repository | Provenance of the paper's results; the later work adds a provenance table |
| The deadline formula was printed with its signs inverted; Φ has a floor and no ceiling; a cluster override is never cleared | Became the multiplicative, clamped, one-shot deadline law (Phase 1) |
| The bucket classifier reads three booleans and no deadline or utility | See the [register](08_Decision_Register.md), row P03 |
| S2A/S2B, the "gated" stages, have no production caller; the live utility gate is a no-op | Freeze decision D3; register row F01 |
| Exp 3's headline scheduling experiment calls only S3a, never `FLScheduler` | Motivated an end-to-end experiment (Exp 4) |
| The selector is a single-agent DDQN trained in a toy simulator, not CTDE on a digital twin; its weight init is unseeded | The learned part of the paper was weaker than described |
| `ChannelDDQN` is inference-only with no trainer; the paper's Eq. 1 controller was missing from the repo | Added as `l1/channel_utility.py` in EX-4.3 |

The rebuttal draft ([SEC26_Rebuttal_Draft.md](../SEC26_Rebuttal_Draft.md)) conceded the platform and wording errors and named the missing end-to-end experiment as the first priority for a revision. **What the SEC'26 decision was is not recorded in the repository.** The draft anticipates a decision on 29 Jul, and later documents treat the work as a revision aimed first at ICNC (Paper Revision Plan v2) and then IPDPS 2027 (the build plan's addendum); the build plan calls SEC 2026 "the last submission".

### 1.2 Experiment 4 and the freeze (23 Jul to 17 Aug)

Experiment 4 is the integrated end-to-end run: real processes, real TCP, real Keras training, a two-pass mission, cross-mule FedAvg ([EX4_Development_Record.md](../Experiment%20documents/EX4_Development_Record.md)). It was built in four stages on 23 to 25 Jul (`efe6b382`, `ba45a9b3`, `5af7da45`, `31426c46`, `c1f0ae9b`, `24315ac7`, `fc71e702`, `38b3e0f6`, `e7492ac4`, `8575aea6`, `225109ff`), with a remediation cycle that dismantled a tilt toward the mule arm. Its headline, from the Phase-3 matrix of 13 Aug (640 trials): H1 beats H0 under a failing backhaul (final AUC +0.054, p < 0.0001) and loses to it when the network is healthy (update yield 1.75 against 3.51). The matrix's honest framing is that HERMES is an availability mechanism that costs throughput when availability is not the problem ([HERMES_Matrix_Results.md](../HERMES_Matrix_Results.md)).

Two things from this period shaped what followed:

1. **The layer-1 effect did not survive in one sweep and was "confirmed" in another.** An earlier H2-versus-H3 dead-zone sweep showed a positive effect of adaptive channel selection (Cliff's δ +0.14 to +0.37 in all five cells). It was contaminated (22 rows with no model, an unpaired effect size); the clean re-run, 200 of 200 rows ok with 20 paired seeds per cell, found all 20 tests ties, and the effect was retracted (`7549c80d`, `ff6d5c5e`, 13 Aug). The Phase-3 matrix of the same day, at a different operating point, reported the L1 effect as confirmed at 40 seeds (`e5eea925`), and on 17 Aug it was restated as an effect on whether a run reaches the target, not on how fast (`29ac8375`). That confirmation was later found to have measured the wrong thing (section 6, defect 3). The records never reconcile the two same-day verdicts (section 9, item 8).
2. **The L2 pipeline was frozen on 13 Aug** (`db60fc7c`) with six decisions, among them D3 (the readiness gates are removed from the claims, not wired) and D4 (**Experiment 4 makes no RL claim; the RL question belongs to a later experiment**). Amendments 1 to 3 followed the same day (`85f6d0d5`, `d0dca4b4`, `7a9a38ce`); a fourth gave the D1 and D2 baselines authority over their own admission on 17 Aug (`2e9fdc9e`).

### 1.3 The diagnosis and the decision memo (26 Aug to 17 Sep)

The team's own methodology review (a document the memo names, *HEREMES Methodology REDEFINED.docx*, which is **not in the repository**; unverified beyond the memo's account of it) judged the problem novel and drones-as-relay-hosts somewhat novel, but the solution method too simple to justify RL, with the open worry that a cross-heuristic would do as well. The proposal on the table was a trajectory-navigation layer plus joint optimisation with RF band selection.

The answer is the decision memo, built as an interactive artifact over 26 Aug to 16 Sep and committed on 17 Sep (`fe1c9ed0`, `605c3aec`; [the memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md)). Its argument:

- A trajectory layer composes into a chain, and a cross-heuristic solves a chain. **A coupling does not decompose**, and that is testable.
- The coupling was severed in code at three cuts: the contact link had no band variable (cut 1); the band was decided once per mission, not per arrival (cut 2); dwell was a 1 s constant and the only upward edge could abort but not re-plan (cut 3).
- The RL test is three properties on two clocks: decomposition, delayed consequence, non-stationarity. The comparison runs on a 2 × 2 of decision scope against objective denomination, so a win can be attributed.
- Computed from the prototype's constants, the band best at takeoff lands in a trough at two of four stops: −0.14 committed against +0.75 best-available at arrival. None of the gap comes from picking the wrong band; all of it comes from committing at all.

The memo's own corrections log (its section 9) has ten entries. One records that team review caught the first draft's ring diagram, which put band choice at the origin and argued for a per-stop decision while drawing a per-mission one; it was redrawn as two clocks.

The memo also fixes in advance what each outcome of its delayed-consequence test (the (b) row) builds. Flat on both clocks: a joint heuristic and a published negative result. Live on the flight clock only: a learned per-arrival policy under a deterministic planner. Live on both: a hierarchy, not to be entered without that evidence. What was measured is flat on the flight clock; the plan-clock cell was not tested.

### 1.4 The prototype lineage

`hermes_rl/` is the original joint-action prototype: one discrete action (waypoint × base station × channel), a hybrid trainer that fixes the waypoint by heuristic and learns (base station, channel). It is a separate repository (initial commit `a8a453f`, 28 Apr 2026, by FyneappleJuice; its own `.git`, untracked in this one). A seeded, golden-pinned copy lives at `experiments/sim/drone_env/` (`c4fc2ddc`, 28 Sep). Its trainer selected its best checkpoint and ran its final evaluation on one fixed episode, so SEC'26 Table VI's 76.40 is a best-of-about-30 on one instance; the vendored copy has no licence from its author (open item).

### 1.5 The build plan (27 to 28 Sep)

[FeRRy_Build_Plan.html](../FeRRy_Build_Plan.html) (dated 27 Sep; first committed with `9cc2c369`, 28 Sep) fixes five contributions: **C1** reach is a decision, **C2** one derived objective, **C3** one deadline in three roles, **C4** two clocks re-decided per stop, **C5** fairness under physical cost. It sets seven phases, one learned object, and a rule that every hard gate stays deterministic. Its inputs include a Novelty Audit Rev. 3 (23 Sep) and a Chen comparison note, which the Related Work Notes say **are not in the repository**, and a deck of 23 Sep, which I did not find either (unverified).

---

## 2. Chronology

| Date (2026) | Event | Hash or source |
|---|---|---|
| 28 Apr | `hermes_rl` prototype's first commit | `a8a453f` (nested repo) |
| 21 Jul | Paper-versus-code audit generated | SEC26_Code_Audit.md |
| 23–25 Jul | Exp 4 stages EX-4.0 to EX-4.3 | `efe6b382` … `225109ff` |
| 13 Aug | Layer-1 effect retracted; L2 pipeline frozen; Amendments 1–3; Phase-3 matrix (640 trials) | `7549c80d`, `db60fc7c`, `85f6d0d5`, `d0dca4b4`, `7a9a38ce`, `e5eea925` |
| 17 Aug | τ set to 0.82; whole-scheduler baselines D1/D2; SOTA pilot; tight-budget result at n = 40 | `29ac8375`, `2e9fdc9e`, `c765ac45`, `fe021b12`, `61f6343e` |
| 26 Aug–16 Sep | RL decision memo built as an artifact (off-repo) | memo header |
| 17 Sep | Memo committed | `fe1c9ed0`, `605c3aec` |
| 28 Sep | Phase 0 (defects, statistics, trace scorer, `drone_env` vendored) and Phase 1 (age-aware aggregation) | `b8f2e3d3`, `3dc06ac6`, `c4fc2ddc`, `25062c93`, `9cc2c369` |
| 29 Sep | Phase 0/1 audit fixes and Phase 2 (baselines D3–D5, multi-mule); Phase 3 (mission clock, contact link, Amendment 10); route-level delivery bound | `edda564d`, `9147211c`, `7c7a269d`, `9cd1c48e`, `80ea8d19`, `6de4cd2b` |
| 30 Sep (docs); commit dated 1 Oct | Phase 4 (plan clock) | `48deb740`, `969cc408` |
| 2 Oct | Phase 5 (flight clock, FerrySim, E3); addendum Studies 5.11–5.15 written | `9c918188`, `12a9aab4`, `f3d8e695`, `e50495b5` |
| 2–3 Oct | Addendum builds: decision cost, scale family, interference settings, F+L1, hover switch, non-IID data, training time, `--model-arch` | `66eb58a7`, `6ba4ae38`, `79531f6c`, `c73b65e9`, `06c2bca2`, `6ec3826a`, `fdc716fb`, `41c4f78f`, `81b36e5e`; merge `524f6e69` |
| 5 Oct | FQ design note; MIT `LICENSE`; the `exp5` launcher; TTL pilot, knee pilot, S\* pilot; U10, U11, O1; Amendment 11; the FerrySim re-pin | `bf3e5792`, `7d52958d`, `fea3fc75`, `a92e15cd`, `0aa4f41b`, `c78759ac`, `c0b710aa`, `1b8da50d`, `be555cf8`, `4d3baadb` |
| 5–6 Oct | Smoke run of every stage; code gate (6,375 tests) | `fd81a60b`, `5636d68a` |
| 6 Oct | Batch 1 and `sens` run (00:36–06:40) and scored; RL headroom (06:42) and calibration (to 09:17); findings written; **option A chosen at 13:58**; Study 5.5 sweep (13:58–23:10); verdict applied | `6f6f2bd6`, `c7846991`, `78070e38`, `663ebb8e`, `cbc1b7a9`, `93eefc72`, `5585a490`, `a7604125` |
| 6–7 Oct | E3's five trainings (23:10–00:10) and checkpoint; batch 2's settings complete | `cef6d5af` |
| 7 Oct | FX-dwell, FX-cov, `agg:asynchfl` polynomial, power-throttling opt-out; Study 5.11 (a) rerun alone | `62640438`, `deb1c9fb`, `c5b1b88f` |
| **8 Oct** | **Full paper due (IPDPS 2027, AoE)** | build plan, addendum |

Authorship of the 81 commits since 17 Sep, from `git log`: 42 under the account name Keko787, 24 Kevin Kostage, 3 Kevin S Kostage and 12 as Claude; 56 carry a `Co-Authored-By: Claude` trailer. IPDPS requires AI-generated text to be declared in the Acknowledgements (build plan).

---

## 3. Phase-by-phase ledger

Sizes and test counts are the build plan's and the Freeze's own figures, "including tests and docstrings".

| Phase | What was built | Commits | Tests added (per the record) | Exit gate |
|---|---|---|---|---|
| **0** Ground truth | Backhaul index and baseline-age fixes; per-mission budget stamp; `holm_bonferroni`, `friedman_test`, `factorial_2x2`, `compare_to_reference`; the trace scorer; `drone_env` vendored | `b8f2e3d3`, `3dc06ac6`, `c4fc2ddc` | 81 new, plus the prototype's 33; suite 944 passing | The plan's gate is a re-run of the L1 cell and the 60 s SOTA cell on fixed code; **the records do not say it was run** (unverified) |
| **1** Age-aware aggregation | Version threading on four message types; delta-form merge with w = n·v·s(age) and the cutoff; cluster fold at rate η; FedProx; budgeted Pass 2; the multiplicative deadline law and priority key; agg:plain/cutoff/fedbuff/asynchfl | `25062c93` | 70 new; agg:plain byte-identical to before | Study 5.1 (not run) |
| **2** Baselines without a channel | 2-OPT; D4 (FedEx with CARP); D3 (Whittle); D5 (FedCS, degraded); `agg:fedex`; multi-mule runtime hardening; Amendments 8 and 9 | `edda564d`, `9147211c` | adversarial review with mutation checks; a 2-mule integration test | The plan's gate is 5.3's channel-free cells at 30 and 60 s, n = 40; not stated as run. Batch 1's 5.3 core flies the same arms on the simulated clock |
| **3** Mission clock and contact link | `l1/mission_clock.py`, `channel_model.py`, `contact_link.py`; one predicate for every walk; re-plan instead of abort; Amendment 10 (RF link) | `7c7a269d`, `9cd1c48e`, `80ea8d19`, `6de4cd2b` | 1,238 new in 36 files; 2,798 tests match the pre-edit baseline (29 Sep) | Re-baseline of H1, D1–D4 on the new clock at 40 paired seeds, after the pilots; batch 1's 5.3 core flies these arms at 20 trials per cell |
| **4** Plan clock | `plan/` (types, score, search, member subsets, hover); `s3d_age_cap.py`; `build_ferry_plan`; flight slot fillings F and FX; arms F, FX, FB+, F-cov, F-cap, F-prio | `48deb740`, `969cc408` | 1,862 new in 15 files; the search equals brute force up to 6 devices | FX end to end at 30 s and 60 s; 5.4 and 5.8 pilots |
| **5** Flight clock | `pair_features`, `pair_q`, `pair_replay`, `pair_slot`; FerrySim (`experiments/ferrysim/`); E3 (`chen_dqn`, `next_stop`); checkpoint format 2 with refusals | `9c918188`, `12a9aab4` | 1,310 new in 14 files; 6,133 tests on 2 Oct, five known failures | **Split**: code gate met; campaign gate = headroom, sweep, verdict, stack trials (now run) |
| **5 addendum** Studies 5.11–5.15 | Planner-time and footprint columns; scale family to N = 96; interference as flags; F+L1; hover switch; non-IID shards and detection metrics; training time and device energy on the clock; `--model-arch` | `66eb58a7` … `524f6e69` | in the merge (55 files, +6,459 lines) | Part of the campaign |
| **6** Campaign | `scripts/exp5/{launch,scoring,repin,fit_time_probe}.py` and `params.toml`; the pilots; batch 1; the RL stages; scoring | `7d52958d` … `c5b1b88f` | code gate 6,375 (6 Oct) and 6,396 (7 Oct, `62640438`) | Batches 2 and 3 pending |

Checked by me on 7 Oct: `pytest --collect-only` on this checkout collects 6,387 tests and fails to import two files (`test_exp4_analysis.py`, `test_exp4_no_eval_guard.py`, a `numpy.dtype size changed` binary mismatch), which fits the record's 6,396 but does not prove it. The suite was not run.

**Build-plan items that landed elsewhere or differently.** The plan's `experiments/exp5/` and `experiments/analysis/exp5.py` were built as `scripts/exp5/` (launcher and scorer); tables, figures and a provenance table are not built. `oracle/zhai_oracle.py` (O1) landed as `experiments/analysis/o1_oracle.py` (`1b8da50d`, 5 Oct), although the build plan's Phase 5 rows still call it deferred. `monolithic_ho.py` (M1) is not built, by design. FerrySim lives in `experiments/ferrysim/`, not `selector/ferry_sim.py`, because `hermes/` may not import `experiments/`.

### 3.1 Scheduler Freeze amendments

| # | Date | One line | Commit or section |
|---|---|---|---|
| 1 | 13 Aug | In-flight abort when the next stop cannot be reached, and a synthetic TIMEOUT for devices the gate skipped | `85f6d0d5`; Freeze §5a |
| 2 | 13 Aug | S3c, mission-level widening of every window; off by default | `d0dca4b4`; §5b |
| 3 | 13 Aug | Raw loss and example counts carried on the device-to-mule path for the Oort baseline | `7a9a38ce`; §5c |
| 4 | 17 Aug | Whole-scheduler baselines D1/D2 own admission and order (the ordering-only arms were vacuous) | `2e9fdc9e`; §5d |
| 5 | 27 Sep | Backhaul schedule indexed by mission; D1/D2 age from the last CLEAN | `3dc06ac6`; §5e |
| 6 | 28 Sep | Every mission's budget runs from its own start | `3dc06ac6`; §5f |
| 7 | 28 Sep | The FeRRy build opens the frozen surface behind switches; principles 1, 5 and 12 restated | `9cc2c369`; §5g |
| 8 | 28 Sep | Baselines are budget-checked in flight, not held to our per-device deadline; a plan's diagnostics are its own | `edda564d`; §5h |
| 9 | 28 Sep | A mule failure fails the trial; bootstrap and reconnects survive | `edda564d`; §5i |
| 10 | 29 Sep | A silent device keeps its RF link (finding P-02) | `7c7a269d`; §5j |
| 11 | 5 Oct | A trial that fails before its shutdown shuts its processes down | `be555cf8`; §5m |

Phases 4 and 5 and unit U11 needed no amendment (§5k, §5l, §5n): every change sits behind a switch whose default keeps the pinned pipeline. The code behind every recorded Exp 4 result is the tag `exp4-recorded`.

---

## 4. Studies ledger (5.1 to 5.15)

The launcher stages are in [the reproducibility guide](../Experiment_5_Reproducibility_Guide.md). "Scored" means paired comparisons by bootstrap CI, Wilcoxon and Holm per study at α = 0.05.

| Study | Question | Batch | Status on 7 Oct | Headline where scored |
|---|---|---|---|---|
| **5.1** Aggregation with age | Does age-weighted merging with the deadline as cutoff beat plain and async rules? | 2 | **Ready**; one open decision (bound-derived weights); `agg:seq` decided out | none |
| **5.2** Deadline form | Does one per-device deadline beat FedCS's and Oort's? | 2 | **Ready** (F-round, F-pref built 5 Oct, `c0b710aa`) | none |
| **5.3** Whole-scheduler comparison | Where does F stand against D1–D5, H1, E3? | core in 1; rest in 2 | **Core done and scored** (640 trials); D5, H0, E3, relaxed budget ready | section 5.1 |
| **5.4** Reach | Is choosing b̄ at plan time worth it? | 2 | **Ready**; O1 oracle and sweep knobs built 5 Oct | none |
| **5.5** γ sweep | Does the in-flight choice need a horizon? | sweep done; stack check in 2 | **Verdict: flat** (6 Oct) | section 5.3 |
| **5.6** When learning helps | Does the learned score track stop spacing over the interference period? | 2 | **Ready** as E3 against FX; M1 not built | none |
| **5.7** One objective | Do the dwell and coverage terms matter? | 2 | **Ready**: FX, FX-dwell, FX-cov, D4 (built 7 Oct); learned-reward grid dropped with the learned score | none |
| **5.8** Fairness under cost | Is F fair when reaching costs time and dwell? | 2 | **Ready** | none |
| **5.9** Scale and robustness | Do results hold at 12–24 devices, 1–3 mules? | core in 1; rest in 2 | **Core done and scored** (300 trials) | section 5.1 |
| **5.10** Real radios | AERPAW validation | — | **Blocked**: needs testbed access | none |
| **5.11** Decision cost and scaling | What do F's own decisions cost as N grows; does it scale with mules? | (a), (b) in 1; (c) in 3 | (a) done twice, (b) done and scored; (c) waits for its pilot | section 5.1, 5.2 |
| **5.12** Compute heterogeneity | Does F hold up with stragglers and larger models? | 3 | **Blocked** on `pilot3` (p512 sets the training-time levels); built 3 Oct | none |
| **5.13** Data heterogeneity | Do age and coverage matter more under non-IID data? | 2 | **Ready**; built 3 Oct (`6ec3826a`) | none |
| **5.14** Component ablations | What does each mechanism contribute? | 1 | **Done and scored** (720 trials) | section 5.1 |
| **5.15** Radio layer | How sensitive is F to the channel; does an adaptive backhaul pay? | 3 | **Blocked** on `pilot3` (harsher amplitude, lossier exponent) | none |

**Pilots and stages.** TTL (session timeouts 36, 34, 34, 23 s at N = 6, 12, 18, 24), knee (150, 180, 240, 262 s; stress half; τ = 0.71; 600 trials), S\* (2 at every N), the FerrySim re-pin: all done on 5 Oct. `quick` (a reviewer's reproduction of 5.3's headline cell) and `pilot3` are ready. Batch 2 is 350 jobs and 9,220 trials (660 shared with batch 1), roughly 50 h; batch 3 is about 1,460 trials, roughly 3 h ([Readiness](../Experiment_5_Readiness.md)).

---

## 5. Results to date

### 5.1 The stack (batch 1 and `sens`, 6 Oct)

Time to τ = 0.71, simulated seconds, lower is better, 20 paired trials per cell; Holm family = the study. Source: `results/exp5/scores/b1/{s53,s59,s511b,s514}.md`, commit `c7846991`.

| Cell | F | FX | H1 | D1 | D2 | D3 | D4 | Which differences are claims |
|---|---|---|---|---|---|---|---|---|
| N = 6, 1 mule, knee | 194 | 178 | 282 | 282 | 266 | 280 | 264 | F beats H1, D1, D2, D3, D4 (Holm p ≤ 0.008). **FX is 16 s ahead of F; not a claim (p = 0.146)** |
| N = 6, 1 mule, stress | 178 | 175 | 220 | 238 | 229 | 209 | 264 | Only D4 |
| N = 6, 3 mules, knee | 52.3 | 56.4 | 64.6 | 64.6 | 64.6 | 64.6 | 55.0 | None except D4 with its own merge (101) |
| N = 12, 1 mule, knee | 486 | 424 | 589 | — | — | 507 | 459 | None |
| N = 24, 1 mule, knee | 749 | 789 | 951 | — | — | 911 | 752 | F beats H1 and D3; not D4 or FX |
| N = 12, 3 mules (5.11 b) | 151 | **102** | 165 | — | — | — | 128 | **FX beats F** (p = 0.0065) |

What this supports and what it does not:

- F's lead over the published schedulers is real at N = 6, one mule, the knee budget, and at N = 24 against H1 and D3. It is not a general lead: it vanishes at the stress budget, with three mules, at N = 12, and against D4 at N = 24.
- **F and FX are not separated, but FX's mean is lower in most cells.** FX's mean time to τ is below F's in every 5.11 (b) cell and at N = 6 and N = 12 in 5.9, and above it at N = 24. One of those differences is a claim: FX over F at N = 12 with three mules. At the N = 6 knee FX leads by 16 s (not a claim). The `sens` stage reports FX against F as a claim (Holm p = 0.0231) only because its Holm family is 6 comparisons (3 session-timeout cells × 2 arms), not 5.3's 28: **the claim depends on the multiplicity family.**
- D1 and H1 have identical means at the N = 6 knee (282, median 285); I did not trace why (unverified).
- `sens` flew the same cell at session timeouts × 0.75, × 1 and × 1.5 and every scored number is identical across the three. The timeout does not move any outcome on the simulated clock at these cells.

**5.14, component ablations (network AoU, N = 6, 40 trials, 720 in all).** At the knee, whole stops, forced local search, S\* + 1, cap off and hover off are all identical to F with cap S\* (0.715), and the weighted coverage rank differs by 0.002. At the stress budget whole stops differs most (0.897 against 0.810, not a claim; Holm p = 0.078) and the weighted rank by 0.005 (0.805). **Of the switches only F+L1 against F on the seconds backhaul is a claim**: 0.689 against 0.883 at the knee and 0.786 against 0.970 at stress (p = 1.7e-4 both). At N = 6, most mechanisms are inert; they are reported as inert, not dropped.

**5.11 (a), decision cost.** The batch-1 run was side by side with all other jobs and possibly throttled; the rerun alone on 7 Oct (`c5b1b88f`) made the same plans episode for episode and ran 1.5 to 2.5 times faster. Mean plan time in auto mode: N = 6 0.037 s, N = 24 0.142 s, N = 48 0.409 s, N = 96 1.531 s (p95 1.996 s). Report from the rerun.

### 5.2 The learned score's calibration (6 Oct)

Source: [the calibration findings](../Experiment_5_RL_Calibration_Findings.md), `results/exp5/rl/{headroom,calibration}/`, commit `78070e38` (the stages ran from `cb292ae5` and `c7846991`, per their launcher manifests).

| | Result |
|---|---|
| Headroom, 200 episodes per cell, six cells | the room over FX is 0.012–0.042, and 0.005–0.019 for the in-flight slot; ε = 0.01 in every cell; the N = 6 cells barely decide (2–13 % of sorties hold more than one decision); `greedy_1` beats FX in all six |
| Calibration, `jittery56` (γ ∈ {0, 0.9} × 3 seeds) | flat; γ = 0 passed the sanity check |
| Calibration, `clean` (negative control) | **sanity-failed by 0.0005** (2 of 3 seeds below the floor) |
| Training curves | no run climbs; validation bounces by about ε; TD loss hardly moves |
| Diagnosis | FQ at γ = 0 sees what `greedy_1` sees yet lands on FX, not on `greedy_1`: it probably **copied FX**. Held as a hypothesis |

The findings document proposes a one-revision screen on the clean cells (more exploration, a lower learning rate, steadier selection). **It was not run.** Option A, the sweep as pre-registered, was chosen at 13:58 on 6 Oct.

### 5.3 Study 5.5's sweep and verdict (6 Oct, 23:10): flat, FX stays

Source: `results/exp5/rl/s55/verdict.json`, commit `93eefc72`. Sixty trainings (γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99} × 10 seeds, family `jittery56`, default learner), 1,000 held-out episodes per cell. The headline means below are over the two N = 12 cells (`jit-n12-90`, `jit-n12-180`); the N = 6 cells are flown as the control and appear in the file's `per_cell` block.

| γ | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| Mean held-out return | −0.0795 | −0.0776 | −0.0779 | −0.0774 | −0.0778 | −0.0779 |

- **Rising? No.** No γ beats γ = 0 by ε = 0.01 (gains 0.0015–0.0020; the best, γ = 0.75, Holm p = 0.14).
- **Flat? Yes.** Every γ > 0 is equivalent to γ = 0 within ±ε (TOST p ≤ 1.3e-4).
- **Sanity check: passed**, though 3 of the 10 γ = 0 seeds fall below the floor.
- **Against the fixed rules:** the best learned score (−0.0774) sits at FX's level (−0.0786) and trails `greedy_1` (−0.0716) by 0.0059 (CI [−0.0066, −0.0050]). `replace_fx = false`, `keep_learned = false`. `greedy_1` against FX is +0.0070, below ε, so the "user decides" flag was not raised.
- **Consequences applied in `params.toml`:** `gamma_star = 0.75` recorded only; FQ arms leave batch 2 except the stack check (FQ-g75, FQ-g0); 5.7's learned scores do not train; M1 is not built; FX-dwell and FX-cov were built so 5.7 can still test the plan terms.
- **E3 (Chen's recipe, γ = 0.99, five seeds):** held-out bytes returns 1.933–1.941, within 0.4 %; seed 0 (the median-validation seed) is the checkpoint batch 2 flies.

### 5.4 The null result, stated plainly

1. **The learning claim of C4 is a null.** The pre-registered test found no benefit of a learned in-flight score over the fixed rule FX, and none of looking ahead. The paper can say the learned score matched FX. It cannot say no learner could do better.
2. **The null may be a weak test.** The learner probably copied FX (the diagnosis above). Its training is mostly FX-guided: 500 episodes around FX's pair, then ε falling from 0.3 to 0.05. The headroom over FX is thin (at most 0.042 per episode) and the N = 6 cells give the slot almost nothing to decide. The screen that would separate "a ceiling" from "a learner that stopped learning" is written down and unrun.
3. **What stays standing is the structure, with a fixed rule.** A plan at the dock and a re-decision on measured signal at each stop by FX. The evidence for it is the batch-1 table above: real at N = 6 and N = 24 against H1 and D3, absent elsewhere, and F-versus-FX is mixed.
4. **The plan clock was never tested for a slope.** The build plan reads 5.5 as: a flat curve means FX is FeRRy's filling and the null is published, and a learned plan-time value, with a hierarchy, is considered only if the plan clock shows a slope too. The plan-time delayed-consequence test (γ across missions) was not run, so the plan clock's cell stays empty and no hierarchy is entered.
5. **Not tested at all:** Study 5.4 (is picking b̄ worth it) and its oracle, 5.6, 5.7, 5.10, and every batch 2 and 3 study. The paper's C1 and C2 claims rest on studies that have not run.
6. **Older results that should not be cited unqualified.** The Phase-3 matrix and the SOTA comparison (13 and 17 Aug) predate Amendments 5 and 6; the Freeze says every affected cell is re-run before it is cited, and batch 1 is that re-baseline on the new clock. Neither results document carries the caveat (section 9).

---

## 6. Defects found and fixed on the way

| # | Defect | Found / fixed | Effect on recorded results | Source |
|---|---|---|---|---|
| 1 | A contaminated dead-zone sweep (22 rows with no model; unpaired effect size) showed a layer-1 effect that a clean re-run did not | 13 Aug, `7549c80d` | A reported positive effect retracted; no end-to-end benefit claimed for L1 | Rebuttal draft, "LAYER-1 RUN" |
| 2 | L2 defects: the deadline was a sort key only (nothing compared it to a clock), and the selector's init was unseeded | 13 Aug, `e14ad268` | S3b added as a hard gate, opt-in | `s3b_feasibility.py` header; Freeze |
| 3 | **Backhaul loss schedule read only at mission 1** (`UpBundle` has no `mission_round`) | found 27 Sep, fixed `3dc06ac6` | Every `--l1-channel` cell compared mission-1 bands held for the whole trial. Round closure at k = 1 re-scores from 0.831 to 0.675 (H2) and from 0.838 to 0.813 (H3). The L1 confirmation "measured that difference, not per-mission adaptation" | Freeze Amendment 5 |
| 4 | **Any non-CLEAN outcome reset a D1/D2 device's age** | found 27 Sep, corrected 28 Sep (first said "rarely") | In the 60 s traces only 34 of 608 scheduled D1 devices came back CLEAN, so D1's route and admission change in most missions of every realism cell | Amendment 5 |
| 5 | **Stale budget stamp**: stamped only on a DOWN, so an empty mission planned against the old stamp | `3dc06ac6`, Amendment 6 | 72–81 % of missions were empty at 60 s; in one D1 trial the budget at planning fell from 60 s to about 39 s by mission 4. Every budgeted recorded cell is affected | Amendment 6 |
| 6 | The in-flight re-check held D1/D2 to our per-device deadline | 28 Sep, `edda564d` | A baseline's first choice could be refused by our gate | Amendment 8 |
| 7 | RF sockets dropped a device silent for 30 s and its loop then spun (353,202 calls in 0.2 s) | 29 Sep, `7c7a269d` | None expected on recorded runs (longest silence in kept traces 23.3 s) | Amendment 10 |
| 8 | **`drone_env` evaluates on its training instance**: seeds ignored by default, best of about 30 evaluations on one episode | found 28 Sep, **left as found** | SEC'26 Table VI's 76.40 is a best-of-about-30 score on one instance | `experiments/sim/drone_env/README.md` |
| 9 | S2A/S2B never run; the live utility gate is a no-op | audit 21 Jul; Freeze D3 13 Aug | Documented, not fixed | Audit §C |
| 10 | V alone let the empty plan beat a device that fits | 30 Sep (R11) | The plan key ranks served share before V | Build plan, Phase 4 |
| 11 | A capped far device could be starved for good (PLAN-1, E2E2-01) | 30 Sep, the user's decision | Hover stops added | Build plan, Phase 4 |
| 12 | Phase 5 final check: five confirmed findings (5.6's cells not built; FQ-dwell and FQ-cov untrainable on their own plans; no tie between a checkpoint and its tag; device-serve columns are harness artifacts; one false rationale) | 2 Oct | Each fixed or recorded (Freeze §5l, R21–R25) | Build plan, Phase 5 |
| 13 | A driver leak: a trial failing at startup left processes running | R14; accepted 5 Oct, `be555cf8` | None changed a result | Amendment 11 |
| 14 | **FerrySim's cells flew placeholder budgets** (N = 6: 45 / 90 s; N = 12: 120 / 180 s) | 5 Oct, `4d3baadb` | Re-pinned to the stack's measured (75, 150) and (90, 180) s; two tests that held a budget as a literal were fixed. At the measured N = 6 budgets the control mostly gives the slot the dock alone | Re-pin commit message |
| 15 | **p512's not-ready band was out of reach**: every device's first fit starts at the first takeoff, so mission 1 adds about 20 points | 6 Oct, `e27f475d` | The band is read after each mule's first mission | Commit message |
| 16 | Smoke trainings stopped short of the learner's warm-up | 5 Oct, `fd81a60b` | Smoke-only | Readiness |
| 17 | **Windows power throttling** held jobs on efficiency cores (about 28 % CPU on the second host) | 6 Oct; opt-out `62640438`; rerun `c5b1b88f` | The first hour of the sweep's trainings ran throttled (outcomes seeded and simulated, so unaffected); wall-clock measurements taken while throttled are suspect, and batch 1 may have been throttled (Readiness); 5.11 (a) rerun | Commit messages |
| 18 | `agg:asynchfl` used an exponential, which Async-HFL does not | 7 Oct, `62640438` | No recorded run used it; switched to the polynomial (a + 1)^−q | Commit message |
| 19 | The Matrix Results header said 560 trials; the runs are 640 (A1 40 + A2 480 + C1 80 + C2 40) | uncommitted edit in the working tree | Count corrected | `git diff` |

---

## 7. Decisions taken

The records name "the user" for decisions and "the orchestrator" for the resolutions R1 to R29 of the build; I attribute no further than that.

| When | Decision | Recorded in |
|---|---|---|
| 13 Aug | Freeze D1–D6: S3b's mechanism frozen; S2A/S2B removed from the claims; Exp 4 makes no RL claim | Freeze §2 |
| 17 Aug | τ = 0.82; reach-rate as the lead metric; whole-scheduler baselines own admission | `29ac8375`, `3337f5a1`, `2e9fdc9e` |
| 29 Sep | Band classes by bandwidth on one carrier (D1); seconds-axis channel and the re-run bill (D2); payload and energy (D3); keep both readings of `deadline_bounds = delivery`; the pilot plan (`replan` with `trim`); sign-off of a six-failure baseline | Build plan, Phase 3 |
| 30 Sep | D4: S counts the device's own mule's missions, from the S\* tool, never below 2. Δ is the whole mission on b̄, κ = 1. Coverage weight is age × (1 + miss streak). Member subsets for the F family (not the recommendation). FX built now. D arms' drops reported only. **Hover stops**, after the final check | Build plan, Phase 4 |
| 1 Oct | Phase 5: the score chooses within the plan, FX's pair when nothing fits; FerrySim is the real system in one process; one score per channel regime; the reward is the merge weight; Study 5.5's rule fixed in advance; E3 is a numpy port; **H2 and H3 leave Exp 5**; checkpoints committed only with consent and after a LICENSE | Build plan, Phase 5 |
| 2 Oct | The addendum and its split: pilots and cores before 8 Oct, the rest in the revision window | Build plan, addendum |
| 5 Oct | The whole campaign flies the jittery contact channel; τ from the knee pilot; F-family law multiplicative; merge period T = T_nom; N = 6's knee accepted at the grid's edge; second host; E3 at γ = 0.99 and Chen's settings; FedProx ρ = 0.01; 5.6 at 40 trials; `agg:seq` out; Amendment 11; U11 approved (option A) | `params.toml`; Freeze §5m, §5n |
| 6 Oct | p512's band read after the first mission; **option A**, the 5.5 sweep as pre-registered (13:58); the verdict applied by rule | `e27f475d`; findings; `5585a490` |
| 7 Oct | FX-dwell and FX-cov built; `agg:asynchfl` polynomial; 5.11 (a) reported from the rerun | `62640438`, `c5b1b88f` |

---

## 8. What remains

**Dates stated in the build plan (IPDPS 2027):** full paper **8 Oct 2026**; early rejects 30 Nov; rebuttal 3 Dec; first-round decisions 18 Dec; revised submissions **18 Jan 2027**; final decisions 2 Feb; camera-ready 20 Feb. Ten double-column pages; double-anonymous, so the paper's artifact is to be a separate repository made once the experiments are complete; a reproducibility appendix is required on acceptance; AI-generated text is declared in the Acknowledgements.

| Item | State | Needs |
|---|---|---|
| Batch 2 | built and ready: 5.1, 5.2, the rest of 5.3 (D5, H0, E3, relaxed budget), 5.4 with O1, 5.5's stack check, 5.6, 5.7, 5.8, the rest of 5.9, 5.13 | about 50 h of machine time; the 5.1 weights decision |
| `pilot3`, then batch 3 | `pilot3` ready (80 trials plus FerrySim); batch 3 (5.12, 5.15, 5.11 (c)) blocked on it | `report pilot3 --apply`, then hand-set values (5.15's harsher amplitude and exponent, 5.11 (c)'s knees) |
| `quick` | ready after batch 1 | a reviewer's reproduction of 5.3's headline cell |
| Scoring and archives | batch 1 and `sens` scored; trace archives committed as checksums only | scoring of the later batches; publishing the archives needs the user's go-ahead |
| Option B screen | fixed in the findings, unrun | only if reviewers press on "the learner copied FX"; it must run on the clean cells |
| Plan-time (b) test | untested | γ swept across missions; the memo (§5) enters a hierarchy only on live evidence on both clocks |
| 5.1's bound-derived merge weights | open | a derivation, a sensitivity check (about 240 trials), or drop |
| Study 5.10 | blocked | AERPAW access |
| `drone_env` licence | open | ask the author or remove the vendored copy |
| Citations | partly done | Chen 2023 is in the Related Work Notes (full text read 6 Oct); Bayerlein 2021 is not there as its own entry; the Novelty Audit Rev. 3 and the Chen comparison note are absent from the repository |
| The host | open risk | session timeouts and knees were measured on the first host; the campaign now runs on the second |

---

## 9. Record discrepancies found while writing

These do not change a result; each is a place where two records disagree or one is stale. Nothing was edited to correct them.

1. **Matrix Results and SOTA Results carry no Amendment 5/6 caveat.** The Freeze says every `--l1-channel` and D1/D2 cell is re-run before it is cited, yet both documents present their L1 and D1/D2 results unqualified. SOTA Results is headed "Run 2026-08-13" while its commits are dated 17 Aug.
2. **Reproducibility Guide §11 is stale.** It is headed "As of 6 Oct 2026" and lists "Next: batch 1, and the RL campaign" and "p512's band, to decide", though batch 1 ran 00:36–06:40 on 6 Oct and the p512 band was decided the same day.
3. **Batch sizes disagree.** The guide's stage table gives batch 2 as "about 9,100–10,100" trials and 35 h, and batch 3 as 1,060 trials and 1.5 h; Readiness gives 9,220 trials and roughly 50 h, and about 1,460 trials and roughly 3 h.
4. **The calibration findings describe the family means as taken over "the four jittery cells".** The verdict files list two cells (`jit-n12-90`, `jit-n12-180`, plus `cln-n12-90`, `cln-n12-180` for the clean family), and FX's −0.0786 reproduces as the mean of those two cells' `per_cell` values; the N = 6 cells are in the file but outside the headline mean.
5. **The build plan's Phase 5 rows call O1 deferred** while `1b8da50d` built it on 5 Oct; its Study 5.5 and 5.6 rows still say "needs the training campaign".
6. **Phase 4's date.** Docs date it 30 Sep (Freeze §5k, build plan); the commits are dated 1 Oct.
7. **An open item already done.** HERMES_Joint_RL_Methods lists Chen et al. 2023 as still to add to the Related Work Notes; it was added on 6 Oct (`5b9a7aec`). Bayerlein 2021 is still absent.
8. **Two same-day verdicts on layer 1.** The retraction of 13 Aug (no end-to-end L1 effect) and Matrix Results' "L1 CONFIRMED" (also 13 Aug, n = 40, AUC +0.046) are at different operating points and are not reconciled in either document; Amendment 5 then says the matrix's L1 cells are re-run before they are cited.

---

## 10. Sources

Git: `git log` of the main branch to `c5b1b88f` (1,215 commits; 81 since 17 Sep). Records: [Experiment_5_Readiness.md](../Experiment_5_Readiness.md), [Experiment_5_RL_Calibration_Findings.md](../Experiment_5_RL_Calibration_Findings.md), [Experiment_5_Reproducibility_Guide.md](../Experiment_5_Reproducibility_Guide.md), [FeRRy_Build_Plan.html](../FeRRy_Build_Plan.html), [HERMES_Scheduler_Freeze.md](../HERMES_Scheduler_Freeze.md), [HERMES_Matrix_Results.md](../HERMES_Matrix_Results.md), [SEC26_Code_Audit.md](../SEC26_Code_Audit.md), [SEC26_Rebuttal_Draft.md](../SEC26_Rebuttal_Draft.md), [EX4_Development_Record.md](../Experiment%20documents/EX4_Development_Record.md), [HERMES_Paper_Revision_Plan.md](../Paper%20Revision/HERMES_Paper_Revision_Plan.md), [the decision memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md). Result files: `results/exp5/scores/{b1,sens}/`, `results/exp5/rl/{headroom,calibration,s55,e3}/`, `scripts/exp5/params.toml`.

*Numbers are copied from those records and files as of 7 Oct 2026; none of the experiments was re-run for this document. Statements marked unverified rest on a record alone.*

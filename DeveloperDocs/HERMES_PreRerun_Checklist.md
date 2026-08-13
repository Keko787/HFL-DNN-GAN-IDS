# Pre-re-run checklist — settle these before spending compute

**Purpose.** Every sweep costs ~30–60 min of the box, and each unsettled decision that lands
*after* a sweep invalidates it. This document is a **gate**: we re-run when everything below is
either done or explicitly deferred with a reason. It exists so we pay the compute once.

**Status:** open. Nothing is scheduled to re-run until §5's exit criteria are met.

**Progress:** Phase 0 — 8 of 11 closed (3 open, all no-re-run). Phase 1 — **done, scheduler
frozen**, amended twice since (§1a). Phase 2 — **done, full-text verified**; baselines chosen, and
the reading *reversed* the first pass's recommendation (§4). Phase 3 — **§5.0 pilot run, result in
§5.0a**; the matrix itself (§5.1) is **the remaining blocker**. Phase 4 — not started.

**Exit criteria status: 3 of 6 met.** ✅ 2 (frozen) · ✅ 3 (baselines chosen — nothing is
retroactively scorable, so both are re-run arms) · ✅ 4 (pilot recorded). ❌ 1 (three Phase-0 items
need an explicit *deferred-with-reason* decision) · ❌ 5 (**the matrix is not written or costed**)
· ❌ 6 (follows from 5).

**Next action — the two decisions that unblock everything, in §5.1a.** Neither costs compute; both
are design calls that the matrix cannot be written without:

1. **Is deadline enforcement on in the headline?** The single biggest re-run driver (−0.225 mission
   completion, measured). Freeze D1 deliberately left this to the matrix so it is costed once.
2. **Do the two SOTA baselines ship as arms?** They are **re-run items** (§4) and, unlike every
   other arm, they **do not exist in code yet** — that is implementation work which must be scoped
   *before* the matrix is costed, not discovered during it.

> Trace retention is **done** (§4) — pass `--keep-event-traces` on every run from here, so nothing
> expensive is paid for twice.

---

## 1. The re-run ledger — what actually forces a re-run

The single most useful distinction here. Most open items do **not** need new trials.

| Change | Forces a re-run? | Why |
|---|---|---|
| Turn on deadline enforcement (`--mission-budget-s`) | **Yes** | Changes participation by ~29 % (measured) |
| Turn on in-flight abort (Amendment 1, A1) | **Only with** `--mission-budget-s` | Rides the same toggle; inert without a budget |
| Turn on abandoned-device widening (Amendment 1, A2) | **Only with** `--mission-budget-s` | Same toggle; changes Φ trajectories once the gate drops anyone |
| Turn on window adaptation (`--mission-window-adaptation`, Amendment 2) | **Yes** if enabled | Changes every deadline once the mule falls below target |
| Trained selector weights instead of random-init | **Yes** (H2/H3 only) | Changes the arm being measured |
| Selector seeding fix | **Yes** (H2/H3 only) | Weights differ from the committed rows |
| Apply `dead_zone` to the mule arms | **Yes** (H2/H3 only) | Would make the L1 sweep a real severity sweep |
| Add a SOTA baseline arm | **Yes** (new arm) | No existing data for it |
| Score **any** baseline retroactively | **Not possible** | The per-contact trace is deleted at trial teardown (§4), so there is nothing to replay. Both chosen baselines are re-run items |
| Retain event traces (`--keep-event-traces`) | **No** — but do it *with* the matrix | Changes no trial behaviour; it only stops the run-dir JSONL being deleted. ~9.7 KB/trial. Skipping it means the next baseline request costs another full re-run |
| Change N, mule count, or `n_missions` | **Yes** | Different operating point |
| Wire S2A/S2B readiness gating | **Yes** if it changes admission | Currently inert |
| — | — | — |
| `ε_prop` calibration | **No** | Energy is a post-hoc function of recorded columns |
| Manuscript deadline-sign correction | **No** | Prose/algorithm text only |
| Figure regeneration, new plots | **No** | Re-analysis of committed CSVs |
| Additional statistics on existing metrics | **No** | Re-analysis |
| Re-framing claims (severity-sweep wording, L1 retraction) | **No** | Already done via re-analysis |

> **Rule of thumb:** if it changes what the *trial does*, it forces a re-run; if it changes what we
> *say about the trial*, it does not. Prefer the latter wherever it is honest to do so.

## 1a. Scheduler amendments since the freeze — and why nothing recorded was lost

Three mechanisms have been added to the **frozen** L2 surface since 2026-08-13. All were taken
*before* the Phase-3 matrix ran, which is the only reason they were cheap: batching them in costs
nothing now, whereas adding them after the matrix would have cost a second full re-run.

| # | Mechanism | Toggle | Default |
|---|---|---|---|
| **A1** | **In-flight abort.** S3b was pre-flight only; the mule kept flying stops it could no longer serve, burning budget and delaying delivery of updates already aboard. It now re-checks feasibility from its **current** pose and clock before each stop and turns for home when the remainder is unreachable. | `--mission-budget-s` | off |
| **A2** | **Abandoned-device widening.** `RoundCloseDelta` is emitted only from inside a contact session, so devices S3b dropped — or an abort abandoned — never widened Φ and were dropped again next mission. **A starvation loop created by the S3b gate itself.** They now receive a `TIMEOUT` delta, exactly as a missed contact does. | `--mission-budget-s` | off |
| **A3** | **Mission-level window adaptation (S3c).** The per-device rule only ever learns "this device was missed"; it is blind to "the mule is systematically not completing its circuit". S3c tracks served/planned over a rolling window of missions and widens **every** window together while the mule is below target. Bounded, and it only ever widens — shrinking stays the per-device rule's job. | `--mission-window-adaptation` | off |

**Why the recorded sweeps still stand.** Every one of these is **inert at its default**, and that
inertness is pinned by test rather than asserted: with no budget the abort never fires and nothing
is widened, and with the S3c toggle off the window scale is exactly `1.0`, so `compute_deadline`
reduces to the original formula term for term. 44 tests cover the three; the full unit suite is
**609 passing**.

**The honest caveat.** Inert-by-default means the *numbers* are reproducible, not that the frozen
files are unchanged — they are changed, and anyone re-deriving a committed CSV must check out the
matching commit rather than assume `main` reproduces it.

**Operational consequence — new CSV columns.** Rows now carry `mission_budget_s` and
`mission_window_adaptation`, so a results file says for itself which mechanisms were live. This
means **an existing CSV cannot be resumed** by a newer runner: it fails loudly with
`pass allow_schema_change=True to override`. That is the desired behaviour, not a defect — rows
recorded with these mechanisms active must never be pooled with historical rows, so being forced
into a new file is the correct outcome. Start a new CSV; do not override.

---

## 2. Phase 0 — Correctness & provenance *(highest priority)*

Audit equations, code behaviour, provenance, calibration, trial counts, questionable figures.

**Already closed** (this session — no action):

- [x] Deadline computed but never enforced → S3b gate, cost measured (−0.225 completion)
- [x] **S3b could starve the devices it dropped** → they never received a `RoundCloseDelta`, so Φ
      never widened and the same devices were dropped every mission. Closed by A2 (§1a); the mule
      also aborts a doomed remainder rather than flying it (A1), and S3c adds the mission-level
      signal the per-device rule cannot see (A3).
- [x] Selector unseeded → `rng_seed` threaded to the network
- [x] Zero-evaluation trials recorded `status=ok` → now `no_eval`, excluded from every metric
- [x] AUC rendered above 1.0 → bootstrap CIs, bounded axes, guards in `figstyle.py`
- [x] Layer-1 figure built from a different sweep than its table → regenerated, effect **retracted**
- [x] `ε_bit` fork (7.0e-10 vs 1.2e-9) → reconciled, marked verified

**Still open:**

- [ ] **`ε_prop` is a placeholder** (10.0 J/m, `REPLACE-FROM-PLATFORM-SPEC`). Until measured or
      explicitly declared, report **normalized energy only**. *(No re-run.)*
- [ ] **Manuscript deadline equation + Algorithm 1** still print the update with the sign reversed.
      Code is correct; the paper is not. *(No re-run.)*
- [ ] **Provenance table** for the paper: which results are Chameleon, AERPAW DT, simulation, full
      multi-process. *(No re-run.)*
- [x] ~~**Decide the status of inert L2 machinery**~~ — **resolved by the freeze.** S2A/S2B
      readiness and beacons / `BEACON_ACTIVE` are **removed from the claims, not wired**
      (Freeze D3, D6); `mule_energy` stays frozen at 1.0 and is stated as a scope limit. No re-run:
      nothing about trial behaviour changed.

---

## 3. Phase 1 — Freeze the scheduling methodology ✅ **DONE**

> **Frozen 2026-08-13** — see [`HERMES_Scheduler_Freeze.md`](HERMES_Scheduler_Freeze.md) for the
> six decisions (D1–D6), the frozen file surface, and the unfreeze procedure. Changing anything in
> that surface invalidates recorded sweeps.
>
> Decisions in brief: **D1** the S3b mechanism is frozen but its default is a *matrix* parameter,
> not a code default — so the enforcement choice is costed once in Phase 3; **D2** feasibility
> constants frozen (they still need a platform citation); **D3** S2A/S2B readiness removed from the
> claims rather than wired; **D4** Exp 4 makes no RL claim; **D5** `dead_zone` stays H0-only — that
> is correct, the error was sweeping it in a mule-only comparison; **D6** beacons stay unexercised.

<details><summary>Original task list (all resolved by the freeze)</summary>

- [x] **Decide the deadline-enforcement default.** Options: keep opt-in (committed results stand,
      but the scheduler ships non-enforcing), or make it default-on (honest, costs 29 % completion,
      forces a re-run of everything). *Pick one and write it down — this is the single biggest
      re-run driver.*
- [x] **If enforcing: fix the feasibility constants.** `cruise_speed_m_s`, `session_time_s`, and the
      budget itself. The probe shows the deadline floor (~34 % of contacts unreachable) is driven by
      cruise speed and field radius, not by the budget — so those constants *are* the experiment.
- [x] **Resolve S2A/S2B**: wire the real readiness gate (`FL_Threshold = 0.60`, 5 s advert freshness)
      or drop readiness from the contribution claims. Today an inline gate with `min_utility = 0.0`
      runs instead, which cannot reject anything.
- [x] **Decide the selector story.** Train weights (`experiments.exp3.train_a4`) and re-run H2/H3, or
      state plainly that Exp 4 makes no RL claim. Currently random-init, so H2-vs-H1 measures only
      within-bucket ordering.
- [x] **Decide whether `dead_zone` should apply to the mule arms.** Today it is H0-only, which is why
      the H2/H3 "dead-zone sweep" was one configuration under different seeds.
- [x] **Freeze.** After this point, scheduler changes invalidate sweeps.

</details>

---

## 4. Phase 2 — SOTA baseline research  ✅ **FULL-TEXT VERIFIED — blocker cleared**

> **Verified:** [`HERMES_SOTA_Baseline_Candidates.md`](HERMES_SOTA_Baseline_Candidates.md).
> **The full text overturned the first pass.** The scan recommended **FedCS** and dismissed
> **Oort**; the papers say the opposite:
>
> * **FedCS** has an explicit **Resource Request** step *before* selection — clients report channel
>   state, capacity and data size **every round**. A mule cannot obtain that without flying there.
> * **Oort** is retrospective by design — "a client's utility can only be determined **after** it
>   has participated" — with utilities cached from prior participation and an explicit staleness
>   term. That is exactly the state a mule has.
> * **Power-of-Choice** splits: `pow-d` does query loss pre-selection (as suspected), but the
>   published `rpow-d` variant reuses the last loss a client reported, and ports directly.
>
> **The capability argument survives but narrows.** The obstacle is not learned selection or
> utility ranking — it is specifically **pre-selection reporting**. That is a sharper and more
> defensible claim, and it says our architecture is compatible with the strong baselines.

The gap flagged by 74A. Closed at the reading stage; implementation choices remain.

- [x] Survey recent UAV-FL / FL-UAV scheduling work (Tier A general FL selection, Tier B
      UAV-specific, Tier C AoI/freshness).
- [x] Reduce each candidate to a decision rule and judge implementability.
- [x] **Full-text verification of every rule that affects the decision** — done; see the candidates
      doc §2 (correction table) and §6 (verification log). Three non-recommended candidates remain
      abstract-level and are flagged **not cleared for citation**.
- [x] **Choose the baselines.** **Oort** (cited, faithfully implementable, and its staleness term is
      a direct rival to our Φ-widening) + **MAX-AoI / staleness-greedy** (established named
      comparator, needs only last-served time), with **FedCS in degraded form** as the explicit
      capability contrast.
- [x] **Comparison mode — checked against the harness. Retroactive scoring is impossible.** Not
      just for Oort: the per-contact record lives in the run-dir JSONL, `consume_run_dir` folds it
      into aggregates, and `driver.py`'s `finally: orch.cleanup()` **deletes the trace at
      teardown**. The committed CSVs keep only aggregate loss. `RoundCloseDelta.utility` is an S2B
      readiness term (`w1·perf + w2·diversity`), not a training loss, so it is not a stand-in for
      Oort's utility. **Both chosen baselines are therefore re-run items** — see the new ledger row
      in §1.
- [x] ✅ **Trace retention implemented** — `--keep-event-traces` (plus `--trace-dir`). Off by
      default, changes **no** trial behaviour; it only stops the run dir being deleted. Verified
      end-to-end: a captured trace yields per-contact `device_served` / `device_serve_failed`
      events with timestamps **and** device positions (positions live only in the configs, so both
      `*.jsonl` and `*.json` are kept), from which last-served time — what MAX-AoI needs — is
      directly derivable. Traces are captured **before** the timeout check, so timed-out trials
      keep theirs too. **Cost: ~9.7 KB per trial, ≈2.3 MB for a 240-trial matrix.**
      *Known limitation:* `device_served` carries no `mission_round`; round attribution requires a
      timestamp interval join against `mission_started` / `mission_completed`. Recoverable, but not
      a direct field — worth knowing before writing a scorer.
      **Use it on the matrix run.** Without it, the next comparator a reviewer asks for costs a
      third full re-run; with it, future baselines are re-parses.
- [ ] Document the chosen baselines + the fairness argument (same harness, same seeds) before running.
- [x] **Related Work material captured** — [`HERMES_Related_Work_Notes.md`](HERMES_Related_Work_Notes.md)
      holds the reading, the verified findings, the architectural taxonomy, and the starvation/fairness
      thread that positions Amendments 1–2 against prior art. *(No re-run; it is a writing input.)*

---

## 5. Phase 3 — Build the final experimental matrix

### 5.0 Pre-step — the S3c pilot *(runs after Phase 2, before the matrix is fixed)*

**Why this is a pre-step and not a matrix cell.** If S3c enters the matrix as an axis it **doubles
every cell**. A short pilot decides whether it belongs there at all — and all three possible
outcomes make the matrix *smaller or cheaper*, which is the same trade this whole document exists
to enforce:

| Pilot outcome | Consequence for the matrix |
|---|---|
| **Null even with the budget on** | Drop S3c from the matrix entirely. Report it as a negative result: the mechanism exists, is tested, and does not move the metric at this operating point. |
| **Recovers part of enforcement's −0.225 completion cost** | That *is* S3c's headline, and it is a paired one-flag comparison at fixed budget — not an axis. |
| **Only bites at tight budgets** | Pin it to a single setting rather than sweeping it. |

**Sequencing.** After the Phase-2 SOTA baseline, because that settles the **arm list** and S3c
cannot sensibly be designed against a matrix whose arms are still open. Before the matrix, because
that is the only point at which the pilot can still save the doubling.

**Costings** — measured from recorded sweeps (`exp4_s3b` n=100, `h2h3_l1` n=80) plus a stub run at
the target configuration, at **concurrency 3** (at 5 shards the box exhausted memory and ~30 % of
trials failed):

| Design | Trials | Est. wall-clock |
|---|---|---|
| **Stub pilot, 1 cell, 20 paired seeds** ← *start here* | 40 | **~8 min** |
| Real-model, 1 cell, `n_missions=6` | 40 | ~19 min |
| Real-model, `n_missions=8` | 40 | ~27 min |
| + budget ladder (120 / 60 / 30 s, spanning the ~60 s knee) | 120 | ~80 min |
| + second arm (H1 **and** H3) | 240 | ~2 h 40 m |

Per-trial means behind those figures: real-model `n_missions=4` **37.8 s** (p90 52.6), real-model
`n_missions=6` **69.5 s** (p90 81.5), stub `n_missions=6` **26.7 s**.

> **Design trap — do not run this pilot at `n_missions=4`.** The default history window is 5, and
> the first mission has no history at all, so at the headline mission count S3c can barely act. An
> A/B there would likely show nothing **whether or not the mechanism works** — a null that cannot
> be interpreted, which is the worst outcome available. Use `n_missions≥8`, or lower
> `--mission-window-history` to 3, and say which in the write-up.

- [x] Run the stub pilot (2 configurations × 20 paired seeds, budget on, `n_missions≥8`), **with
      `--keep-event-traces`**. ✅ **Done 2026-08-13** — 40/40 trials `ok`, seeds matched on every
      pair, traces retained. `results/exp4_s3c/{off,on}.csv`.
- [x] Record which of the three outcomes it produced. ✅ **See §5.0a below.**
- [ ] Carry the §5.0a decision (**pin, do not sweep**) into §5.1's axis list.

### 5.0a Pilot result — non-null, but narrow and *transient*

**Configuration:** H1, N=6, `rrf=50`, `n_missions=8`, `--realism`, `--mission-budget-s 120`,
20 paired seeds, stub. Arms differ by **exactly one flag**; `mission_budget_s` identical, seeds
matched on all 20 pairs (both verified from the provenance columns).

| Metric | off | on | diff | CI95 | p | δ | |
|---|---|---|---|---|---|---|---|
| **update_yield** | 1.6062 | 1.8000 | **+0.1938** | [+0.0625, +0.3127] | **0.0178** | +0.278 | **meets the claim rule** |
| rounds_closed | 6.700 | 7.150 | +0.450 | [+0.0000, +0.9000] | 0.0701 | +0.210 | tie (near) |
| mission_completion_rate | 0.7000 | 0.6750 | −0.0250 | [−0.0752, +0.0250] | 0.6180 | −0.070 | tie |
| coverage / missions_completed | — | — | 0 | — | 1.0 | 0 | identical on every pair (ceiling) |
| jains_fairness, participation_entropy, completion_fairness | — | — | ≈0 | straddles 0 | >0.49 | — | tie |

**⚠ Two caveats that must travel with this number.**

1. **It does not survive multiplicity correction.** Eight metrics were tested; Bonferroni/Holm put
   the threshold at α = 0.05/8 = **0.00625**, and p = 0.0178 exceeds it. Under the null, testing
   eight metrics gives ~34 % odds of at least one hit at α = 0.05. On the p-value alone this is
   **suggestive, not established** — and this project has already retracted one result that looked
   like this.
2. **Completion moved the other way** (−0.025, not significant). More devices served costs mission
   time. If this holds up it is a **trade-off, not a free win**, and should be reported as one.

**What rescues it from "probably noise" — the mechanism was verified independently of the
statistics,** from the retained traces:

| Mission round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| post-S3b queue, off | 2.10 | 2.35 | 2.75 | 2.75 | 3.25 | 3.25 | 3.65 | 3.40 |
| post-S3b queue, on | 2.10 | 3.45 | 3.45 | 3.10 | 3.40 | 3.50 | 3.50 | 3.65 |
| diff | **0.00** | +1.10 | +0.70 | +0.35 | +0.15 | +0.25 | −0.15 | +0.25 |

* **Round 1 is exactly identical** — no history, so the scale is exactly 1.0. The inert-by-default
  guarantee, visible in live data rather than only in a unit test.
* **Rounds 2–4 diverge, then converge.** The `off` arm reaches a similar queue size eventually,
  using per-device widening alone.
* Served events per trial: **72.55 → 78.65 (+8.4 %)**.

**Interpretation — and this is the useful finding.** S3c does not reach a *better* steady state; it
**reaches the workable state faster**. The per-device rule gets there on its own, given enough
missions. S3c's value is in the warm-up, exactly where a per-device rule has the least information.

**Testable prediction this generates:** the advantage should **shrink as `n_missions` grows** and
**grow at small `n_missions`**. That is a sharp, falsifiable claim and a much better basis for a
paper sentence than "+0.19 update yield".

**Decision — pin, do not sweep.** It is not an axis (it does not interact with the other axes; it
interacts with *mission count*). Recommended: fix `--mission-window-adaptation` on or off in the
matrix, and if the transient claim is worth making, test it with a **small `n_missions` ladder**
(4 / 8 / 16 at one cell) rather than by doubling every cell. That ladder is the confirmatory run
that would also settle caveat 1.

Both configurations write **one new CSV each** — an existing file cannot be resumed now that rows
carry provenance columns (§1a).

### 5.1a The two decisions that block the matrix *(no compute — design calls)*

The matrix cannot be written, let alone costed, until these are settled. Both were deliberately
deferred to this point so each is paid for once.

#### Decision A — is deadline enforcement on in the headline?

Freeze **D1** left this to the matrix on purpose. The facts, all measured:

* **On:** the scheduler the paper describes — a deadline that actually binds. Costs **−0.225
  mission completion (−29 %)**, and carries Amendments A1 (in-flight abort) and A2 (starvation
  fix) with it, since both ride the same toggle. Every published participation figure changes.
* **Off:** reproduces the committed numbers, but ships a scheduler whose "deadline" is a sort key.
  That is exactly what the L2 trace flagged, and it is only defensible if stated plainly in the
  text.

> **Recommendation: on.** The paper's contribution is a *deadline-aware* scheduler; a 29 % honest
> cost is easier to defend than a deadline that does not bind, and the audit already found it once.

#### Decision B — do the SOTA baselines ship as arms, and which?

**They do not exist in code.** `ARMS = ("H0","H1","H2","H3")`; nothing implements Oort, MAX-AoI or
FedCS. Neither is retroactively scorable (§4), so each is a **new arm** — implementation work that
must be scoped *before* the matrix is costed rather than discovered inside it.

The two are **not** equal effort, and this asymmetry should drive the choice:

| Baseline | What it needs | State available today? | Effort |
|---|---|---|---|
| **MAX-AoI / staleness-greedy** | last-served time + positions, ranked highest-age first with nearest-predecessor pathing | **Yes — already in `DeviceSchedulerState`** (`last_contact_ts`, `idle_time_ref_ts`, `missed_count`); positions known; slots into the existing `target_selector` extension point | **Low** — a new selector, no new data path |
| **Oort** | statistical utility `\|B_i\|·√(mean Loss²)` + speed + staleness | **No.** There is **no per-device training loss anywhere in the device→mule types.** `RoundCloseDelta.utility` is `w1·perf + w2·diversity`, an S2B readiness term — not a loss, and using it would be a different algorithm wearing Oort's name | **High** — new data path: device computes loss → new protocol field → selector |

**Both touch the frozen L2 surface** (`selector/`), so this is **Freeze Amendment 3**. As with
Amendments 1–2: doing it *before* the matrix costs nothing, doing it *after* costs a second full
re-run.

> **Recommendation: MAX-AoI first, Oort second.** MAX-AoI is cheap, needs no new data, and is the
> UAV/AoI-shaped comparator 74A actually asked for — it directly rivals our bucket+deadline
> ordering. Oort is the more citable name but costs a protocol change; take it if the schedule
> allows, and if not, Related Work already states precisely why (§3 of the candidates doc) rather
> than pretending it was infeasible.

### 5.1 The matrix itself

Finalize E1–E4, baselines, scaling, sensitivity, statistics, provenance labels — **once**.

- [ ] Arms: H0, H1, H2, H3 + whichever baselines §5.1a-B selects. State which pairwise comparisons
      are valid (today: H1-vs-H0 and H3-vs-H2 only; H2/H3-vs-H0/H1 is **not** paired). A new
      baseline arm is paired with the mule arms only if it shares their backhaul model.
- [ ] **S3c: pinned, not swept** (§5.0a). It interacts with *mission count*, not with the other
      axes, so it is a fixed setting here. If the transient claim is wanted, cost the separate
      `n_missions` 4/8/16 ladder instead of doubling every cell.
- [ ] Axes and their ranges: `dead_zone × link_quality`, regime, N, mule count, `n_missions`,
      mission budget.
- [ ] **Decide the two scheduler toggles (§1a).** `--mission-budget-s` carries A1+A2 with it;
      `--mission-window-adaptation` is independent. Both default off, so *not* deciding means
      shipping a scheduler whose deadline is a sort key — defensible only if stated plainly.
- [ ] **Window adaptation enters as a paired A/B, never as a new default** — and only if §5.0's
      pilot says it earns a place. It is a one-flag delta on an otherwise identical configuration,
      which is the cheapest clean comparison available and the reason it was built as a toggle. Its
      own parameters (`--mission-window-target`, `--mission-window-gain`,
      `--mission-window-history`, `--mission-window-max-scale`) are matrix values, not code
      defaults — the same rule as D1. Expect it to matter **only** when the S3b gate binds; with no
      budget there is nothing for a wider window to rescue, so an adaptation-only arm should read
      as a tie by construction.
- [ ] Seeds per cell (≥20) and the pairing key.
- [ ] Statistics: paired Wilcoxon + Cliff's δ + bootstrap CI; claim only when CI excludes 0 **and**
      p<0.05.
- [ ] Provenance label per row.
- [ ] **Cost the matrix in wall-clock before launching**, and cap shard concurrency at **3** — at 5
      concurrent shards the box exhausted memory and ~30 % of trials failed. Use the measured
      per-trial means in §5.0 rather than the old ~47 s rule of thumb: cost depends strongly on
      `n_missions` (37.8 s at 4, 69.5 s at 6, real-model).

### Exit criteria — re-run only when all are true

1. Phase 0 open items are closed **or** explicitly deferred with a written reason.
2. The scheduler is **frozen** (§3), including the enforcement decision.
3. Baselines are chosen, and anything scorable retroactively has been scored (§4).
4. **The S3c pilot has run and its outcome is recorded** (§5.0) — so the matrix is not doubled by
   an axis nobody has evidence for.
5. The matrix is written down and costed (§5.1).
6. Every item in the §1 ledger marked "forces a re-run" is either **in** this matrix or **out** of
   the paper.

---

## 6. Phase 4 — Run / re-run / re-parse

**Prefer re-analysis of valid existing data.** Run new trials only where needed to answer an
unresolved methodological claim.

**Existing assets — reusable without re-running:**

| Asset | Rows | Still valid for |
|---|---|---|
| `results/exp4_paper/h0h1_all.csv` | 519 ok | The crossover surface (§6 of the validity record) |
| `results/exp4_paper/h2h3_dz_*.csv` | 200 ok | The L1 null — but see the dead-zone caveat |
| `results/exp4_s3b/*.csv` | 100 ok | Deadline-enforcement cost ladder |
| `results/exp4_paper/h2h3_l1.csv` | 80 ok | §7.3 single-cell L1 result |

**Caveat carried on all of them:** recorded with deadline enforcement **off**, window adaptation
**off**, and an **unseeded, untrained** selector. Fine for the comparisons they support (both arms
shared the configuration); not fine as absolute participation figures. None of them carry the
`mission_budget_s` / `mission_window_adaptation` provenance columns, because they predate them —
their absence *is* the provenance, and it is why they cannot be resumed or extended in place
(§1a).

**Sequence when the gate opens:** re-parse first → then the smallest sweep that answers what
re-parsing cannot → then the full matrix.

---

## 7. Deferred, with reasons

| Item | Why deferred |
|---|---|
| Re-run the headline H0/H1 sweep with enforcement on | Would change §6's published numbers; batched into the Phase-3 matrix so it is paid once |
| Unified cross-layer energy (L1 switching + L2 flight + L3 compute) | Needs `ε_prop`; energy stays normalized until then |
| GAN direction | Deferred to the journal/extended version; criteria to un-defer are in the revision plan §6.0 |
| Switching-induced upload loss in the L1 loss metric | Documented as a caveat; the reported gain is an upper bound |

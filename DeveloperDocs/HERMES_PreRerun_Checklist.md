# Pre-re-run checklist — settle these before spending compute

**Purpose.** Every sweep costs ~30–60 min of the box, and each unsettled decision that lands
*after* a sweep invalidates it. This document is a **gate**: we re-run when everything below is
either done or explicitly deferred with a reason. It exists so we pay the compute once.

**Status:** open. Nothing is scheduled to re-run until §5's exit criteria are met.

**Progress:** Phase 0 — 8 of 11 closed (3 open, all no-re-run). Phase 1 — **done, scheduler
frozen**, amended twice since (see §1a). Phase 2 — first pass done, **full-text verification is the
current blocker**. Phase 3 — not started; its **§5.0 S3c pilot is costed and ready to run** the
moment Phase 2 clears. Phase 4 — not started.

**Next action when Phase 2 clears:** the ~8-minute stub pilot in §5.0. It is the cheapest decision
on the board — it can only shrink the matrix.

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

## 4. Phase 2 — SOTA baseline research  *(first pass done)*

> **First pass:** [`HERMES_SOTA_Baseline_Candidates.md`](HERMES_SOTA_Baseline_Candidates.md) —
> a scoped scan, **abstract-level only, not yet full-text verified**. Recommends picking **two**:
> a **FedCS-style deadline-feasibility** selector (the closest analogue to our own gates) and an
> **AoI/staleness-greedy** policy (implementable on mule-visible state, and retroactively
> scoreable). It also argues why Oort / Power-of-Choice are *not* faithfully implementable here —
> they assume the server can poll clients before choosing, which a data mule cannot — and that this
> is a capability argument worth making in Related Work rather than a gap to apologise for.
>
> Run `/deep-research` (prompt in that document's §5) for the exhaustive, verified version.

The gap flagged by 74A and still unanswered. Do the reading **before** the matrix is fixed, so a
baseline arm can be designed in rather than bolted on.

- [x] Survey recent UAV-FL / FL-UAV scheduling work — **first pass done** (Tier A general FL
      selection, Tier B UAV-specific, Tier C AoI/freshness). Abstract-level only.
- [x] Reduce each candidate to a decision rule and judge implementability — **done for the scan**,
      including the finding that Oort / Power-of-Choice assume the server can *poll* clients before
      choosing, which a data mule cannot, so they are not faithfully implementable here.
- [ ] **Full-text verification of every rule.** ⚠ **Blocking** — the scan is abstract-level, so
      treat its readings as hypotheses. Run `/deep-research` (prompt in the candidates doc §5) or
      read the papers directly before implementing or citing anything.
- [ ] **Decide the comparison mode — retroactive first.** Recommendation on the table: **FedCS-style
      deadline-feasibility** + **AoI/staleness-greedy**, both scoreable against committed data with
      no re-run. Needs confirming against the harness (does it expose last-served time and
      per-contact history?).
- [ ] Document the chosen baselines + the fairness argument (same harness, same seeds) before running.

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

- [ ] Run the stub pilot (2 configurations × 20 paired seeds, budget on, `n_missions≥8`).
- [ ] Record which of the three outcomes above it produced, **in this document**, before the matrix
      is fixed.
- [ ] Only if it is non-null: decide axis-vs-pinned and carry the decision into §5's arm/axis lists.

Both configurations write **one new CSV each** — an existing file cannot be resumed now that rows
carry provenance columns (§1a).

### 5.1 The matrix itself

Finalize E1–E4, baselines, scaling, sensitivity, statistics, provenance labels — **once**.

- [ ] Arms: H0, H1, H2, H3 + SOTA baseline(s). State which pairwise comparisons are valid
      (today: H1-vs-H0 and H3-vs-H2 only; H2/H3-vs-H0/H1 is **not** paired).
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

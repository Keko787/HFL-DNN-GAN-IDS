# Pre-re-run checklist — settle these before spending compute

**Purpose.** Every sweep costs ~30–60 min of the box, and each unsettled decision that lands
*after* a sweep invalidates it. This document is a **gate**: we re-run when everything below is
either done or explicitly deferred with a reason. It exists so we pay the compute once.

**Status:** open. Nothing is scheduled to re-run until §5's exit criteria are met.

**Progress:** Phase 0 — 7 of 10 closed (3 open, all no-re-run). Phase 1 — **done, scheduler
frozen**. Phase 2 — first pass done, **full-text verification is the current blocker**.
Phases 3–4 — not started.

---

## 1. The re-run ledger — what actually forces a re-run

The single most useful distinction here. Most open items do **not** need new trials.

| Change | Forces a re-run? | Why |
|---|---|---|
| Turn on deadline enforcement (`--mission-budget-s`) | **Yes** | Changes participation by ~29 % (measured) |
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

---

## 2. Phase 0 — Correctness & provenance *(highest priority)*

Audit equations, code behaviour, provenance, calibration, trial counts, questionable figures.

**Already closed** (this session — no action):

- [x] Deadline computed but never enforced → S3b gate, cost measured (−0.225 completion)
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

Finalize E1–E4, baselines, scaling, sensitivity, statistics, provenance labels — **once**.

- [ ] Arms: H0, H1, H2, H3 + SOTA baseline(s). State which pairwise comparisons are valid
      (today: H1-vs-H0 and H3-vs-H2 only; H2/H3-vs-H0/H1 is **not** paired).
- [ ] Axes and their ranges: `dead_zone × link_quality`, regime, N, mule count, `n_missions`,
      mission budget.
- [ ] Seeds per cell (≥20) and the pairing key.
- [ ] Statistics: paired Wilcoxon + Cliff's δ + bootstrap CI; claim only when CI excludes 0 **and**
      p<0.05.
- [ ] Provenance label per row.
- [ ] **Cost the matrix in wall-clock before launching** (trials × ~47 s ÷ concurrency), and cap
      shard concurrency — at 5 concurrent shards the box exhausted memory and ~30 % of trials failed.

### Exit criteria — re-run only when all are true

1. Phase 0 open items are closed **or** explicitly deferred with a written reason.
2. The scheduler is **frozen** (§3), including the enforcement decision.
3. Baselines are chosen, and anything scorable retroactively has been scored (§4).
4. The matrix is written down and costed (§5).
5. Every item in the §1 ledger marked "forces a re-run" is either **in** this matrix or **out** of
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

**Caveat carried on all of them:** recorded with deadline enforcement **off** and an **unseeded,
untrained** selector. Fine for the comparisons they support (both arms shared the configuration);
not fine as absolute participation figures.

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

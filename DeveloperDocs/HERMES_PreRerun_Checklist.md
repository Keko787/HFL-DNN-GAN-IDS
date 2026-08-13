# Pre-re-run checklist — settle these before spending compute

**Purpose.** Every sweep costs ~30–60 min of the box, and each unsettled decision that lands
*after* a sweep invalidates it. This document is a **gate**: we re-run when everything below is
either done or explicitly deferred with a reason. It exists so we pay the compute once.

**Status:** ✅ **GATE OPEN** (2026-08-13). All six exit criteria met; the matrix in §5.1 is cleared to run.

**Progress:** Phase 0 — 8 of 11 closed (3 open, all no-re-run). Phase 1 — **done, scheduler
frozen**, amended twice since (§1a). Phase 2 — **done, full-text verified**; baselines chosen, and
the reading *reversed* the first pass's recommendation (§4). Phase 3 — **§5.0 pilot run, result in
§5.0a**; the matrix itself (§5.1) is **the remaining blocker**. Phase 4 — not started.

**Exit criteria status: 6 of 6 met — THE GATE IS OPEN.** ✅ 1 (Phase-0 items closed: two deferred
with reasons, provenance table written — §2, §2a) · ✅ 2 (frozen; amended three times, each inert by
default) · ✅ 3 (baselines chosen **and implemented**) · ✅ 4 (pilot recorded, §5.0a) · ✅ 5 (matrix
written and costed, §5.1) · ✅ 6 (every "forces a re-run" ledger item is **in** the matrix or
explicitly **out**).

**The matrix is written and costed: 680 trials, ≈2.4 h at concurrency 3** (§5.1). Six arms exist:
`H0 H1 H2 H3 B1 B2`. Enforcement is **ON**; S3c is **pinned off**.

**RUNNING (2026-08-13).** Sweeps **A** (H0 vs H1, 520 trials) and **C** (H2 vs H3, 40 trials) are
executing — `experiments/exp4/run_matrix.sh` → `results/exp4_matrix/`.

> **Sweep A is two invocations, not one cross product** — and this was caught only after a first
> launch had to be killed. `dead_zone` and `link_quality` describe how the clean regime is *degraded
> into* the jittery one; under `clean` they are **inert**. Passing all three axes to one grid gives
> 2 × 4 × 3 = **24 cells**, of which the 12 clean ones are the *same configuration repeated* — ~440
> wasted trials, and 12 cells that look distinct in analysis but are not. Running clean at one point
> and jittery across the surface gives 1 + 12 = **13 cells / 520 trials**, matching the committed
> design and the cost model. *Same class of error as Freeze D5: sweeping an axis that varies nothing
> for the condition under test.*

**Sweep B is ON HOLD (§5.1b).** The pre-launch smoke found H1/B1/B2 produce **identical** results at
the designed operating point: S3b fixes *who* is served before the policy runs, and the policy only
reorders an already-decided set. It needs re-targeting at the binding band first — a redesign is
proposed in §5.1b and needs a decision, because the band sits in a heavily-degraded regime.

> **Two constraints the matrix respects.** `B2` is **real-model-only** (the stub's loss is a random
> draw), so it cannot appear in stub pilots. And **`--keep-event-traces` goes on every run** (§4) —
> without it, the next comparator a reviewer asks for costs another full re-run.

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

**Now closed — all three resolved 2026-08-13, none forces a re-run:**

- [x] **`ε_prop` is a placeholder (10.0 J/m)** → **DEFERRED, with the deferral already enforced in
      code.** Two facts settle this:
      1. It is an **Experiment-3 constant** (`exp3.epsilon_prop_J_per_m`). **Experiment 4 models no
         propulsion energy at all** (scope boundary), so it cannot affect the matrix — this was
         never an Exp-4 blocker.
      2. The deferral is **not a promise, it is wired**: `calibration.toml` marks it
         `"exp3.epsilon_prop_J_per_m" = "placeholder"`, `Calibration.exp3_is_paper_grade` returns
         False as a result, and `analysis/exp3.py:1758` passes
         `placeholder_watermark=not cal.exp3_is_paper_grade` — so **every Exp-3 energy figure is
         watermarked automatically** until a platform spec replaces the value.

      *Action to un-defer:* replace the value in `calibration.toml` and flip its status to
      `verified`; the watermark disappears on its own. Report **normalized energy only** until then.

- [x] **Manuscript deadline equation + Algorithm 1 print the update with the sign reversed** →
      **DEFERRED — prose-only, and it cannot be fixed from this repository.** The only manuscript
      artefact here is `Presentation Documents/sec26-paper74.pdf`, a compiled PDF; the LaTeX source
      lives with the author. Recording the exact correction so it can be applied without re-deriving
      it. **The code is correct**; the paper is not:

      | Outcome | What the code does (`s3_deadline.py:190-200`) | Effect on Φ |
      |---|---|---|
      | **CLEAN** (served on time) | `Φ ← max(MIN, Φ − FAST_PHASE_ON_TIME_SHRINK_S)`, shrink **5 s** | **narrows** |
      | **MISSED** | `Φ ← Φ + FAST_PHASE_MISSED_WIDEN_S`, widen **10 s** | **widens** |

      The intuition the text must match: *a device we keep reaching earns a tighter window; a device
      we miss earns a looser one.* Widening is deliberately **twice** the shrink (10 s vs 5 s), so
      recovery from neglect is faster than the drift back to strictness — that asymmetry is a design
      choice and worth one sentence in the paper.

- [x] **Provenance table** → **WRITTEN**, below (§2a).

- [x] ~~**Decide the status of inert L2 machinery**~~ — **resolved by the freeze.** S2A/S2B
      readiness and beacons / `BEACON_ACTIVE` are **removed from the claims, not wired**
      (Freeze D3, D6); `mule_energy` stays frozen at 1.0 and is stated as a scope limit. No re-run:
      nothing about trial behaviour changed.

## 2a. Provenance table — what each result actually ran on

Written to close the Phase-0 item. **Every claim in the paper must carry one of these labels**, so a
reader never has to guess whether a number came from hardware, a simulator, or a process tree on one
box.

| Experiment | Environment | What that means concretely | Artefacts |
|---|---|---|---|
| **Exp 1** | **Chameleon testbed** | Real distributed hardware, real network between nodes. The strongest provenance we have. | `results/exp1_chameleon*.csv` |
| **Exp 3** | **Simulation** (`experiments/exp3/sim_env.py`) | An abstracted event simulator. **No processes, no sockets.** Flight budget, propulsion energy and the deadline *design* live here — and so does the `ε_prop` placeholder, which is why Exp-3 energy figures carry a watermark. | `results/exp3*/` |
| **Exp 4** | **Full multi-process, single host** | Real OS processes, real TCP sockets, real TensorFlow training, real two-pass hierarchical FedAvg — but all on **one machine over loopback**. Not a distributed deployment: latency and bandwidth are loopback, not WAN. | `results/exp4_paper/`, `results/exp4_s3b/`, `results/exp4_s3c/` |
| **AERPAW digital twin** | **NOT USED** | No AERPAW artefacts exist in this repository. Any figure or text implying AERPAW provenance is **wrong** and must be corrected — see the architecture review's divergence **D-1**, which found Fig. 2 annotating the RF selector "Training: CTDE on AERPAW digital twin" while §III-A correctly calls it future work. | — |

**Three honesty notes that belong with the table:**

1. **Exp 4 is "real processes", not "real deployment".** The distinction matters: it validates the
   *software integration* — that L2 and L3 actually work end to end over sockets — not network
   behaviour under real RF or WAN conditions.
2. **Exp 3 and Exp 4 measure different things and must not be pooled.** The scheduling results in
   the manuscript come from `sim_env.py`, not the multi-process topology (architecture review
   divergence **D-3**).
3. **The GAN contribution has never executed inside HERMES** (divergence **D-2**):
   `StubGeneratorHost` returns zero tensors and `exp4/model_task.py` discards the synth batch. Real
   DNN-IDS weights *are* aggregated; the generator's contribution is not. Any claim that HERMES
   exercises the GAN is unsupported by these results.

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

> ### ✅ **DECIDED 2026-08-13: enforcement is ON in the headline matrix.**
> Consequences, all now committed to:
> * Every mule-arm participation figure changes; §6's published numbers are superseded, not
>   amended. The −0.225 completion cost becomes a **reported result**, not a footnote.
> * A1 (in-flight abort) and A2 (starvation widening) are **live** in every headline trial, since
>   they ride this toggle. They stop being "implemented but unexercised".
> * `mission_budget_s` must be **fixed and justified** as a matrix value. The S3b probe puts the
>   deadline floor at ~34 % of contacts at *any* budget (driven by cruise speed and field radius,
>   not the budget) with a knee near 60 s — so the chosen value needs a stated rationale, not a
>   round number.

#### Decision B — do the SOTA baselines ship as arms, and which?

**They do not exist in code.** `ARMS = ("H0","H1","H2","H3")`; nothing implements Oort, MAX-AoI or
FedCS. Neither is retroactively scorable (§4), so each is a **new arm** — implementation work that
must be scoped *before* the matrix is costed rather than discovered inside it.

The two are **not** equal effort, and this asymmetry should drive the choice:

| Baseline | What it needs | State available today? | Effort |
|---|---|---|---|
| **MAX-AoI / staleness-greedy** | last-served time + positions, ranked highest-age first with nearest-predecessor pathing | **Yes — already in `DeviceSchedulerState`** (`last_contact_ts`, `idle_time_ref_ts`, `missed_count`); positions known; slots into the existing `target_selector` extension point | **Low** — a new selector, no new data path |
| **Oort** | statistical utility `\|B_i\|·√(mean Loss²)` + speed + staleness | **No.** There is **no per-device training loss anywhere in the device→mule types.** `RoundCloseDelta.utility` is `w1·perf + w2·diversity`, an S2B readiness term — not a loss, and using it would be a different algorithm wearing Oort's name | **High** — new data path: device computes loss → new protocol field → selector |

**Correction (checked, not assumed): MAX-AoI needs _no_ freeze amendment.** An earlier draft of
this section said both baselines touch the frozen surface. They do not. `hermes/scheduler/policies/`
is **absent from the frozen file list** (§5 of the freeze), and so is `experiments/exp4/driver.py`.
The project already ships `ArrivalOrderPolicy` and `EdfFeasibilityPolicy` there, both exposing the
same `rank_contacts` call shape as `TargetSelectorRL` and swapped through the same constructor
slot — so a MAX-AoI arm is *a new file in an unfrozen package plus driver wiring*.

**Oort still does** need one, because it requires a new per-device loss field on the device→mule
path. That part remains a **Freeze Amendment 3** when we get to it.

#### Oort — exact implementation scope *(traced against the code, 2026-08-13)*

**Smaller than expected. The transport already exists.** `adv.utility` already travels
device → `FLReadyAdv` → `RoundCloseDelta` → `fold_round_close_delta` → `state.last_utility`. Oort
needs two more values down that same live pipe, not a new pipe.

**Both ingredients are already computed and then thrown away.** `LocalTrainResult` carries
**`loss`** and **`num_examples`** — exactly Oort's `√(mean Loss²)` and `|B_i|`. Today
`_update_utility` reads them only to derive `performance_score` and discards the raw values.

| File | Change | ~LOC | Frozen? |
|---|---|---|---|
| `hermes/types/scheduler.py` | 2 optional fields each on `FLReadyAdv`, `RoundCloseDelta`, `DeviceSchedulerState` | ~8 | no |
| `hermes/mission/client_mission.py` | retain `result.loss` / `result.num_examples`; pass them in `build_ready_adv` | ~8 | no |
| `hermes/mission/host_mission.py` | copy both into `RoundCloseDelta` at its 3 construction sites | ~6 | no |
| `hermes/scheduler/stages/s3_deadline.py` | fold both into state, beside the existing `last_utility` line | **~2** | **YES ⇒ Amendment 3** |
| `hermes/scheduler/policies/oort.py` | new `OortPolicy` | ~90 | no |
| `experiments/exp4/driver.py` | arm `B2` | ~4 | no |
| `tests/unit/test_oort_policy.py` | policy + plumbing | ~180 | — |

**Total ≈ 300 LOC, and the frozen surface is touched by two lines** that are strictly inert for
H0–H3 (the new fields default to `None`/`0`, so existing arms fold nothing). Comparable to, or
slightly smaller than, the S3c work.

**Three judgement calls that must be decided and _stated_, not silently resolved:**

1. **⚠ Oort is real-model-only — this is the real constraint, not the code.** The stub's
   `LocalTrainResult` returns `loss=uniform(0.1,0.3)` and `num_examples=randint(4,16)` — **pure
   noise**. Ranking on that is a random-ordering baseline wearing Oort's name, which is worse than
   not running it. *Cost impact is nil for the headline* (H0 already forces `--real-model`, so the
   matrix is real-model anyway), but it means **Oort cannot be smoke-tested on the fast stub path**,
   which slows iteration and rules it out of cheap pilots.
2. **The system-speed term.** Oort's utility multiplies statistical utility by a straggler penalty
   over client compute/comm speed. **We have no per-device compute speed.** Either drop the term
   (and say so) or map it to per-device contact reliability (and say so). Dropping is the more
   honest default: it is the *statistical utility* half of Oort, and the paper should call it
   "Oort's statistical-utility selection" rather than "Oort".
3. **Loss semantics.** Oort defines `√(Σ_k Loss(k)²/|B_i|)` — the RMS over per-sample losses. Keras
   `evaluate` hands us the **mean** loss. Close, monotone in the same direction, **not identical**.
   Record it as a stated approximation rather than letting a reader assume exactness.

**Verdict: ~300 LOC, one small freeze amendment, and no new data path — but three fidelity
statements the paper must carry.** The honest framing is that we implement Oort's *statistical
utility + staleness* ranking on mule-visible state, which is precisely the part that ports.

> ### ✅ **IMPLEMENTED as arm `B2` — Freeze Amendment 3, 2026-08-13.**
> `hermes/scheduler/policies/oort.py`; `--arms B2`. Estimate held: the frozen surface was touched
> by **three assignments** in `fold_round_close_delta`, all inert for H0–H3 and pinned by test.
> 25 new tests; **676 unit tests pass**.
>
> Verified end to end in-process rather than inferred from a green trial: a real `LocalTrainResult`
> loss reaches `statistical_utility` intact through advertisement → delta → fold.
>
> **Both guards fire.** The driver refuses `B2` without `--real-model`; the policy raises
> `OortUnusableError` if devices have been served but no loss arrived. The stub's random loss can
> never masquerade as a ranking signal.
>
> **Still to do:** add B1/B2 to §5.1's arm list and cost them. Note B2 is real-model-only, so it
> cannot appear in stub pilots.

> **Recommendation: MAX-AoI first, Oort second.** MAX-AoI is cheap, needs no new data, and is the
> UAV/AoI-shaped comparator 74A actually asked for — it directly rivals our bucket+deadline
> ordering. Oort is the more citable name but costs a protocol change; take it if the schedule
> allows, and if not, Related Work already states precisely why (§3 of the candidates doc) rather
> than pretending it was infeasible.

> ### ✅ **DECIDED 2026-08-13: MAX-AoI first, then Oort if the schedule allows.**
>
> **Fairness design — what a baseline arm may and may not replace.** A baseline must replace our
> *policy*, not our *physics*, or the comparison is rigged in either direction:
>
> | Stage | Baseline arm | Why |
> |---|---|---|
> | **S1** eligibility | **kept** | slice membership is structural, not a policy choice |
> | **S3a** RF clustering | **kept** | contacts are a physical fact of `rf_range_m`, not a ranking |
> | **S3b** feasibility | **kept** | both arms must face the *same* budget constraint or the comparison is meaningless |
> | **S3** bucket tiers | **replaced** | this *is* our policy |
> | **S3.5** ordering | **replaced** | this *is* our policy |
>
> So the arm is `S1 → S3a → S3b → order by age descending`, with distance as the tie-break
> (matching the literature's greedy form: highest AoI, then nearest predecessor).
>
> **The bucket-tier worry turned out to be moot — verified, not assumed.** The concern was that a
> policy in the `target_selector` slot only re-orders *within* a bucket, so our tier order would
> outrank the baseline's ranking and flatter us. Probing the real scheduler across six rounds shows
> **every round has exactly one non-empty bucket** — `NEW` in round 1 (all devices are new),
> `SCHEDULED_THIS_ROUND` thereafter, with `BEACON_ACTIVE` never populated (Freeze D6). So the tier
> walk never discriminates in Exp 4, and a policy in that slot orders the **whole round**. It is a
> faithful full-ordering baseline through the existing extension point.
>
> *Carry this into the write-up:* the bucket tiers are **not exercised** by Exp 4 — one more item
> for the scope-limits list, alongside S2A/S2B and beacons.
>
> ### ✅ **MAX-AoI is IMPLEMENTED — arm `B1`, 2026-08-13.**
> `hermes/scheduler/policies/max_aoi.py`, exposed through the existing
> `target_selector` slot; `--arms B1`. **No frozen file touched.** 17 policy tests;
> **651 unit tests pass**. Verified end to end: `contact_policy='max_aoi'` reaches the mule
> process (H1 records `None`), and the two arms produce **genuinely different visit orders** on
> the same seed — so B1-vs-H1 isolates the ranking policy and nothing else.
>
> Design points worth keeping: a contact's age is its **stalest member** (max, not mean — a
> neglected device must not hide behind well-served neighbours in the same cluster); a
> never-served device is **infinitely stale**, which is both correct AoI semantics and the
> "explore the unvisited" behaviour; and the ordering is **deterministic and independent of input
> order**, so the paired comparison is reproducible. A test pins that it ranks on age and **not**
> on `deadline_ts` — otherwise it would be our own policy wearing a baseline's name.
>
> **Still to do for B1:** decide whether it is paired with H1 (same backhaul model ⇒ yes) and add
> it to §5.1's arm list and cost.

### 5.1 The matrix — WRITTEN AND COSTED ✅ *(2026-08-13)*

**Costed by [`experiments/exp4/cost_matrix.py`](../experiments/exp4/cost_matrix.py)**, not by
arithmetic in prose — re-run it whenever an input changes.

#### The design principle that shapes it

**Freeze D5: `dead_zone` and `link_quality` are consumed only in the H0 branch.** Sweeping them
across mule arms is one configuration under different seeds — the exact error already made once in
the H2/H3 dead-zone sweep. So the surface is swept **for H0-vs-H1 and nowhere else**. That single
decision is what keeps this matrix at 680 trials instead of several thousand.

#### Three sweeps, because three comparisons are not poolable

| # | Sweep | Arms | Cells | Seeds | Trials | Wall |
|---|---|---|---|---|---|---|
| **A** | Architecture surface | H0, H1 | 13 | 20 | 520 | **101 m** |
| **B** | Scheduling policy | H1, B1, B2 | 2 | 20 | 120 | **32 m** |
| **C** | L1 adaptivity | H2, H3 | 1 | 20 | 40 | **13 m** |
| | | | | | **680** | **≈2.4 h** |

Upper bound if every arm ran at its sweep's slowest arm: **3.0 h**. Traces: **6.4 MB**.

* **A** — the headline participation claim. 13 cells = clean (1; `dead_zone`/`link_quality` do not
  apply) + jittery (4 × 3). This is the committed design **re-run with enforcement ON**.
* **B** — the reviewer-facing baseline comparison, at **one fixed** `(dead_zone, link_quality)`
  point, clean and jittery. One flag differs between arms, so it isolates the ranking policy.
* **C** — the L1 claim. Separate because `--l1-channel` changes the backhaul model in both arms.

#### Fixed parameters, with reasons

| Parameter | Value | Why |
|---|---|---|
| `--mission-budget-s` | **120 s** | Decided ON (§5.1a-A). 120 s is the "slack" point where the enforcement cost was *measured* (−0.225 completion), so the headline is priced against a figure we already have rather than a fresh unknown. The S3b probe puts the deadline floor at ~34 % of contacts at *any* budget and the knee near 60 s, so 120 s binds without being punitive. |
| `--mission-window-adaptation` | **off** | Pinned, not swept (§5.0a). The pilot effect was narrow, did not survive multiplicity correction, and is **transient** — it interacts with mission count, not with these axes. Reported as a separate finding. |
| `--keep-event-traces` | **on** | Without it no future baseline can be scored against this matrix (§4), and the next comparator costs a third full re-run. |
| `N`, `rrf`, `n_missions` | 6, 60 m, 4 | Unchanged from the committed design, so A is a like-for-like re-run and the enforcement delta is attributable to enforcement. |
| Seeds | **20** paired | Pairing key `(cell_id, trial_index)`. |
| Concurrency | **3** | At 5, the box exhausted memory and ~30 % of trials failed. |

#### Valid comparisons — and the one that is not

* ✅ **H1 vs H0** (A) — the architecture claim.
* ✅ **B1 vs H1**, **B2 vs H1** (B) — the policy claim; same transport, realism and seeds.
* ✅ **H3 vs H2** (C) — the L1 claim.
* ❌ **H2/H3 vs H0/H1** — different backhaul model *and* non-aligned seeds. Not paired, never pool.
* ❌ **B1/B2 vs H0** — the baselines are mule arms; comparing them to flat FL confounds policy with
  architecture.

#### Statistics and provenance

Paired Wilcoxon + Cliff's δ + percentile bootstrap CI, via
[`experiments/analysis/stats.py`](../experiments/analysis/stats.py) — one implementation, so the
claim rule means the same thing everywhere. **Claim only when the CI excludes 0 AND p < 0.05.**
With several metrics per comparison, **state the multiplicity position explicitly** rather than
reporting the one that hit (the §5.0a pilot is the worked example of why). Every row carries
`mission_budget_s` and `mission_window_adaptation`, so the file is self-describing.

### 5.1b Sweep B is ON HOLD — the policy comparison is vacuous as designed

**Caught by the pre-launch smoke, before spending the 32 minutes.** At the designed operating point
(N=6, `rrf`=60, budget 120 s) **H1, B1, B2, H2 and H3 all produce byte-identical `final_auc` and
`update_yield`.** Not similar — identical. A budget ladder confirms it is not a budget-value
problem:

| budget | H1 yield | B1 yield | H1 AUC | B1 AUC | |
|---|---|---|---|---|---|
| 120 s | 1.25 | 1.25 | 0.92506 | 0.92506 | identical |
| 60 s | 1.25 | 1.25 | 0.92506 | 0.92506 | identical |
| 30 s | 0.75 | 0.75 | 0.41002 | 0.41002 | identical |
| 15 s | 0.75 | 0.75 | 0.68892 | 0.68892 | identical |

The budget *is* binding — yield falls from 1.25 to 0.75 — and the arms are still identical.

**Why, structurally.** **S3b decides _who_ is served; the policy only decides the _order_ of an
already-decided set.** S3b runs *before* the bucket walk (deliberately — so a learned selector
cannot resurrect what it drops), and the mule then visits every contact remaining in its queue. If
the whole queue gets served, ordering changes nothing that any metric records. Ordering can only
change outcomes when the queue is **truncated in flight** — i.e. when the A1 abort fires — or when
contacts fail stochastically part-way.

**There is a binding band, and it is narrow.** Probing wider/tighter configurations:

| N | field radius | budget | Result |
|---|---|---|---|
| 6 | 100–150 m | 15–120 s | identical — mule serves its whole queue |
| 12 | 300 m | 45 s | **DIFFER** — B1 yield 0.75 vs H1 0.25; completion 0.167 vs 0.083 |
| 12 | 300 m | 25 s | identical — both collapse |
| 16 | 400 m | 30 s | identical — coverage 0.0 for both |

So the comparison is only non-vacuous in a band between "serves everyone" and "serves nobody".

**This is a finding, not just an obstacle.** *Scheduling policy only matters when the mule can serve
some but not all of its queue* — and that is worth stating in the paper, because it bounds the
claim. A baseline comparison run outside that band would have produced a tie and been reported as
"our policy is no better", which would have been **false for the wrong reason**.

**Recommended redesign — sweep the budget rather than fixing it.** Make the arc itself the result:
`N=12`, `field_radius=300`, budgets `{120, 60, 45, 30}` × {H1, B1, B2} × 20 seeds = **240 trials,
≈50 min**. That reports *where* policy matters instead of asserting it at one point, and it
subsumes the tie as evidence rather than hiding it.

> **Decision required before running B.** The binding band sits at ~8–17 % mission completion — a
> stressed regime. Reporting "our policy wins where the system is heavily degraded" is honest but
> narrow, and that trade-off is a judgement call about what the paper claims, not a technical one.

#### A bug the costing caught — worth carrying as a lesson

Calibrating B2 first showed it at **0.45× H1**, which would have looked like good news. It was not
a routing effect: **B2 was dying at mission 2** and was cheap because it was doing less
(`missions_completed` 1.0 vs 4.0). Two defects, both found only because the calibration checked
*what the arms accomplished*, not just how long they took:

1. **The ranking signal lagged a full round.** `FLReadyAdv` is built *before* the session trains, so
   its loss describes the **previous** round. Fixed by carrying the loss on `GradientSubmission`
   (which already carried `num_examples`), so it arrives **in-session** with the update it
   summarises. Without this the baseline would have been silently handicapped — ranking on
   round-old information — and it would have lost for the wrong reason.
2. **The stub guard was over-eager.** It keyed on *contacts*, so a device contacted twice whose
   sessions both failed — legitimately lossless — tripped it and killed the mission. Now keyed on
   `on_time_count` (successful participations).

After the fix: H1 39.0 s, B1 38.6 s, B2 39.6 s, all 4/4 missions, 0 failures.

> **The transferable lesson:** a baseline that is unexpectedly *cheap* is a red flag, not a saving.
> Cost calibration must check what a trial accomplished, or a broken arm gets priced in as a fast
> one — and would then have gone into the matrix and lost, looking like a real result.

#### Before launching

- [x] Calibrate `B1`/`B2` against `H1` — done, all within 3 %.
- [ ] Re-run `cost_matrix.py` if any input changed.
- [ ] Firm up `H2`/`H3`, still the only **estimated** figures (scaled from no-budget runs by H1's
      enforcement ratio). A 2-trial calibration would settle them; sweep C is only 13 m, so the
      exposure is small.
- [ ] Start **new** CSVs — committed files cannot be resumed now that rows carry provenance (§1a).
- [ ] Free memory check before launch; cap concurrency at 3.

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

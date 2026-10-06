# SOTA baseline candidates — full-text verified

**Current state (6 Oct 2026): §0 below.** FeRRy's Exp 5 runs nine state-of-the-art arms (H0, D1–D5,
E3, O1, M1) and six aggregation rules, chosen in the build plan's Baselines section
([`FeRRy_Build_Plan.html`](FeRRy_Build_Plan.html)). Batch 1 has measured D1–D4 against FeRRy. **§§1–7
record the August 2026 selection for Exp 4, kept as written:** the arms named `B1` and `B2` there are
`D1` (MAX-AoI) and `D2` (Oort) now.

**Status of the August pass: VERIFIED for every candidate that affected the decision then.** The first pass was
abstract-level; this revision checked the load-bearing rules against the papers themselves. **It
overturned the central conclusion.** Candidates that do not affect the choice are marked
⚠ *still abstract-level* and must not be cited without reading them.

---

## 0. Current state — FeRRy / Exp 5 (6 Oct 2026)

### 0.1 Three families of arms

The build plan's rule:
- **SOTA arms** reproduce a published method as faithfully as a mule allows.
- **FeRRy arms** are the system with one mechanism switched off, so each contribution (C1–C5) has a test that removes it.
- **Aggregation arms** vary the merge (L3) with the route held fixed.

**Attribution rule:** scheduler arms vary the route with the merge fixed at `agg:cutoff`, and aggregation arms vary the merge with the route fixed, never both at once. D4 runs twice, with its own merge and with FeRRy's, so that its route and its merge are judged separately.

### 0.2 The SOTA arms

**Verification column:** the August pass (§2, §6) checked FedCS, Oort and Power-of-Choice against full text. The newer arms' rules are as the build plan describes them, taken from the novelty audit of 23 Sep 2026 (not in this repository); **this document does not record a full-text check of them.** Read each before citing it.

| Arm | Method and source | What it decides | The port, and its deviations | What it isolates against FeRRy | Built / run | Verified |
|---|---|---|---|---|---|---|
| **H0** | Synchronous FedAvg over a live link, every client every round | Nothing | A `dead_zone` share of clients is unreachable under jitter; the rest succeed with probability rel × link quality | Whether muling pays when links fail (under jitter its AUC fell from 0.943 to 0.587 while H1 held 0.922–0.933) | built; Study 5.3 in batch 2, on the wall clock only | n/a |
| **D1** | MAX-AoI greedy, the standard AoI comparator (was `B1`) | Who is served, and in what order | Oldest contact first, admitted through the budget walk; age runs from the last CLEAN (Phase 0 fix) | Value- and deadline-blind fairness | built; **run in batch 1** | named comparator (§6) |
| **D2** | Oort's statistical utility, Lai et al., OSDI 2021 (was `B2`) | Who is served, and in what order | \|B_i\|·\|loss\| plus the staleness bonus; no system-speed term; mean, not RMS, loss; L(i) is the last CLEAN round. The utility is not normalised before the bonus, and the bonus is 0.1·log R/√L rather than √(0.1·log R/L), so it is about 1e-4 of the utility | A learned-utility selector with an exploration bonus, the direct rival to window widening | built; **run in batch 1** | ✅ full text (§2) |
| **D3** | Whittle index on weighted AoU, Cui et al., IEEE TMC 23(6), 2024 | Who is served | Closed-form index over (age, connected); ω_i from Oort's utility instead of a Shapley estimate; ρ_i from reachability history | The fairness reference: optimal for value-weighted age when selection is free. FeRRy should match it while paying for travel | built; **run in batch 1** | ⚠ not recorded |
| **D4** | FedEx-Async with CARP, Bian, Shen, Chen, Xu, IEEE TMC 24(6), 2025 | Device-to-mule assignment and each mule's tour | Gibbs assignment, a 2-OPT visit-all tour, 1/N accumulation applied on return; no gates, band, deadline or re-plan. Runs as `D4fedex` (its own merge, `agg:fedex`) and `D4` (route-only, `agg:cutoff`) | **The closest prior:** a travel-only bound, fixed reach, never skips. Report how often its tour overruns the budget | built; **run in batch 1** (both forms) | ⚠ not recorded |
| **D5** | FedCS, degraded, Nishio and Yonetani, ICC 2019 | Who joins this round | The largest set that fits one round deadline, with last-known state in place of the Resource Request; labelled degraded. Reads device training time once it exists (5.12) | Admission by one round deadline against per-device deadlines | built; Study 5.12 in batch 3 | ✅ full text (§2) |
| **E3** | DQN over the contact graph, after Chen et al., IEEE GLOBECOM Workshops 2023 | The next stop, at each arrival | Single agent; Chen's observation design; feasibility mask; bytes reward. **Never labelled FedQMIX**; scored on update yield and round closure, never bytes. As built: a numpy port trained in FerrySim, a whole scheduler picking each next stop among those S3b's budget admits; deviations: stops, not grid moves; one agent; no model-aided learning; the pair learner's masked double DQN | A learned per-step policy with none of FeRRy's deadline, plan-time or coverage machinery | built; trains in `rl-e3` (6 Oct); stack trials in batch 2 | ⚠ not recorded |
| **O1** | Offline oracle, after Zhai, Yuan, Wang, Yang, IEEE TWC 24(3), 2025 | Everything, offline | Exhaustive band class × clustering × route × per-stop band for N ≤ 6 under visit-within-S | The optimality gap for Study 5.4 | built; Study 5.4 in batch 2 | ⚠ not recorded |
| **M1** | Monolithic agent, after Ho et al., IAAA 2025 | (band, stop), with no gates | One DDQN over the full product, with a positions-plus-phase state; no plan, no mask | Whether the gates earn their keep | **not built**: only if Study 5.5 keeps a learned score | ⚠ not recorded |
| cite only | Mestoukirdi et al., GLOBECOM 2022; Power-of-Choice `pow-d`, `rpow-d`; Yang et al., JSAC 2025 (contextual bandit) | — | Mestoukirdi optimises scheduling and trajectory jointly, so it cannot be ported into a fixed router; `pow-d` needs a loss query before selection. `rpow-d` and Yang's bandit are optional selection-only arms if a reviewer asks | — | citation | Mestoukirdi and Power-of-Choice ✅ (§6) |

### 0.3 The aggregation arms (Study 5.1, batch 2)

| Rule | Merge | Source | Role |
|---|---|---|---|
| `agg:plain` | `num_examples`-weighted mean | HERMES | The baseline every rule must beat |
| `agg:cutoff` | n_i·v_i·s(age_i), zero past a_max; FedProx on devices | FedAsync's hinge (Xie et al., 2019) with Yang et al.'s cutoff (JSAC 2025); Shen et al.'s proximal term (IoTJ 2024) | FeRRy's rule |
| `agg:fedbuff` | Apply after K updates arrive, K = slice size | Nguyen et al., AISTATS 2022 | Count-triggered against deadline-triggered |
| `agg:asynchfl` | Exponential staleness decay at the mule and at the cluster, plus a proximal term | Yu et al., Async-HFL, IoTDI 2023 | A hierarchical-decay reference |
| `agg:seq` | Carry the model device to device; the route is the SGD sequence | After Cui et al., TMC 2024 | **Decided out** (it needs a protocol change) |
| `agg:fedex` | 1/N accumulation, applied when the mule returns | FedEx-Async | D4's faithful merge |

### 0.4 What batch 1 measured (6 Oct 2026)

Time to τ = 0.71 in simulated seconds, lower is better. Twenty paired trials per cell; Holm within each study. The scores are in [`results/exp5/scores/b1/`](../results/exp5/scores/b1/index.md); the arm in bold is the faster one where the difference is a claim.

| Cell | F (FeRRy) | H1 | D1 | D2 | D3 | D4 (FedEx route) | D4fedex (FedEx route and merge) |
|---|---|---|---|---|---|---|---|
| N = 6, 1 mule, knee | **194** | 282 | 282 | 266 | 280 | 264 | 281 (n.s.) |
| N = 6, 1 mule, stress | **178** against D4 | 220 (n.s.) | 238 (n.s.) | 229 (n.s.) | 209 (n.s.) | 264 | 281 (n.s.) |
| N = 6, 3 mules (both budgets) | **52** against D4fedex | 64–65 (n.s.) | 64–65 (n.s.) | 64–65 (n.s.) | 64–65 (n.s.) | 55 (n.s.) | 101 |
| N = 12, 1 mule, knee | 486 | 589 (n.s.) | — | — | 526 (n.s.) | 459 (n.s.) | — |
| N = 24, 1 mule, knee | **749** against H1, D3 | 951 | — | — | 911 | 752 (n.s.) | — |
| N = 12, 2 mules | 226 | 263 (n.s.) | — | — | — | 222 (n.s.) | — |
| N = 12, 3 mules | 151 (FX 102, a claim for FX) | 165 (n.s.) | — | — | — | 128 (n.s.) | — |
| N = 18, 3 mules | 186 | 149 (n.s.) | — | — | — | 186 (n.s.) | — |

**How to read it:**
- **FeRRy beats every SOTA arm only at N = 6 with one mule at the knee budget**, by 26–31%.
- **At N = 12 and 24, D4 (FedEx's route with FeRRy's merge) ties F.** F still beats H1 and D3 at N = 24, by 18–21%.
- **Under the stress budget, only D4 is beaten.**
- **With three mules, nothing separates,** except FedEx's own merge (D4fedex), which loses to F. The budget never binds there, since each mule serves two devices.
- **So FedEx's route is competitive as N grows, and its merge is what costs it.** That is the attribution the D4/D4fedex pair was built for. Write it as found.
- **Not run yet:**
  - H0, E3 and O1 are in batch 2.
  - D5 is in batch 3.
  - The aggregation arms are Study 5.1, in batch 2.
  - M1 is built only if 5.5 keeps a learned score; the calibration read flat ([findings](Experiment_5_RL_Calibration_Findings.md)).

**Bottom line up front:** the first pass recommended **FedCS** and dismissed **Oort** as
un-implementable. The full text says the opposite. FedCS polls clients before selecting; Oort is
retrospective and runs on exactly the state a data mule has.

> **Writing the paper, not choosing arms?** See
> [`HERMES_Related_Work_Notes.md`](HERMES_Related_Work_Notes.md). This document decides *which
> baselines we run*; that one holds *what we say about the literature* — including papers worth
> discussing that are unusable as arms.

---

## 1. What we actually need

Reviewer 74A's complaint is the absence of recent UAV-FL baselines. The bar for a *usable* baseline
here is narrower than "a relevant paper":

1. Its scheduling rule must reduce to **an algorithm we can re-implement** in our harness (same
   mobility, seeds, budgets, metrics) — comparing against a paper's *reported numbers* on a
   different setup proves nothing.
2. It must only need **state a data mule can actually have.** Our mule sees a device *only at a
   contact*. A baseline needing per-round pre-selection reporting is not implementable here — and
   saying so is itself a defensible contribution point, **provided we say it about the right
   papers**.
3. Ideally **scoreable retroactively** against committed per-trial data (positions, per-contact
   success, deadlines, updates/round) — no re-run. See the checklist's re-run ledger.

## 2. The correction — what full text changed

| Candidate | First-pass reading | What the paper actually says | Verdict |
|---|---|---|---|
| **FedCS** | "**Yes, closest analogue**" — recommended | Protocol 2 has an explicit **Resource Request** step *before* Client Selection: "Clients who receive the request notify the operator of their resource information" — wireless channel state, computational capacity, and relevant data size, **every round, before selection** | **INVERTED** — needs pre-selection polling |
| **Power-of-Choice** `pow-d` | "Weak — requires querying loss before selecting" | **Confirmed.** The server sends the global model to the candidate set and those clients "compute and send back to the central server their local loss" before selection | **Confirmed** |
| **Power-of-Choice** `rpow-d` | *not identified* | A published variant that avoids the query entirely: clients send accumulated averaged loss **when they participate**, and "the server uses the latest received value from each client as a proxy" | **NEW — implementable** |
| **Oort** | "Weak — needs per-round global utility" | Retrospective by design: "a client's utility can only be determined **after it has participated** in training." Utilities are cached from prior participation, plus an explicit staleness term | **INVERTED — implementable** |
| **MAB freshness/energy** (Sensors 2024) | "Strong retroactive candidate" | **Confirmed** as UCB with reward `μ̄ = α·Ē + (1−α)·FM`, freshness `FM(m,t) = t − a·C_m`, no advance polling. **But** its UAV is a *hovering base station* serving all devices each round, not a mule making discrete contacts | **Confirmed rule, fidelity caveat** |
| **Vehicular reputation** (CJA 2024) | "Plausible retroactive scoring" | Heavier than described: reputation over data quality **and** computation capability, a **consortium blockchain** maintaining reputation, and an asynchronous-parallel RL resource scheduler | **Weaker** — faithful re-implementation is out of scope |
| **MAX-AoI greedy** (Tier C) | "arguably the single best fit" | **Confirmed as a standard named comparator** — evaluations report against "random, round-robin, periodic update, and MAX-AoI"; the greedy form picks highest-AoI and recursively finds the nearest predecessor for the path | **Confirmed** |

**Why this matters beyond bookkeeping.** Had we published the first pass's framing, Related Work
would have dismissed Oort — one of the most-cited selection papers in FL — on a property it does not
have. Any reviewer who knows the paper would read that as not having read it, which is precisely
the criticism 74A already made.

## 2a. The candidate the first pass missed — closest architectural match found

**UAV-Aided Multi-Community Federated Learning** (Mestoukirdi, Esrafilian, Gesbert, Li — GLOBECOM
2022, [arXiv:2206.02043](https://arxiv.org/abs/2206.02043)). Surfaced by a general UAV-FL sweep, not
by the original scan. Full-text verified.

**Why it matters more than anything in the original list:** its UAV is **not** a hovering base
station. It flies a trajectory of **discrete stops** and devices transmit only when it is nearby —
the same physical model as our mule. Every Tier A/B candidate either assumes persistent
connectivity or is architecture-agnostic; this one shares our problem shape.

Its device-importance metric is also a direct structural analogue of ours:

```
ψ_c = CoV of validation accuracy across community c        (heterogeneity proxy)
δ_k = p_k · ψ_c · λ   if the device failed or was NOT scheduled last round
      p_k · ψ_c       otherwise
```

That **λ multiplier for devices that failed or went unscheduled** is prior art for the exact
problem Freeze Amendment 1 (A2) addresses — a device the scheduler skipped must be made more
attractive next time, or it starves. They solve it with an importance multiplier; we solve it with
window widening plus mission-level adaptation. **That contrast belongs in Related Work**, and it is
a much better citation than "no one has considered starvation".

**But it is not usable as our baseline arm**, for one specific reason: trajectory and scheduling are
**jointly optimised** (alternating sub-problems, greedy graph-based trajectory initialisation).
Re-implementing the scheduling rule against our fixed S3a/S3.5 routing would not be faithful to the
design — it would be their metric inside our router, which is neither their system nor a clean
comparison. The state it needs is otherwise mule-compatible: validation accuracies arrive
*alongside the models during uplink* (retrospective, refreshed every ℓ rounds), and locations,
dataset sizes and participation history are all known.

**Verdict: cite in Related Work as the closest architectural prior; do not implement as an arm.**
If a reviewer asks for a UAV-specific comparator rather than a general-FL one, this is the paper to
discuss, and the joint-optimisation scope difference is the honest reason it is discussed rather
than run.

## 3. The capability argument — narrower, and now correct

The first pass claimed Oort and Power-of-Choice "assume the server can *poll* clients before
choosing." That is **true of FedCS and of `pow-d`**, and **false of Oort and `rpow-d`**. The
defensible version:

> A data mule learns a device's state **only by flying to it**. Policies that require every
> candidate to *report* channel state, capacity, or current loss **before** the round's selection
> is made — FedCS's Resource Request step, Power-of-Choice's `pow-d` loss query — cannot run on a
> mule without being given information the architecture denies them. Policies whose ranking signal
> is **retrospective** — Oort's post-participation utility, Power-of-Choice's `rpow-d` stale-loss
> proxy, AoI/staleness — port directly, because "what I learned last time I visited you" is exactly
> what a mule has.

That is a sharper claim than the original: the obstacle is not *learned selection* or *utility
ranking*, it is specifically **pre-selection reporting**. It also lands better, because it says our
architecture is compatible with the strong baselines and incompatible only with a specific
assumption — which is a capability statement rather than an excuse.

**Consequence for FedCS:** still worth including, but as the *degraded* comparator, and labelled as
such. Its spirit — admit what fits the deadline — is what S3b does, so it remains the most
informative contrast; we simply have to substitute last-known state for the Resource Request and
say so in one sentence.

## 4. Recommendation (revised)

Pick **two**, with a third as the capability contrast:

1. **Oort** — highly cited, so its absence is conspicuous; faithfully implementable on mule-visible
   state; and its staleness term `Util(i) ← U(i) + 0.1·log(R)/√L(i)` is a **direct rival to our own
   Φ-widening / starvation mechanism** (Freeze Amendments 1–2). That makes it the sharpest
   available test of whether our L2 design earns its complexity.
2. **MAX-AoI / staleness-greedy** — "fly to the device whose update is oldest." Established as a
   named baseline in the AoI literature, needs only last-served time, trivially scoreable
   retroactively, and a genuine rival to bucket+deadline ordering.
3. **FedCS (degraded)** — as the explicit capability contrast, with the substitution stated.

Drop from consideration: the vehicular reputation paper (blockchain + RL, out of scope) and the
A3C joint-placement paper (trajectory optimisation, much larger scope). Keep the Sensors MAB as
optional — its rule is implementable and its freshness term is computable from our data, but its
hovering-base-station model is a different problem shape, which must be stated if used.

## 4a. Triage — the broad UAV-FL literature, and why most of it is not a baseline

A general UAV-FL sweep returns mostly work that is *relevant reading* but **fails the baseline bar**
for a structural reason, not a quality one. Recording the categories so this is not re-litigated:

| Category | Example work | Why it is not a baseline arm |
|---|---|---|
| **UAV as FL *client*** | drones training on their own aerial imagery / RF captures | Our UAV **carries** updates; it does not generate training data. Different role, so there is no scheduling rule to port. |
| **UAV as flying base station / relay** | UAV hovers to provide connectivity to ground IoT | Assumes the UAV **restores persistent connectivity** — which is the assumption our architecture exists to remove. Their scheduler cannot be run on a mule without inventing the link it presumes. |
| **Aggregation algorithms** | FedWT (MST-weighted tree aggregation), ClusterAvg, over-the-air aggregation | No scheduling component. Ours is an L3 question, and Exp 4 already uses two-pass hierarchical FedAvg. |
| **Byzantine / adversarial robustness** | UAV-assisted heterogeneous FL against Byzantine attacks | Orthogonal threat model. Worth citing if we make robustness claims; we do not. |
| **Joint trajectory + resource RL** | DRL/A3C trajectory + power + scheduling co-design | Optimises the flight path itself. Our trajectory follows from S3a clustering and the contact queue, so a faithful port would replace the system under test. |
| **Application-level UAV anomaly detection** | adaptive FL for UAV anomaly detection under non-IID | Closest to our *application* (IDS), useful for motivation and the non-IID framing, but contains no target-scheduling rule. |

**The single distinction that decides all of these:** does the UAV **restore connectivity**, or does
it **substitute for it**? Almost all UAV-FL work does the former — the drone is infrastructure that
flies. HERMES does the latter — the drone is transport. A scheduler written for the first case
assumes state that only exists in the first case. That is the capability argument in §3, and this
table is the evidence that it generalises beyond the three Tier A papers.

> **Scope honesty:** the rows above are triaged **by architecture class**, from abstracts and
> summaries — not full-text verified individually. That is sufficient to exclude them as *arms*,
> and insufficient to *cite* them. Any of these that ends up in Related Work must be read first.

## 5. Open questions before implementing

- [x] Full-text check every rule that affects the decision — **done, §2**.
- [x] **Can the chosen baselines be scored on the committed CSVs? — NO. Both need new trials.**
      Checked against the harness:

      * The committed CSVs carry only **aggregate** loss (`init_loss`, `final_loss`) — no
        per-device, per-contact value.
      * `RoundCloseDelta` *does* carry a per-device `utility`, but it is
        `w1·performance + w2·diversity`, **device-computed as an S2B readiness term** — not a
        training loss, and not Oort's statistical utility `|B_i|·√(mean Loss²)`. Reusing it would
        be a different algorithm wearing Oort's name.
      * More decisively: the per-contact record lives in the orchestrator's run-dir JSONL
        (`{cluster,mule,device}-*.jsonl`). `consume_run_dir` folds it into aggregates and the
        `finally: orch.cleanup()` in `driver.py` **deletes the trace at teardown**. Nothing
        per-contact survives a trial.

      So **retroactive scoring is impossible for any policy**, not just Oort — there is no trace to
      replay. MAX-AoI is affected identically: last-served time exists at runtime and is discarded.

- [x] **Done since:** `--keep-event-traces` keeps each trial's run-dir, and every Exp 5 stage passes it. The trace scorer re-parses them, and the traces are archived per stage (`exp5 pack`). *(The August text follows.)*
      ⚠ **Retain event traces BEFORE the matrix runs.** This is the load-bearing action. The
      driver already calls `shutdown_all(cleanup_tmpdir=False)` and only deletes in the `finally`,
      so preserving the JSONL is a small, opt-in change (`--keep-event-traces` → copy the run-dir
      alongside the CSV). **Without it we will pay for the entire matrix and still be unable to
      score any new baseline against it** — forcing a third full re-run the first time a reviewer
      asks for another comparator. With it, every future baseline is a re-parse.
- [x] Fairness statement: same mobility, seeds, budget, metrics — written down before running. **Done for Exp 5:** every arm of a cell flies the same seeds (sha256 of base seed, cell and trial, never the arm), the same budgets and the same metrics, all fixed in `scripts/exp5/params.toml` before each stage ran.
- [ ] ⚠ The three candidates still at abstract level (Data-Efficient Energy-Aware, A3C
      Privacy-Preserving, Contribution-Based) are **not** cleared for citation. They are not
      recommended, so this does not block Phase 2 — but do not cite them without reading them.

## 6. Verification log

Checked 2026-08-13 against full text, not abstracts:

| Source | How verified |
|---|---|
| Power-of-Choice (arXiv 2010.01243) | Full text; algorithm πpow-d steps, and the πcpow-d / πrpow-d variants |
| Oort (arXiv 2010.06081) | Full text; utility formula, Algorithm 1 line 17 staleness term, exploration of unselected clients |
| FedCS (arXiv 1804.08333) | Full text; Protocol 2 step order and Resource Request contents, Algorithm 3 greedy criterion |
| Fairness-Enhanced MAB (Sensors 2024) | Full text via PMC (MDPI returns 403); Eq. 12/14/15/16 |
| Vehicular reputation (CJA 2024) | Publisher abstract + indexed summaries — **enough to disqualify on scope**, not full text |
| MAX-AoI greedy | Confirmed as a named comparator across the AoI/UAV scheduling literature |
| UAV-Aided Multi-Community FL (GLOBECOM 2022) | Full text; connectivity model, CoV metric and the λ unscheduled-device penalty, joint trajectory/scheduling structure |
| Broad UAV-FL sweep (§4a) | Triaged **by architecture class** from abstracts — sufficient to exclude as arms, **not** sufficient to cite |

## 7. Sources

**Tier A — verified, citation-ready:**

- **FedCS** — Nishio & Yonetani, *Client Selection for Federated Learning with Heterogeneous
  Resources in Mobile Edge*. **Proc. IEEE ICC 2019** ·
  [doi:10.1109/ICC.2019.8761315](https://doi.org/10.1109/ICC.2019.8761315) ·
  [arXiv:1804.08333](https://arxiv.org/abs/1804.08333)
- **Oort** — Lai, Zhu, Madhyastha & Chowdhury, *Oort: Efficient Federated Learning via Guided
  Participant Selection*. **USENIX OSDI 2021** · [arXiv:2010.06081](https://arxiv.org/abs/2010.06081)
- **Power-of-Choice** — Cho, Wang & Joshi. Preprint
  [arXiv:2010.01243](https://arxiv.org/abs/2010.01243) *(Client Selection in Federated Learning:
  Convergence Analysis and Power-of-Choice Selection Strategies)*; published **retitled** as
  *Towards Understanding Biased Client Selection in Federated Learning*, **AISTATS 2022**, PMLR
  151:10351–10375 — [proceedings](https://proceedings.mlr.press/v151/jee-cho22a.html).
  ⚠ **Cite the AISTATS version for the venue, but confirm the `pow-d`/`cpow-d`/`rpow-d` naming
  survived into it** — our capability argument (§3) refers to `rpow-d` by name, so the citation must
  point somewhere that label appears.
- **UAV-Aided Multi-Community FL** — Mestoukirdi, Esrafilian, Gesbert & Li. **IEEE GLOBECOM 2022**
  (SAC Aerial Communications) · [arXiv:2206.02043](https://arxiv.org/abs/2206.02043)

**Added for FeRRy (Sep–Oct 2026), as the build plan's Sources list them.** ⚠ This document records no full-text check of these; read each before citing it.

- **FedEx** — Bian, Shen, Chen, Xu, *Indirect-Communication Federated Learning via Mobile Transporters*. **IEEE TMC 24(6), 2025.** The closest prior (arm D4).
- **Cui et al.** — Cui, Yang, Wu, Feng, Hu, *Data Value Based Asynchronous FL for UAV Swarm Under Unstable Communication*. **IEEE TMC 23(6), 2024.** The Whittle index (arm D3) and the sequential merge (`agg:seq`, decided out).
- **Chen et al.** — Chen, Esrafilian, Bayerlein, Gesbert, Caccamo, *Model-Aided Federated RL for Multi-UAV Trajectory Planning in IoT Networks*. **IEEE GLOBECOM Workshops 2023.** The source of arm E3; never label E3 "FedQMIX".
- **Zhai et al.** — Zhai, Yuan, Wang, Yang, *UAV-Enabled Asynchronous FL*. **IEEE TWC 24(3), 2025.** The coverage term and the offline oracle (arm O1).
- **Ho et al.** — *Energy-Efficient DDPG-Based UAV-Assisted Asynchronous FL*. **IAAA 2025.** The monolithic agent (arm M1, built only if 5.5 keeps a learned score).
- **FedAsync** — Xie et al., 2019. The staleness hinge in `agg:cutoff`.
- **FedBuff** — Nguyen et al., **AISTATS 2022**. `agg:fedbuff`.
- **Async-HFL** — Yu et al., **IoTDI 2023**. `agg:asynchfl`.
- **Yang et al.** — **IEEE JSAC 2025**: the age cutoff in `agg:cutoff`, and a contextual-bandit selector (cite only).
- **Shen et al.** — **IEEE IoTJ 2024**: the proximal term in `agg:cutoff`.
- **Chen et al.** — **IEEE TVT 2025**: the theory track's citation for mobility as mixing. The full reference is not held in the repository.

**Other sources:**

- [A Fairness-Enhanced Federated Learning Scheduling Mechanism for UAV-Assisted Emergency Communication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934714/)
- [Client selection and resource scheduling in reliable federated learning for UAV-assisted vehicular networks](https://www.sciencedirect.com/science/article/pii/S100093612400236X)
- [Reliability- and Connectivity-Constrained Age-of-Information Optimization for UAV Swarm IoT Data Collection](https://arxiv.org/abs/2608.00061)
- [Entropy-Based Age-Aware Scheduling Strategy for UAV-Assisted IoT Data Transmission](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192429/)
- [Age of Information minimization in UAV-aided data collection for WSN and IoT applications: a systematic review](https://www.sciencedirect.com/science/article/abs/pii/S1084804523000711)
- [Data-Efficient Energy-Aware Participant Selection for UAV-Enabled Federated Learning](https://www.researchgate.net/publication/373116833_Data-Efficient_Energy-Aware_Participant_Selection_for_UAV-Enabled_Federated_Learning) ⚠ abstract-level
- [Privacy-Preserving Federated Learning for UAV-Enabled Networks](https://www.researchgate.net/publication/346510744_Privacy-Preserving_Federated_Learning_for_UAV-Enabled_Networks_Learning-Based_Joint_Scheduling_and_Resource_Management) ⚠ abstract-level
- [Contribution-Based Resource Allocation for Effective Federated Learning in UAV-Assisted Edge Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11511571/) ⚠ abstract-level

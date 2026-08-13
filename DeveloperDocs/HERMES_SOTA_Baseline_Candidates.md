# SOTA baseline candidates — full-text verified

**Status: VERIFIED for every candidate that affects the decision.** The first pass was
abstract-level; this revision checked the load-bearing rules against the papers themselves. **It
overturned the central conclusion.** Candidates that do not affect the choice are marked
⚠ *still abstract-level* and must not be cited without reading them.

**Bottom line up front:** the first pass recommended **FedCS** and dismissed **Oort** as
un-implementable. The full text says the opposite. FedCS polls clients before selecting; Oort is
retrospective and runs on exactly the state a data mule has.

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

- [ ] ⚠ **Retain event traces BEFORE the matrix runs.** This is the load-bearing action. The
      driver already calls `shutdown_all(cleanup_tmpdir=False)` and only deletes in the `finally`,
      so preserving the JSONL is a small, opt-in change (`--keep-event-traces` → copy the run-dir
      alongside the CSV). **Without it we will pay for the entire matrix and still be unable to
      score any new baseline against it** — forcing a third full re-run the first time a reviewer
      asks for another comparator. With it, every future baseline is a re-parse.
- [ ] Fairness statement: same mobility, seeds, budget, metrics — written down before running.
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

## 7. Sources

- [Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies](https://arxiv.org/abs/2010.01243)
- [Oort: Efficient Federated Learning via Guided Participant Selection](https://arxiv.org/abs/2010.06081)
- [Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge (FedCS)](https://arxiv.org/abs/1804.08333)
- [A Fairness-Enhanced Federated Learning Scheduling Mechanism for UAV-Assisted Emergency Communication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934714/)
- [Client selection and resource scheduling in reliable federated learning for UAV-assisted vehicular networks](https://www.sciencedirect.com/science/article/pii/S100093612400236X)
- [Reliability- and Connectivity-Constrained Age-of-Information Optimization for UAV Swarm IoT Data Collection](https://arxiv.org/abs/2608.00061)
- [Entropy-Based Age-Aware Scheduling Strategy for UAV-Assisted IoT Data Transmission](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192429/)
- [Age of Information minimization in UAV-aided data collection for WSN and IoT applications: a systematic review](https://www.sciencedirect.com/science/article/abs/pii/S1084804523000711)
- [Data-Efficient Energy-Aware Participant Selection for UAV-Enabled Federated Learning](https://www.researchgate.net/publication/373116833_Data-Efficient_Energy-Aware_Participant_Selection_for_UAV-Enabled_Federated_Learning) ⚠ abstract-level
- [Privacy-Preserving Federated Learning for UAV-Enabled Networks](https://www.researchgate.net/publication/346510744_Privacy-Preserving_Federated_Learning_for_UAV-Enabled_Networks_Learning-Based_Joint_Scheduling_and_Resource_Management) ⚠ abstract-level
- [Contribution-Based Resource Allocation for Effective Federated Learning in UAV-Assisted Edge Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11511571/) ⚠ abstract-level

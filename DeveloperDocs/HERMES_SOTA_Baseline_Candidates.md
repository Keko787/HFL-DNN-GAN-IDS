# SOTA baseline candidates — first pass

**Status: FIRST PASS, not the final selection.** This is a scoped literature scan to give Phase 2
of the [pre-re-run checklist](HERMES_PreRerun_Checklist.md) something concrete to argue with. It has
**not** been through full-text verification — every claim below is from abstracts/summaries and must
be checked against the paper before anything is implemented or cited.

> **Run `/deep-research` for the exhaustive version.** A ready-to-paste prompt is in §5.

---

## 1. What we actually need

Reviewer 74A's complaint is the absence of recent UAV-FL baselines. The bar for a *usable* baseline
here is narrower than "a relevant paper":

1. Its scheduling rule must reduce to **an algorithm we can re-implement** in our harness (same
   mobility, seeds, budgets, metrics) — comparing against a paper's *reported numbers* on a
   different setup proves nothing.
2. It must only need **state a data mule can actually have.** Most UAV-FL work assumes continuous
   connectivity and per-round global knowledge (all clients' loss, channel state, energy). Our mule
   sees a device *only at a contact*. A baseline needing per-round global state is not
   implementable here — and saying so is itself a defensible contribution point.
3. Ideally **scoreable retroactively** against committed per-trial data (positions, per-contact
   success, deadlines, updates/round) — no re-run. See the checklist's re-run ledger.

## 2. Candidates

### Tier A — general FL client selection (reviewers expect these)

These are the standard comparators in the FL-selection literature. They are *not* UAV-specific,
which is both their weakness and the reason their absence is conspicuous.

| Baseline | Decision rule (as understood) | State needed | Data-mule viable? |
|---|---|---|---|
| **FedCS** | Greedily admit clients whose compute+comms fit within a round deadline; maximise clients per round | Per-client compute/comm capability, round deadline | **Yes, closest analogue** — this is essentially deadline-feasibility selection, i.e. a direct comparator for our S1+S3b gates |
| **Power-of-Choice** (Cho et al., arXiv 2010.01243) | Sample a candidate pool, query local training loss, select highest-loss clients | Per-client **local loss on demand** | **Weak** — requires querying loss before selecting; a mule learns loss only after flying there |
| **Oort** | Rank by statistical utility (loss × data size) blended with system speed; mostly exploit, some explore | Per-client loss + speed + bandwidth, per round | **Weak** — same objection, needs per-round global utility |

> **The honest framing:** Power-of-Choice and Oort assume the server can *poll* clients before
> choosing. In a DDIL data-mule setting that assumption fails — which is exactly the capability gap
> the paper argues. Implementing them faithfully means giving them information our system cannot
> have; implementing them fairly means degrading them to what a mule can see. **Say which you did.**

### Tier B — UAV-specific FL scheduling

| Paper | Rule (as understood) | Notes for us |
|---|---|---|
| **Client selection + resource scheduling for UAV-assisted vehicular FL** (2024) | Reputation-based selection combining data quality + compute capability; RL-optimised scheduling | Reputation ≈ our contact-reliability history. Plausible retroactive scoring if reputation can be computed from recorded per-contact success. |
| **Fairness-Enhanced FL Scheduling for UAV-Assisted Emergency Communication** (Sensors, 2024) | **Multi-Armed Bandit** selection weighting model freshness against energy | MAB over contacts is directly implementable in our bucket walk, and fairness/staleness is a metric we already record (Jain, entropy, participation). **Strong retroactive candidate.** |
| **Data-Efficient Energy-Aware Participant Selection for UAV-Enabled FL** (2023) | Prioritise fast, high-reliability participants; exclude malicious ones | Reliability prioritisation maps onto our `rel_i`. Energy term needs our unresolved `ε_prop`. |
| **Privacy-Preserving FL for UAV-Enabled Networks** (2020) | A3C joint device selection + UAV placement + resource management | Joint trajectory optimisation — a much larger scope than our gated selector; likely out of scope to re-implement faithfully. |

### Tier C — Age-of-Information / freshness

AoI-aware UAV data collection is the closest *problem shape* to a data mule: which stale node to
visit next, under travel cost.

* **AoI-minimisation in UAV-aided data collection** — an established review area, so an
  AoI-greedy baseline is defensible and easy to justify.
* **Topology-coupled urgency scheduling for UAV swarm IoT collection** (2026) — weights each
  cluster's AoI urgency by a connectivity score.

> **A staleness-greedy baseline is arguably the single best fit**: "always fly to the device whose
> update is oldest" is a strong, intuitive, *implementable* policy that needs only what our mule
> already knows (last-served time), and it is a genuine rival to deadline ordering. It is also
> trivially **retroactively scoreable**.

## 3. Recommendation for Phase 2

Pick **two**, not six:

1. **FedCS-style deadline-feasibility selection** — the closest analogue to our own gates, and it
   forces us to state what our scheduler adds beyond "admit what fits".
2. **AoI/staleness-greedy** — simple, strong, implementable on mule-visible state, and it directly
   challenges the bucket+deadline ordering.

Optionally a third, **MAB freshness-vs-energy** (Tier B), if the fairness angle is worth the space.

Then state plainly, with the reasoning above, **why Oort / Power-of-Choice are not implementable
without giving them information a data mule cannot obtain** — that is a capability argument, not an
evasion, and it belongs in Related Work.

## 4. Open questions before implementing

- [ ] Full-text check every rule above — these are abstract-level readings.
- [ ] For each chosen baseline: can it be scored on the **committed** CSVs, or does it need new
      trials? (Checklist §1 ledger.)
- [ ] Does the harness expose the state each baseline needs (last-served time, per-contact history)?
- [ ] Fairness statement: same mobility, seeds, budget, metrics — written down before running.

## 5. Prompt for `/deep-research`

```
Which recent (2019–2026) UAV-assisted federated learning papers propose client/target scheduling
or selection policies that could serve as a re-implementable state-of-the-art baseline for a UAV
data-mule FL system? Context: a UAV physically flies to edge devices, collects model updates over
short-range RF at contact stops, and carries them back to a base station; federation continues
without persistent end-to-end connectivity. The scheduler is a gated pipeline: hard gates
(eligibility, deadline feasibility) decide which contacts are legal, then a bounded learned
selector only reorders within an already-admitted set.

For EACH candidate: (1) the scheduling decision rule reduced to an algorithm — what it optimises
and how it ranks/selects; (2) what state it requires (channel state, data quality, energy,
AoI/staleness, position, deadline); (3) whether that state is obtainable in a data-mule setting or
assumes continuous connectivity; (4) whether the policy could be scored RETROACTIVELY against
recorded per-trial data (device positions, per-contact success/failure, deadlines, updates per
round) or needs new trials; (5) how commonly it is cited as a comparison baseline.

Prioritise: resource-aware / utility-guided FL client selection; UAV trajectory + resource
allocation for FL; Age-of-Information / freshness-aware UAV-FL scheduling; multi-community or
multi-UAV aided FL; mobile-relay / semi-decentralized FL. Exclude pure FL-aggregation papers with
no scheduling component. Flag any baseline standard enough that reviewers would expect to see it.
```

## 6. Sources (first pass)

- [Client selection and resource scheduling in reliable federated learning for UAV-assisted vehicular networks](https://www.sciencedirect.com/science/article/pii/S100093612400236X)
- [A Fairness-Enhanced Federated Learning Scheduling Mechanism for UAV-Assisted Emergency Communication](https://www.mdpi.com/1424-8220/24/5/1599)
- [Data-Efficient Energy-Aware Participant Selection for UAV-Enabled Federated Learning](https://www.researchgate.net/publication/373116833_Data-Efficient_Energy-Aware_Participant_Selection_for_UAV-Enabled_Federated_Learning)
- [Privacy-Preserving Federated Learning for UAV-Enabled Networks](https://www.researchgate.net/publication/346510744_Privacy-Preserving_Federated_Learning_for_UAV-Enabled_Networks_Learning-Based_Joint_Scheduling_and_Resource_Management)
- [Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies](https://arxiv.org/abs/2010.01243)
- [Contribution-Based Resource Allocation for Effective Federated Learning in UAV-Assisted Edge Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11511571/)
- [Age of Information minimization in UAV-aided data collection for WSN and IoT applications: a systematic review](https://www.sciencedirect.com/science/article/abs/pii/S1084804523000711)
- [Reliability- and Connectivity-Constrained Age-of-Information Optimization for UAV Swarm IoT Data Collection](https://arxiv.org/abs/2608.00061)
- [Delay and Overhead Efficient Transmission Scheduling for Federated Learning in UAV Swarms](https://arxiv.org/pdf/2405.00681)

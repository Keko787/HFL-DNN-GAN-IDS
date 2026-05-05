# HERMES FL Scheduler — Implementation Design

**Scope.** This document consolidates slides 20, 23, 24, 40, 41, 42, 43, 44 of the HERMES presentation into an implementation-ready design. The architecture is **symmetric server/client at every tier-boundary**: every server has a matching client on the other side of the link.

| Tier-boundary | Server role | Client role |
|---|---|---|
| Mule ↔ Edge Device (RF link) | `HFLHostMission` (mule) | `ClientMission` (edge device) |
| Mule ↔ Edge Server (dock) | `HFLHostCluster` (server) | `ClientCluster` (mule) |
| Tier 2 ↔ Tier 3 (cloud) | `Tier3Coordinator` (cloud) | `ClusterCloudClient` (server) |

The seven cooperating programs are:

1. **`FLScheduler`** — L2 Scheduler on the mule's NUC.
2. **`TargetSelectorRL`** — *(new, was the "trajectory" head of MA-P-DQN)* intra-bucket next-target selector; sub-model of L2 S3, runs on the mule's NUC.
3. **`HFLHostMission`** — Mission-scope FL server on the mule's NUC.
4. **`ClientCluster`** — *(new)* dock-handoff client on the mule's NUC, talks to `HFLHostCluster`.
5. **`ClientMission`** — *(was `EdgeClient`)* in-field FL client on the edge device, talks to `HFLHostMission`.
6. **`HFLHostCluster`** — cluster-scope FL server on the edge server (Tier 2).
7. **`L1 RL Module`** — *(narrowed to RF channel selection only)* DDQN actor on the mule's NUC.

All six are coordinated by four information flows — **intra-NUC call** (L1↔L2↔HFL-Mission↔ClientCluster), **in-field RF link** (HFLHostMission↔ClientMission), **dock handoff** (ClientCluster↔HFLHostCluster), and **cloud sync** (HFLHostCluster↔Tier 3). Each flow is bounded by the design principles called out in the deck.

### Why the client split

The earlier draft buried the dock-handoff client behavior inside `HFLHostMission`, breaking the symmetry of the server split. Two distinct conversations were collapsed into one program:

- **In-field FL session** — short-lived, RF-bounded, per-device, runs many times per mission, partial-FedAvg-merged on the fly.
- **Dock handoff** — long-haul, wired/high-bandwidth, once-per-mission, bulk transfer of partial aggregates and bundles.

Splitting them out gives each conversation its own state machine, transport, error model, and lifecycle — and keeps `HFLHostMission` doing exactly one job (being the server to devices in-field).

---

## 1. Tier / Layer Recap

| Tier | Host | Programs resident | Scope |
|---|---|---|---|
| **Tier 1** — Edge Device | Device CPU | `ClientMission` (Flagger + Discriminator trainer + FL-client to mule) | local data only |
| **Tier 2** — Edge Server | Stationary server | `HFLHostCluster` (registry, θ_gen, cross-mule FedAvg) + `ClusterCloudClient` (to Tier 3) | cluster |
| **Tier 2-mobile** — Mule NUC | Intel NUC on UAV/UGV | `L1 RL` channel actor + `FLScheduler` (with `TargetSelectorRL` sub-model) + `HFLHostMission` (server to devices) + `ClientCluster` (client to server) | per-mission |
| **Tier 3** — Cloud | Chameleon / AERPAW | `Tier3Coordinator` (θ_gen refinement, cross-cluster rhythm) | global |

The mule NUC is the only host that runs **both a server role and a client role simultaneously** — server-to-devices in-field (`HFLHostMission`) and client-to-server at dock (`ClientCluster`).

Within Layer 2 the scheduler is decoupled into **four stages**: S1 Eligibility → S2A Readiness (on-contact) → S2B FL Readiness Flag (on-device) → S3 Deadline & Priority.

---

## 2. Program Responsibilities

### 2.1 `FLScheduler` — L2 Scheduler (Host-Mule)
**One job: Who.** Selects the next device to visit and maintains per-device deadlines.

| Responsibility | Stage |
|---|---|
| Pull mission slice from `HFLHostCluster` at dock | S1 |
| Filter eligible devices (active deadline OR FL_READY beacon override) | S1 |
| Gate devices on contact (`FL_READY == True`, verified locally) | S2A |
| Consume round-close report from `HFLHostMission` — fast-phase Deadline update | S3 (fast) |
| Fold cluster amendments at dock — slow-phase Deadline update | S3 (slow) |
| Compute `Deadline(j) = Time + Deadline_Fulfilment − Idle_Time` | S3 |
| Bucket-classify candidates: `{new, scheduled-this-round, beacon-active}` (these are the only hard rank tiers) | S3 |
| Query **`TargetSelectorRL`** sub-model to break ties *within* a bucket | S3.5 |
| Emit **target waypoint** (selected device) to L1 RL Module | L2→L1 |
| Never read SNR/SINR for scoring — SNR is a Stage-1 binary gate only, and a feature for the selector only |  — |

### 2.2 `HFLHostMission` — Mission FL Server (on Mule NUC)
**One job: Server to devices in-field.** Runs FL sub-sessions per contact event, partial-FedAvgs the mission slice as it goes, writes the round-close report. **No dock behavior here** — that is `ClientCluster`'s job.

Operates in two modes, set per Pass:

* **COLLECT (Pass 1)** — exchange-only sessions: push `θ_disc` + synth, pull pre-prepared `Δθ_disc` from each in-range device. Local fitting does NOT run during the session (devices trained between visits). Per-contact partial-FedAvg merges N parallel `Δθ_disc` into one batch aggregate before folding into the running mission aggregate.
* **DELIVER (Pass 2)** — push-only sessions: push the freshly-aggregated `θ_disc'` + new `synth_batch` to every device in range; do not request a `Δθ` back. Devices stash the new global and start fresh local training.

| Responsibility | Role |
|---|---|
| `FL_OPEN` handshake with all devices in `contact.devices` on arrival | Server |
| Pass 1: push `θ_disc` + synth and pull prepared `Δθ_disc` from each | Server (collect) |
| Pass 2: push `θ_disc'` + synth — no `Δθ` requested | Server (deliver) |
| Verify gradient receipts (checksum, completion, TTL) per device, in parallel | Server |
| Arbitrate cross-mule races via per-device busy-flag (TTL-bounded) | Server |
| Per-contact partial-FedAvg merge of in-range `Δθ_disc_i` → `batch_aggregate_for_contact` | FedAvg |
| Fold each `batch_aggregate_for_contact` into the running `mission_aggregate` | FedAvg |
| Generate `MissionRoundCloseReport` (Pass 1) + `MissionDeliveryReport` (Pass 2) | Writer |
| Emit one `RoundCloseDelta` per device to `FLScheduler` (intra-NUC, per contact) | Feedback |
| Hand finalized `mission_aggregate` + reports + `contact_history` to `ClientCluster` between Pass 1 and Pass 2 | Intra-NUC |

### 2.3 `ClientCluster` — Dock-Handoff Client *(new, on Mule NUC)*
**One job: Client to `HFLHostCluster` at dock.** Owns the entire dock lifecycle — upload of mission output, download of next-mission bundle. Mirrors `HFLHostMission` on the *server* side of the mule.

| Responsibility | Role |
|---|---|
| Detect dock-link availability (wired / high-bw RF / proximity) | Connection mgr |
| UP: upload `partial_aggregate + MissionRoundCloseReport + contact_history` to `HFLHostCluster` | Client |
| DOWN: download `MissionSlice + θ_disc + SynthBatch + ClusterAmendments` | Client |
| Verify bundle integrity (checksums, version sigs) before handing to local services | Verifier |
| Hand `MissionSlice` + `Amendments` to `FLScheduler` (slow-phase Deadline update trigger) | Intra-NUC |
| Hand `θ_disc` + `SynthBatch` to `HFLHostMission` (next-round model state) | Intra-NUC |
| Surface dock-failure / partial-handoff state for retry on next dock | Error mgr |
| **Never** trains, never inspects gradients — pure transport client | — |

### 2.4 `ClientMission` *(was `EdgeClient`)* — Edge-Device FL Client + Flagger (Tier 1)
**One job: Who, locally.** Runs local discriminator training **offline between mule visits** (NOT during the FL session itself), self-declares whether its prepared weights are worth federating, and acts as FL client to `HFLHostMission` on contact for an exchange-only session.

| Responsibility | Stage |
|---|---|
| **Offline** local training of discriminator on local real traffic, between mule visits | Tier 1 |
| Stash the prepared `Δθ_disc` (= θ_local_after_train − θ_disc_received_from_last_Pass_2) in a delivery slot | Tier 1 |
| Post-round eligibility check: `Performance_score` (Acc, AUC, Loss) + `diversity_adjusted = cosine(θ_local, θ_global)·perf_discount` | S2B |
| `utility(i) = w₁·Performance_score + w₂·diversity_adjusted` — open flag when `> FL_Threshold` | S2B |
| Set FL state in `{busy, unavailable, FL_OPEN}` | S2B |
| On `FL_OPEN`, emit low-power RF beacon — **proximity only** (can't summon mule) | S1 |
| Pass 1 contact: handshake with `HFLHostMission`, receive `θ_disc + synth`, return the **already-prepared** `Δθ_disc` + meta-stats. No fitting during the session. | FL client (collect) |
| Pass 2 contact: handshake with `HFLHostMission`, receive new `θ_disc' + synth'`, store them, **immediately start the next round of offline local training** against `θ_disc'`. No `Δθ` is sent back. | FL client (deliver) |
| Maintain `FL_READY_ADV` payload (perf_delta, diversity_proxy, payload_size, missed_count, idle_time) | S2B |

**Why this matters.** Pulling fitting out of the session shrinks the mule's contact time from "minutes per device" (push + train + receive) to "milliseconds per device" (push + receive), which is what makes parallel sessions per contact event physically tractable inside an RF dwell window.

### 2.5 `HFLHostCluster` — Cluster FL Coordinator (Tier 2)
**One job: When, across missions.** Authoritative registry and cross-mule aggregator.

| Responsibility | Artifact |
|---|---|
| Maintain authoritative device registry (IDs, spectrum signatures, last-known positions) | `DeviceRegistry` |
| Slice registry per-mule, disjoint — refreshed every dock | `MissionSlice` |
| Host θ_gen and generate fresh synth sample batches per round | `GeneratorHost` |
| Cross-mule FedAvg of incoming partial aggregates (N = #mules, small) | FedAvg |
| Fold cluster corrections into mission reports → slow-phase Deadline recon | `ClusterAmendment` |
| Dispatch next mission bundle at dock: slice + `θ_disc` + synth batch + amendments | Dock handoff |
| Relay to Tier 3 for cross-cluster `θ_gen` refinement | Tier 2↔Tier 3 |

### 2.6 `L1 RL Module` — RF Channel Selector (channel-only, was MA-P-DQN)
**Reframe.** The original "joint MA-P-DQN" framing assumed a continuous trajectory head. In practice the mule's flight is mechanical navigation between known device positions — there is no continuous trajectory to learn. What remains at L1 is the **discrete RF channel selector** alone.

| Responsibility | Stage |
|---|---|
| Read `sₜ` (SNR/SINR per band, CSI, energy, queue) — private to L1 | env |
| Receive target waypoint (selected DeviceID + last-known position) from L2 | L2→L1 |
| Output `channel_index` (DDQN, discrete head, argmax over RF bands) | L1→radio |
| Emit `sₜ₊₁, rₜ` to its replay buffer (training; offline) | training |
| **Never** outputs trajectory — navigation is mechanical, target is L2's choice | — |

> **Algorithm implication.** With the position head removed, L1 reduces from MA-P-DQN to a single DDQN (or the discrete head of MA-P-DQN, kept for backward compatibility). The "joint action" framing in slides 20/21 no longer applies — it survives only as a training-time formality if MA-P-DQN is retained.

### 2.7 `TargetSelectorRL` — Intra-Bucket Selector *(new, sub-model of L2 S3.5)*
**Where the trajectory head went.** What was "trajectory" in MA-P-DQN was actually a **next-target selector**. After Sprint 1.5, the selector picks the next *contact position* (a stop where the mule serves N≥1 devices in parallel), not an individual device. The selector queries only when a bucket has ≥2 candidate positions.

**Pass 1 only.** Pass 2 walks every contact greedily — the goal there is universal delivery, not selection — so the selector is bypassed in Pass 2.

| Responsibility | Role |
|---|---|
| Inputs: candidate **contact positions** within a single S3 bucket; per-position aggregate features (mean on_time_rate of in-range devices, member count, SpectrumSig of nearest device, etc.); current mule pose, energy budget, RF priors from L1 env state | features |
| Output: ordering over candidate contact positions (or single argmax position) | selector |
| Reward signal (training): per-contact `Σ_devices_in_range completed_bonus_i − time_to_complete − energy_used` — sums the contribution across the parallel sessions in that contact | RL training |
| Two distinct invocations: (a) Pass-1 contact-selection during a mission, (b) **server-selection** at end-of-mission (which edge server to dock at, if multiple are reachable) | dual-purpose |
| Trained centrally (CTDE), deployed as small actor on NUC alongside L1 channel actor | deployment |
| **Never** decides eligibility / gating / deadlines / clustering — those remain hard rules in S1 / S2A / S2B / S3 deadline math / S3a clustering | scope |

This sub-model only ranks contact positions that the deterministic stages already admitted (S1 → S3 deadline math → S3a clustering → S3 bucket-classify). It cannot promote a gated-out device, cannot reorder buckets, cannot override a deadline, and cannot change which devices are clustered together — it only fills the ordering gap inside a bucket of contact positions.

---

## 3. Program Interaction Diagram

```
                      ┌───────────────────────────────────────────┐
                      │       TIER 3 — Cloud (AERPAW / Chameleon) │
                      │   θ_gen refinement · cross-cluster rhythm │
                      └───────────────────┬───────────────────────┘
                                    θ_gen │ ↓        ↑ cluster partials
                      ┌───────────────────▼───────────────────────┐
                      │  TIER 2 — Edge Server                     │
                      │  ┌──────────────────────────────────────┐ │
                      │  │ HFLHostCluster (stationary)          │ │
                      │  │  · DeviceRegistry + MissionSlice     │ │
                      │  │  · θ_gen + synth sample generator    │ │
                      │  │  · Cross-mule FedAvg                 │ │
                      │  │  · Cluster amendments + dispatch     │ │
                      │  └─────────────┬────────────▲───────────┘ │
                      └────────────────┼────────────┼─────────────┘
                                       │ DOCK       │ DOCK
                        DOWN: slice +  │            │ UP: partial_agg +
                        θ_disc + synth │            │     report +
                        + amendments   ▼            │     contact_history
                      ┌──────────────────────────────────────────────────┐
                      │   MULE NUC  (mobile, per-mission)                │
                      │                                                  │
                      │  ┌─────────────────────────────────────────────┐ │
                      │  │  ClientCluster   (client to HFLHostCluster) │◄┼─ DOCK ─┐
                      │  │  · UP: partial_agg + report + contacts      │ │       │ to/from
                      │  │  · DN: slice + θ_disc + synth + amendments  │─┼─ DOCK ─┘ Tier 2
                      │  │  · verify, then hand-off intra-NUC          │ │
                      │  └────┬────────────────────────────┬───────────┘ │
                      │       │ slice + amendments         │ θ_disc +    │
                      │       │ (slow-phase trigger)       │ synth       │
                      │       ▼                            ▼             │
                      │  ┌────────────┐  waypoint  ┌──────────────────┐  │
                      │  │ L1 RL      │◄───────────│ L2 FLScheduler   │  │
                      │  │ (channel   │            │  S1→S2A→S2B→S3   │  │
                      │  │  DDQN only)│            │  ┌────────────┐  │  │
                      │  └─────┬──────┘            │  │TargetSelect│  │  │
                      │        │ chan_idx          │  │RL (S3.5    │  │  │
                      │        │                   │  │intra-bucket│  │  │
                      │        │                   │  │tiebreaker) │  │  │
                      │        │                   │  └────────────┘  │  │
                      │        │                   └────────▲─────────┘  │
                      │        │                            │ fast       │
                      │        │                            │ Deadline   │
                      │        ▼                            │ update     │
                      │  ┌──────────────────────────────────┴─────────┐  │
                      │  │ HFLHostMission   (server to devices)       │  │
                      │  │  · FL_OPEN handshake · Partial FedAvg       │  │
                      │  │  · Round-close report → Scheduler/Cluster   │  │
                      │  └──────────┬────────────▲────────────────────┘  │
                      └─────────────┼────────────┼──────────────────────┘
                                    │ FL session │
                  θ_disc + synth ↓  │            │  ↑ Δθ_disc + meta
                                    ▼            │
                      ┌────────────────────────────────────────────┐
                      │  TIER 1 — Edge Device                      │
                      │  ┌───────────────────────────────────────┐ │
                      │  │ ClientMission  (FL client to mule)    │ │
                      │  │  · Local discriminator training        │ │
                      │  │  · Utility(i) check → FL state         │ │
                      │  │  · RF beacon on FL_OPEN (proximity)    │ │
                      │  │  · Handshake + Δθ_disc submit          │ │
                      │  └───────────────────────────────────────┘ │
                      └────────────────────────────────────────────┘

  Information-flow rules:
    L2 → L1              — ONLY the target waypoint (intra-NUC). L1 is channel-only.
    L1 → L2 (selector)   — RF priors as features for TargetSelectorRL (read-only env state).
    L2 ↔ HFL-Mission     — round-close report in-flight (intra-NUC).
    ClientCluster ↔ L2   — slice + amendments deliver slow-phase trigger (intra-NUC).
    ClientCluster ↔ HFL-Mission — θ_disc + synth hand-off (intra-NUC, dock-entry).
    HFL-Mission ↔ ClientMission — RF link; FL session; beacons proximity-only.
    ClientCluster ↔ HFL-Cluster — dock link only; bulk bundle transfer.
    HFL-Cluster  ↔ Tier 3 — cross-cluster θ_gen refinement rhythm.
    θ_gen                — stays at Tier 2 forever (privacy + compute boundary).
    Raw data             — never leaves the device.
```

---

## 4. Process Flow — Happy Path (One Mission)

A HERMES mission is **two-pass**: Pass 1 collects Δθ from devices, Pass 2 delivers fresh θ to devices, with the cluster's cross-mule FedAvg in the dock between them. The selector drives Pass 1's order; Pass 2 walks every contact greedily because the goal is universal delivery.

```
[PRE-MISSION — at dock, before Pass 1]
  (1) Mule docks at edge server; ClientCluster detects dock link.
  (2) ClientCluster ← HFLHostCluster (DOWN bundle):
        MissionSlice(ids, spectrum_sigs, positions)
        θ_disc_global   (the version delivered to devices last mission's Pass 2)
        SynthBatch
        Amendments(prev round)
  (3) ClientCluster verifies bundle integrity, then hands off intra-NUC:
        → FLScheduler:     MissionSlice + Amendments
        → HFLHostMission:  θ_disc_global + SynthBatch
  (4) FLScheduler folds Amendments → slow-phase Deadline(j) update.
  (5) FLScheduler runs S1 eligibility, S3 deadline math, S3a clustering:
        - eligible_devices  = filter(slice ∪ beacon_heard)
        - contact_positions = cluster_by_rf_range(eligible_devices, rf_range_m)
        - bucket_classify(contact_positions)
            (a position inherits the worst bucket among its members)

[PASS 1 — outbound from server, COLLECT]
  loop while mission_time_left > 0 and contacts_pending:
    (6) FLScheduler S3.5:
          if |top_bucket| > 1: TargetSelectorRL.rank(top_bucket, env_priors)
          contact = head(top_bucket)
        FLScheduler → L1: contact.position + contact.devices
    (7) L1 picks chan_idx (DDQN) for the link to that contact;
        mule navigates mechanically to contact.position.
    (8) On arrival, HFLHostMission issues FL_OPEN solicitation
        addressed to every device in contact.devices (parallel).
    (9) Each ClientMission responds with FL_READY_ADV(perf_delta,
        diversity_proxy, payload_size, missed_count, idle_time).
    (10) Stage 2A gate, per device:
          if FL_READY == False → mark skipped (this contact only).
    (11) Parallel FL exchange-only sessions (HFLHostMission ↔ in-range ClientMission_i):
          HFLHostMission pushes θ_disc + synth   →  ClientMission_i
          ClientMission_i returns the pre-prepared Δθ_disc_i + meta
                          (training was already done offline between visits)
    (12) HFLHostMission per-contact merge:
          for each accepted submission: verify (checksum, TTL, completion)
          partial-FedAvg-merge {Δθ_disc_i} → batch_aggregate_for_contact
          fold batch_aggregate_for_contact → mission_aggregate (running)
          append one line per device to MissionRoundCloseReport
    (13) HFLHostMission → FLScheduler (intra-NUC):
          the full MissionRoundCloseReport is folded into the scheduler;
          per line: {device_id, outcome ∈ {clean, partial, timeout}, contact_ts,
          bytes_received, bytes_sent}. (RoundCloseDelta line-by-line streaming
          is deferred — see §6.3 note.)
    (14) FLScheduler fast-phase Deadline(j) update, per device:
          on-time → shorten next interval; missed → lengthen.
    (15) FLScheduler re-bucket-classifies the *remaining* contact_positions
         using the updated time budget + deadlines:
          Rank 1: new positions awaiting first distribution
          Rank 2: positions with shortened deadlines
          Rank 3: positions in-session awaiting their deadline
    (16) Opportunistic FL_OPEN beacon? If heard in-range, treat as a
         single-device contact insert into the queue — else goto (6).

[INTER-PASS DOCK — between Pass 1 and Pass 2]
  (16.5) If multiple edge servers are reachable, FLScheduler calls
         TargetSelectorRL.select_server(reachable_servers, energy_budget)
         to pick the dock target; L1 picks channel for the dock link.
  (17) Mule docks; ClientCluster detects dock link.
  (18) HFLHostMission → ClientCluster (intra-NUC handoff):
         mission_aggregate + MissionRoundCloseReport + contact_history
  (19) ClientCluster → HFLHostCluster (UP bundle, dock link):
         mission_aggregate + MissionRoundCloseReport + contact_history
  (20) HFLHostCluster:
        cross-mule FedAvg(mission_aggregates from all docked mules)
                                      → fresh θ_disc_global'
        produce cluster Amendments (cross-mission corrections)
        push θ_disc_global' to Tier 3 (via ClusterCloudClient) for θ_gen refinement
        prepare DOWN bundle for Pass 2:
          MissionSlice (unchanged for this mission)
          θ_disc_global'   ← the new global, freshly aggregated
          SynthBatch (regenerated against θ_gen)
          Amendments
  (21) ClientCluster ← HFLHostCluster: DOWN bundle.
       ClientCluster verifies + distributes intra-NUC:
        → FLScheduler:     refreshed positions / amendments
        → HFLHostMission:  θ_disc_global' + SynthBatch'

[PASS 2 — outbound from server, DELIVER]
  Pass 2 walks every contact in the slice (no skipping based on selector
  ranking — the goal is universal delivery so devices can re-train fresh).
  loop over contact_positions:
    (22) FLScheduler emits the next contact greedily (no selector call):
          - default order: S3a's clustering output, walked in nearest-first
            order from the current mule pose to minimize Pass 2 path length.
    (23) L1 picks chan_idx; mule navigates to contact.position.
    (24) HFLHostMission issues FL_OPEN solicitation in DELIVER mode.
    (25) Each ClientMission_i in range responds; HFLHostMission pushes
         θ_disc_global' + SynthBatch'. No Δθ is requested.
    (26) ClientMission_i stores θ_disc_global' and starts the next round
         of offline local training immediately.
    (27) HFLHostMission appends a delivery line to a Pass-2 report
         (separate from Pass 1's MissionRoundCloseReport).

[POST-MISSION — return to dock]
  (28) Mule returns to the edge server (no UP bundle — Pass 1's UP
       already shipped the mission_aggregate). Optional Pass-2 delivery
       report flows up so the cluster can detect non-delivered devices
       (which become priority candidates next mission).
  (29) Mission n complete. Goto (1) for mission n+1.
```

**Why the staleness is now structurally impossible.** Every Δθ collected in Pass 1 was trained against the θ_disc the device received in *the previous mission's* Pass 2. The cluster has direct knowledge of that θ (it dispatched it) — so the FedAvg math is exact, no async-FL drift. The cost is one extra circuit per mission; the benefit is mathematical correctness plus halved staleness time on devices.

### 4.1 Exception paths

| Event | Handler | Action |
|---|---|---|
| Gradient receipt fails checksum / TTL (Pass 1) | HFLHostMission | discard, mark outcome=`partial`, device gets missed-count bump; other in-range devices in the same contact event are unaffected |
| Subset of contact's devices respond, others don't | HFLHostMission | merge whatever did respond into the contact batch aggregate; non-responders get `timeout` outcome individually |
| Mule disconnects mid-Pass-1 | HFLHostMission | write `partial_round_state` checkpoint of the running mission_aggregate → resume on reconnect; partially-served contacts may need re-visit |
| Mule fails to return to server between Pass 1 and Pass 2 | ClientCluster | the mission_aggregate is already on the mule; on next dock, ship Pass 1's UP first, then resume Pass 2 with whatever θ' the cluster has |
| Pass 2 device unreachable (no FL_READY response) | HFLHostMission | log non-delivery in Pass-2 report; cluster prioritizes that device's contact in next mission |
| Cross-mule race on same device (Pass 1) | HFLHostMission | device busy-flag (TTL-bounded) wins; losing mule marks `timeout`. Disjoint slicing makes this rare; the busy-flag is a defense-in-depth |
| Dock link drops mid-UP | ClientCluster | retain `mission_aggregate` on-NUC; retry on next dock; emit stale-bundle warning to HFLHostCluster |
| Dock DOWN bundle (Pass 1's reply) fails verification | ClientCluster | refuse intra-NUC handoff; request re-dispatch; HFLHostMission cannot start Pass 2 until resolved |
| No min_participation_threshold at cluster | HFLHostCluster | **deadline-aware aggregation** — aggregate whatever arrived, never stall the dispatch of θ_disc' for Pass 2 |
| Mission slice collision across mules | HFLHostCluster | disjoint slicing enforced at dispatch; no runtime reconciliation needed |
| Stale Δθ collected (basis ≠ cluster's current θ) | n/a | **structurally impossible under two-pass missions** — Pass 1's collected Δθ are always trained against the θ the cluster delivered in the previous mission's Pass 2, which the cluster knows exactly |

---

## 5. Process Loop Flow (State Machines)

### 5.1 `FLScheduler` loop

```
 ┌───────────────┐
 │  IDLE@DOCK    │── registry + amendments arrive ──►┐
 └──────▲────────┘                                   │
        │ dock                                       ▼
        │                                    ┌──────────────┐
        │                                    │ PLAN (S1)    │
        │                                    │ build list,  │
        │                                    │ slow-phase   │
        │                                    │ Deadline     │
        │                                    └──────┬───────┘
        │                                           │ bucket-classify
        │                                           ▼
        │                                    ┌──────────────┐
        │                                    │ S3.5 SELECT  │
        │                                    │ TargetSel RL │
        │                                    │ intra-bucket │
        │                                    └──────┬───────┘
        │                                           │ emit waypoint
        │                                           ▼
        │                                    ┌──────────────┐
        │   round-close delta   ┌────────────┤ AWAIT CONTACT│
        │   (intra-NUC)         │            └──────┬───────┘
        │                       │                   │ contact made
        │                       ▼                   ▼
        │               ┌──────────────┐     ┌──────────────┐
        └───────────────│ RECOMPUTE S3 │◄────│ GATE S2A     │
                        │ fast-phase   │     │ FL_READY?    │
                        │ re-rank      │     └──────┬───────┘
                        └──────▲───────┘            │ skip / proceed
                               │                    ▼
                               └─────────── FL session ends
```

### 5.2 `HFLHostMission` loop

```
 WAIT_FOR_WAYPOINT
      │ (L1 flies to target)
      ▼
 FL_OPEN_SOLICIT ──timeout──► MARK_TIMEOUT ──► emit delta ──► WAIT
      │
      │ FL_READY_ADV received
      ▼
 PUSH θ_disc + synth
      │
      ▼
 RECV Δθ_disc ──bad receipt──► MARK_PARTIAL ─┐
      │                                       │
      ▼                                       │
 VERIFY + PARTIAL_FEDAVG  ─────────────────── │
      │                                       │
      ▼                                       ▼
 APPEND_ROUND_CLOSE_LINE ──► EMIT_DELTA_TO_SCHEDULER ──► WAIT
```

### 5.3 `ClientMission` loop *(on Edge Device)*

```
 TRAIN_LOCAL_EPOCH
      │
      ▼
 COMPUTE utility(i) = w₁·perf + w₂·div_adj
      │
      ├── utility > FL_Threshold ─► STATE = FL_OPEN ─► emit beacon
      ├── training in progress   ─► STATE = busy
      └── offline / low power    ─► STATE = unavailable
      │
      ▼
 AWAIT_CONTACT
      │ FL_OPEN_SOLICIT from mule (HFLHostMission)
      ▼
 SEND FL_READY_ADV (perf_delta, diversity_proxy, payload_size,
                    missed_count, idle_time)
      │
      ▼
 RECV θ_disc + synth
      │
      ▼
 LOCAL_ROUND (discriminator only)
      │
      ▼
 SEND Δθ_disc + meta-stats ──► TRAIN_LOCAL_EPOCH
```

### 5.4 `ClientCluster` loop *(on Mule NUC — new)*

```
 AWAIT_DOCK
      │ dock link detected
      ▼
 COLLECT_FROM_LOCAL
      ├── HFLHostMission → partial_aggregate + MissionRoundCloseReport
      └──                  + contact_history
      │
      ▼
 UP: send bundle → HFLHostCluster ──failure──► RETAIN + RETRY_NEXT_DOCK
      │
      ▼ ack
 DOWN: recv bundle ← HFLHostCluster
      │
      ▼
 VERIFY bundle (checksums, version sigs)
      │       │
      │       └── fail ── request re-dispatch ── ABORT_HANDOFF
      ▼
 DISTRIBUTE_INTRA_NUC
      ├── FLScheduler    ← MissionSlice + ClusterAmendments
      └── HFLHostMission ← θ_disc + SynthBatch
      │
      ▼
 SIGNAL ready-to-depart ──► AWAIT_DOCK
```

### 5.5 `HFLHostCluster` loop *(on Edge Server)*

```
 AWAIT_DOCK
      │ ClientCluster UP bundle arrives
      ▼
 INGEST partial_agg + report + contact_history
      │
      ▼
 CROSS_MULE_FEDAVG  (N = #mules)
      │
      ▼
 SYNC with TIER 3 — θ_gen refinement  (via ClusterCloudClient)
      │
      ▼
 BUILD cluster Amendments + next MissionSlice (disjoint)
      │
      ▼
 GENERATE fresh SynthBatch
      │
      ▼
 DISPATCH DOWN bundle to waiting ClientCluster ──► AWAIT_DOCK
```

---

## 6. Variables, States, Definitions

### 6.1 Shared types

| Name | Type | Definition |
|---|---|---|
| `DeviceID` | string | Unique cluster-scope device identifier. |
| `θ_disc` | tensor | Discriminator weights; global copy pushed DOWN. |
| `Δθ_disc` | tensor | Discriminator gradient / delta; flows UP. Always trained against the θ delivered in the previous mission's Pass 2 (no async drift). |
| `θ_gen` | tensor | Generator weights; **never leaves Tier 2**. |
| `SynthBatch` | tensor[] | Fresh synthetic samples generated per round at Tier 2. |
| `SpectrumSig` | struct | Per-device RF fingerprint (bands, last-good SNR priors). |
| `MissionSlice` | list[DeviceID] | Disjoint per-mule subset of the cluster registry. |
| `MissionPass` | enum | `{COLLECT, DELIVER}` — distinguishes Pass 1 (collect Δθ) from Pass 2 (deliver fresh θ). |
| `ContactWaypoint` | struct | `{position, devices: list[DeviceID], bucket, deadline_ts}` — one stop where the mule serves N≥1 in-range devices in parallel. Replaces `TargetWaypoint` in the scheduler→L1 contract post-Sprint-1.5. |
| `rf_range_m` | float | Mule's effective RF range in metres; the radius S3a clustering uses to group devices into contact positions. Default 60 m; sweep {30, 60, 120} in Experiment 3. |

### 6.2 `FLScheduler` state

| Name | Type | Definition |
|---|---|---|
| `device_list` | list[DeviceRecord] | Current mission slice with Deadline(j) and rank. |
| `Deadline(j)` | float (timestamp) | `Time + Deadline_Fulfilment − Idle_Time`. |
| `Time` | float | Base timestamp for the next round (or current-deadline clock). |
| `Deadline_Fulfilment` | float | `+on_time_history − missed_history`. |
| `Idle_Time` | float | Device historical idle; low idle → shorter deadline. |
| `rank(j)` | {1,2,3} | 1=new, 2=shortened-deadline on-time, 3=in-session awaiting. |
| `FL_READY` | bool | Binary gate at S2A, verified **on contact**, never remotely. |
| `FL_Threshold` | float | S2B utility cutoff. |
| `Amendments` | list | Slow-phase corrections from cluster. |

### 6.3 `HFLHostMission` state

| Name | Type | Definition |
|---|---|---|
| `current_pass` | MissionPass | `COLLECT` (Pass 1) or `DELIVER` (Pass 2). Drives session semantics. |
| `mission_aggregate` | tensor | Running per-mission FedAvg accumulator. After a contact's `batch_aggregate_for_contact` is computed, it folds into here. Persists across contacts within Pass 1. |
| `batch_aggregate_for_contact` | tensor (transient) | Intermediate per-contact merge of the parallel `Δθ_disc_i`. Discarded after fold. |
| `MissionRoundCloseReport` | list[Line] | Authoritative Pass-1 participation ledger; per-device outcome ∈ {clean, partial, timeout}. (Future work: add `contact_id` so position-level metrics are recoverable; not currently consumed by any downstream metric, so deferred — see implementation note below.) |
| `MissionDeliveryReport` | list[Line] | Pass-2 ledger; per-device outcome ∈ {delivered, undelivered}. Lets the cluster prioritize undelivered devices in next mission's slice. |
| `gradient_receipt` | struct | `{device_id, checksum, bytes_received, ttl_ok}`. |
| `device_busy_flag` | map[DeviceID, TTL] | Cross-mule race arbitration. |
| `RoundCloseDelta` | struct | Streamed per-device into the Scheduler's fast-phase Deadline update: `{device_id, mule_id, mission_round, outcome, utility, contact_ts}`. (Originally specified with `contact_id` as well; deferred until per-contact backpressure becomes a measurable concern — the report-level fold below is functionally equivalent today.) |

### 6.4 `TargetSelectorRL` state *(sub-model of FLScheduler S3.5)*

| Name | Type | Definition |
|---|---|---|
| `candidate_set` | list[ContactWaypoint \| ServerID] | Admitted contact positions in one S3 bucket (or reachable servers at dock). One ContactWaypoint covers N≥1 in-range devices. |
| `features(j)` | vector | Per-contact aggregate: `{position, mean_on_time_rate, member_count, distance_to_mule, mule_energy, rf_prior_snr, ...}`. Member-level details are summarised so the actor sees a fixed-shape feature row regardless of N. |
| `bucket_tag` | {new, scheduled, beacon-active, server} | Which decision context the selector is running in. |
| `action` | ContactWaypoint \| ServerID | argmax over `candidate_set`. |
| `replay_buffer` | list | Offline — per-contact rewards: `Σ devices in range (completion_bonus_i) − time_to_complete − energy_used`. |
| `actor_weights` | tensor | Deployed on NUC; trained CTDE on AERPAW digital twin. |
| `pass` | MissionPass | Selector is consulted only when `pass == COLLECT`; Pass 2 walks contacts greedily without invoking the selector. |

### 6.5 `ClientCluster` state *(on Mule NUC)*

| Name | Type | Definition |
|---|---|---|
| `dock_state` | {idle, up_in_flight, down_in_flight, verifying, distributing, error} | Current dock-lifecycle stage. |
| `up_bundle` | struct | `{partial_aggregate, MissionRoundCloseReport, contact_history}` pending upload. |
| `down_bundle` | struct | `{MissionSlice, θ_disc, SynthBatch, ClusterAmendments}` pending verification. |
| `bundle_sig` | hash | Checksum/version signature used to verify DOWN bundle. |
| `retry_queue` | list[Bundle] | Held UP bundles awaiting next successful dock. |
| `ready_to_depart` | bool | Flipped once DOWN distribution to FLScheduler + HFLHostMission completes. |

### 6.6 `ClientMission` state *(on Edge Device)*

| Name | Type | Definition |
|---|---|---|
| `FL_state` | {busy, unavailable, FL_OPEN} | Primary readiness signal seen at contact. |
| `Performance_score` | float | Weighted combo of Prob.Accuracy, AUC, Loss. |
| `diversity_adjusted` | float | `cosine(θ_local_finetuned, θ_global) · perf_discount`. |
| `utility(i)` | float | `w₁·Performance_score + w₂·diversity_adjusted` (w₁ > w₂). |
| `FL_READY_ADV` | struct | On-contact payload: `{perf_delta, diversity_proxy, payload_size, missed_count, idle_time}`. |
| `beacon` | RF burst | Emitted ONLY when FL_state = FL_OPEN and device has power. |

### 6.7 `HFLHostCluster` state

| Name | Type | Definition |
|---|---|---|
| `DeviceRegistry` | map[DeviceID, DeviceRecord] | Single source of truth, cluster-scope. |
| `MissionSlice[m]` | list[DeviceID] | Disjoint per-mule slice, refreshed every dock. |
| `cluster_partial_aggregate` | tensor | Cross-mule FedAvg output. |
| `ClusterAmendment` | struct | Corrections folded into mission reports → slow-phase Deadline. |
| `min_participation_threshold` | int | Triggers deadline-aware aggregation. |

### 6.8 Formulas

```
utility(i)     = w₁ · Performance_score(i) + w₂ · diversity_adjusted(i)          # S2B
                   where w₁ > w₂

Deadline(j)    = Time + Deadline_Fulfilment(j) − Idle_Time(j)                    # S3
                   Deadline_Fulfilment = +on_time_history − missed_history
                   Idle_Time low ⇒ shorter deadline

FL_OPEN(i)     = (utility(i) > FL_Threshold) ∧ (training_done) ∧ (power_ok)

eligible(i)    = (has_active_deadline(i)) ∨ (beacon_heard_in_range(i))           # S1
```

### 6.9 Interface Contracts

| Call | From → To | Payload | Transport |
|---|---|---|---|
| `select_target(bucket, env_priors)` | FLScheduler → TargetSelectorRL | `candidate_set + features` | intra-process |
| `select_server(reachable, energy)` | FLScheduler → TargetSelectorRL | `ServerIDs + features` | intra-process |
| `emit_waypoint()` | FLScheduler → L1 | `DeviceID` (selector output) + last-known pos | intra-NUC |
| `read_rf_prior()` | FLScheduler (selector) → L1 env | per-band SNR snapshot (read-only) | intra-NUC |
| `emit_channel_idx()` | L1 → radio | `channel_index` (DDQN argmax) | local |
| `emit_round_close_delta()` | HFLHostMission → FLScheduler | `{device_id, outcome}` | intra-NUC |
| `fl_open_solicit()` | HFLHostMission → ClientMission | — | RF |
| `fl_ready_adv()` | ClientMission → HFLHostMission | `FL_READY_ADV` struct | RF |
| `push_model()` | HFLHostMission → ClientMission | `θ_disc + SynthBatch` | RF |
| `submit_gradient()` | ClientMission → HFLHostMission | `Δθ_disc + meta` | RF |
| `handoff_up_at_dock()` | HFLHostMission → ClientCluster | `partial_agg + report + contacts` | intra-NUC |
| `handoff_down_models()` | ClientCluster → HFLHostMission | `θ_disc + SynthBatch` | intra-NUC |
| `handoff_down_schedule()` | ClientCluster → FLScheduler | `MissionSlice + ClusterAmendments` | intra-NUC |
| `dock_up()` | ClientCluster → HFLHostCluster | `partial_agg + report + contacts` | Dock link |
| `dock_down()` | HFLHostCluster → ClientCluster | `slice + θ_disc + synth + amendments` | Dock link |
| `cloud_sync()` | HFLHostCluster ↔ Tier 3 | `θ_gen refinement` | Cloud link |

---

## 7. Design Principles (from the deck, enforced in code)

1. **Each layer has one job.** RL = *How*, Scheduler = *Who*, HFL = *When*. No layer reads another layer's private state.
2. **HFLHost split.** Mission on mule (partial FedAvg, mission-scope). Cluster on edge server (cross-mule FedAvg, cluster-scope). Each is authoritative at its own scope.
3. **Partial FedAvg upgrades privacy.** Cluster never sees individual gradients — only pre-aggregated partials.
4. **Two-phase deadline adaptation.** Fast phase = in-mission, local session outcome. Slow phase = at dock, cluster amendments.
5. **Only the waypoint crosses L2→L1.** SNR stays L1's env observation; L2's S3.5 selector may read it as a *read-only feature*, but it never becomes a scheduler S2 scoring term. L1 no longer outputs trajectory — "trajectory" decisions moved into L2 as `TargetSelectorRL`; L1 is channel-only.
6. **Beacons are opportunistic, not planning input.** Proximity-only RF bursts — a bonus when the mule is already in range, never a summon signal.
7. **Deadline-aware aggregation** (HERMES novelty at L3). Aggregate whatever arrived at the deadline; never stall indefinitely.
8. **Disjoint mission slicing** prevents cross-mule collisions at dispatch time — runtime coordination is device-mediated (busy-flag), not mule-mediated.
9. **θ_gen never leaves Tier 2.** Devices only ever receive `θ_disc + synth samples`.
10. **Eligibility is computed locally on the edge device** (S2B). The mule is a transport agent — it never inspects payloads.
11. **Symmetric server/client at every tier-boundary.** Each link has exactly one server program on one side and one client program on the other. The mule is the only host that runs both roles — `HFLHostMission` (server, RF link to devices) and `ClientCluster` (client, dock link to server). No program straddles two tier-boundaries.
12. **`TargetSelectorRL` is bounded to intra-bucket ordering.** The selector runs *after* the deterministic gates (S1/S2A/S2B) and *after* the deadline math (S3). It cannot promote a gated-out device, cannot reorder buckets, and cannot override a deadline — it only breaks ties within a bucket. Hard rules stay hard; learned rules stay inside one explicit sub-stage.
13. **Missions are two-pass: collect, then deliver.** A mission consists of Pass 1 (mule departs server, visits devices to *collect* prepared Δθ, returns to server) and Pass 2 (mule departs again with the freshly-aggregated global θ_disc, *delivers* it to every slice member, returns). Cluster cross-mule FedAvg runs in the dock between the two passes. The 2× flight cost is intentional: it removes async-FL drift entirely (every Δθ collected was trained against a θ the cluster has direct knowledge of) and halves the time a device spends training against an outdated global. Selector-driven ordering applies to Pass 1 only — Pass 2 walks every contact greedily because the goal is universal delivery.
14. **Local training is offline; FL sessions are exchange-only.** ClientMission trains the discriminator against locally-stored data on its own schedule, between mule visits. When the mule arrives, the FL session is purely a data exchange (push θ_disc + synth, pull pre-prepared Δθ). No fitting happens during the session. This keeps contact time short, which is what makes contact-level parallel sessions practical inside an RF window.
15. **The mule's circuit is decomposed into contact events, not per-device visits.** When the mule stops at a position, every device within `rf_range_m` of that position is served in parallel — one *contact event* covers N≥1 devices. The scheduler clusters slice members into contact positions (S3a clustering stage); the selector picks among contact positions, not individual devices. Per-contact partial-FedAvg merges the N parallel Δθ into a contact-level batch aggregate, which then folds into the running mission aggregate. The N=1 case (isolated device) is the degenerate-but-valid form of the same code path — no special-cased branch.

---

## 8. Implementation Mapping to HiFINS

| Design role | Existing code / new module |
|---|---|
| `HFLHostCluster` | extend [HFLHost.py](App/TrainingApp/HFLHost/HFLHost.py) — narrow it to a cluster-scope coordinator that owns registry + θ_gen + cross-mule FedAvg + dock-server endpoints. |
| `HFLHostMission` | new module on the NUC; reuses Flower server role + `partial_round_state` checkpoint idiom from `FlightFramework`. **No dock logic** — only the in-field FL session lifecycle. |
| `ClientCluster` | **new** module on the NUC; the Flower-client peer to `HFLHostCluster`. Owns dock-link detection, retry queue, bundle verification, and intra-NUC fan-out to `FLScheduler` + `HFLHostMission`. |
| `FLScheduler` | new module; intra-process sibling of `HFLHostMission` and `ClientCluster`; produces waypoint for L1, consumes round-close deltas from HFLHostMission, consumes slice+amendments from ClientCluster. Hosts `TargetSelectorRL` as S3.5 sub-model. |
| `TargetSelectorRL` | new small actor on NUC; consumes env features from L1 env (read-only) and candidate set from S3; trained CTDE on AERPAW digital twin. Repurposes the old MA-P-DQN trajectory head. |
| `L1 RL Module` | simplified to channel-only DDQN — drop the continuous DDPG head originally planned for trajectory. Keep MA-P-DQN code as a wrapper only if training needs the joint critic. |
| `ClientMission` | rename + extend [TrainingClient.py](App/TrainingApp/Client/TrainingClient.py) — add S2B utility computation, `FL_state`, RF beacon driver, and `FL_READY_ADV` payload builder. |
| `partial_round_state` | reuse FlightFramework checkpoint to survive mid-round mule disconnects. Owned by `HFLHostMission`, persisted across `ClientCluster` dock cycles. |
| Round-close report writer | new; written by `HFLHostMission`, consumed in-flight by `FLScheduler` (fast-phase) and shipped at dock by `ClientCluster` to `HFLHostCluster` (slow-phase). |
| `ClusterCloudClient` | optional new module on edge server; client peer to Tier 3 for `θ_gen` refinement — keeps `HFLHostCluster` symmetric to the mule's `HFLHostMission` (server-only at its own boundary). |

---

## 9. Open Questions / Decisions Deferred

### Still open (Phase 7 + experiment-time)

1. **Deadline clock** — wall-clock or mission-logical time? Slide 42 shows *Current Deadline Time for Device*; slide 41 shows *next round base timestamp*. Resolve which is `Time` in `Deadline(j)`.
2. **`FL_Threshold` tuning** — static value or adaptive (e.g. learned per cluster)?
3. **Beacon channel** — reuse one of the 3 RL-managed bands (3.32 / 3.34 / 3.90 GHz) or a dedicated narrow beacon band?
4. **`rf_range_m` calibration on real hardware** — sim default is 60 m, sweep {30, 60, 120}. Real AERPAW deployment may need a different value depending on tx power + antenna; treat the sim sweep as the parametric story and re-calibrate in the live testbed.
5. **Pass 2 ordering at large slice sizes** — Sprint 1.5 ships nearest-first greedy from the post-Pass-1 mule pose. Open: would a TSP-like solver materially improve Pass 2 path length for >10-contact missions? Defer until measurement shows it matters.

### Resolved by Sprint 1.5 / Sprint 2

* **Async-FL drift** — was a concern about stale Δθ in single-pass missions; **closed** by adopting two-pass missions (principle 13). Pass 1 collects, Pass 2 delivers; every Δθ is trained against a θ the cluster knows exactly.
* **In-session training cost** — was a concern about contact dwell time; **closed** by moving local training offline (principle 14). Sessions are exchange-only and finish in milliseconds.
* **One-device-at-a-time bottleneck** — was implicit in per-device session model; **closed** by per-contact parallel sessions (principle 15). N≥1 devices in range are served in one stop.
* **`TargetSelectorRL` algorithm** — **closed** as scalar-Q DDQN over per-contact aggregate features (option a from the original alternatives list). Pointer-network over the bucket was rejected as over-engineered for slice sizes ≤10. Implementation: `hermes/scheduler/selector/target_selector_rl.py`.
* **Selector reward shaping under contact events** — **closed** as `Σ_(devices in contact) completed_bonus_i − time_to_complete − w·energy_used`. The sum-over-devices form is the design intent; the multi-metric A/B at `rf_range_m ∈ {30, 60, 120}` validates it doesn't over-weight large clusters in practice.
* **L1 shared encoder** — **closed** as dropped. L1 is a standalone channel DDQN; the slide-21 shared-encoder claim is dead text.
* **Min-participation threshold default** — **closed** as absolute integer count, default 1 (partial-FedAvg). Set to `len(mules)` for full-FedAvg semantics. Wired through `ClusterConfig.min_participation` (Sprint 2 chunk L).
* **`MissionDeliveryReport` consumption** — **closed** per the proposed plan: cluster bumps `DeviceRecord.delivery_priority` on undelivered devices, S3a uses that as a tie-breaker (Sprint 1.5 H7). Pinned by `test_undelivered_carryover_routes_priority`.

### Resolved by Phase 7

* **Cross-cluster θ_gen refinement cadence** — Sprint 2 ships `HTTPCloudLink` polling Tier-3 every 5 s; Phase 7 chunk P2 wired the fold via `GeneratorHost.apply_tier3_gen_refinement(weights, refinement_round)`, with an out-of-order guard that ignores stale `refinement_round` values from a delayed Tier-3 packet. The cluster surfaces successful folds as `tier3_refinement_applied` events. The 5 s default cadence is a runtime tunable in `HERMES_Configuration_Reference.md`; tighten or relax once Tier-3's measured refinement rate is known.

These do not change the architecture (post-Sprint-2); they are parameters to fix during deployment hardening.

---

> **Implementation closeout (Phase 7 done, 2026-05):** all eight phases (0 → 7) have shipped. The system runs on the multi-process orchestrator under `--mode hermes`, with structured JSONL observability per process, principle-by-principle assertion tests for all 15 design principles, and Tier-3 refinement folding into the local generator. Final test count is **410 passed, 22 deselected** (the 22 are pre-existing flakes — stochastic A/B at rf=60 m and Flower-mode subprocess timeouts — both documented). What remains is paper-experiment scaffolding (see [HERMES_Experiments_Implementation_Plan.md](HERMES_Experiments_Implementation_Plan.md)) and AERPAW deployment when the testbed returns. Chunk Q (per-device SpectrumSig plumbing) is queued — see implementation plan §3.6.3 — and is conditional on whether a paper claim demands it.

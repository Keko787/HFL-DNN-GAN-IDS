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
**One job: Server to devices in-field.** Runs FL sub-sessions, partial-FedAvgs the mission slice, writes the round-close report. **No dock behavior here** — that is `ClientCluster`'s job.

| Responsibility | Role |
|---|---|
| `FL_OPEN` handshake with devices on arrival | Server |
| Push global `θ_disc` + synth samples DOWN to device | Server |
| Receive `Δθ_disc` UP from device + verify gradient receipts (checksum, completion, TTL) | Server |
| Arbitrate cross-mule races via device busy-flag (TTL-bounded) | Server |
| Weighted merge of `Δθ_disc` across mission slice → `partial_round_state` (FlightFramework) | FedAvg |
| Generate mission round-close report (authoritative participation ledger) | Writer |
| Feed fast-phase `Deadline(j)` update to `FLScheduler` in-flight | Feedback |
| Hand finalized `partial_aggregate` + `MissionRoundCloseReport` + `contact_history` to local `ClientCluster` at dock entry | Intra-NUC |

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
**One job: Who, locally.** Trains the discriminator, self-declares whether its weights are worth federating, and acts as FL client to `HFLHostMission` on contact.

| Responsibility | Stage |
|---|---|
| Local training of discriminator on local real traffic | Tier 1 |
| Post-round eligibility check: `Performance_score` (Acc, AUC, Loss) + `diversity_adjusted = cosine(θ_local, θ_global)·perf_discount` | S2B |
| `utility(i) = w₁·Performance_score + w₂·diversity_adjusted` — open flag when `> FL_Threshold` | S2B |
| Set FL state in `{busy, unavailable, FL_OPEN}` | S2B |
| On `FL_OPEN`, emit low-power RF beacon — **proximity only** (can't summon mule) | S1 |
| On contact: client handshake with `HFLHostMission`, receive `θ_disc + synth`, return `Δθ_disc` + meta-stats | FL client |
| Maintain `FL_READY_ADV` payload (perf_delta, diversity_proxy, payload_size, missed_count, idle_time) | S2B |

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

### 2.7 `TargetSelectorRL` — Intra-Bucket Selector *(new, sub-model of L2 S3)*
**Where the trajectory head went.** What was "trajectory" in MA-P-DQN was actually a **next-target selector** — which device or server to visit next. That decision belongs to *Who* (L2), not *How* (L1). This sub-model lives inside `FLScheduler` S3, queried only when a bucket has ≥2 candidates.

| Responsibility | Role |
|---|---|
| Inputs: candidate set within a single S3 bucket, last-known positions, SpectrumSig per device, current mule pose, energy budget, RF priors from L1 env state | features |
| Output: ordering over the candidate set (or single argmax DeviceID) | selector |
| Reward signal (training): −time_to_complete − energy_used + completed_session_bonus | RL training |
| Two distinct invocations: (a) device-selection during a mission round, (b) **server-selection** at end-of-mission (which edge server to dock at, if multiple are reachable) | dual-purpose |
| Trained centrally (CTDE), deployed as small actor on NUC alongside L1 channel actor | deployment |
| **Never** decides eligibility / gating / deadlines — those remain hard rules in S1 / S2A / S2B / S3 deadline math | scope |

This sub-model only ranks candidates that the deterministic stages already admitted. It cannot promote a gated-out device or override a deadline — it only fills the ordering gap inside a bucket.

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

```
[PRE-MISSION — at dock]
  (1) Mule docks at edge server; ClientCluster detects dock link.
  (2) ClientCluster ← HFLHostCluster (DOWN bundle):
        MissionSlice(ids, spectrum_sigs, positions)
        θ_disc_global
        SynthBatch
        Amendments(prev round)
  (3) ClientCluster verifies bundle integrity, then hands off intra-NUC:
        → FLScheduler:     MissionSlice + Amendments
        → HFLHostMission:  θ_disc_global + SynthBatch
  (4) FLScheduler folds Amendments → slow-phase Deadline(j) update.
  (5) FLScheduler builds session device list (Stage 1).

[MISSION — in field, per-device loop]
  loop while mission_time_left > 0 and devices_pending:
    (6) FLScheduler S3:
          bucket_classify(device_list) → {new, scheduled, beacon-active}
          if |bucket| > 1: TargetSelectorRL.rank(bucket, env_priors)
          target_waypoint = head(top_bucket)
        FLScheduler → L1: target_waypoint
    (7) L1 picks chan_idx (DDQN) for the link to that target;
        mule navigates mechanically to the device's last-known position.
    (8) On arrival, HFLHostMission issues FL_OPEN solicitation.
    (9) ClientMission (device) responds with FL_READY_ADV(perf_delta,
        diversity_proxy, payload_size, missed_count, idle_time).
    (10) Stage 2A gate:
          if FL_READY == False  →  mark device, skip, continue.
    (11) FL sub-session (HFLHostMission ↔ ClientMission):
          HFLHostMission pushes θ_disc + synth   →  ClientMission
          ClientMission trains locally (discriminator only)
          ClientMission returns Δθ_disc + meta   →  HFLHostMission
    (12) HFLHostMission:
          verify gradient receipt (checksum, TTL, completion)
          merge into partial_round_state (Partial FedAvg)
          append line to MissionRoundCloseReport
    (13) HFLHostMission → FLScheduler (intra-NUC):
          RoundCloseDelta{device_id, outcome ∈ {clean, partial, timeout}}
    (14) FLScheduler fast-phase Deadline(j) update:
          on-time → shorten next interval
          missed  → lengthen next interval
    (15) FLScheduler re-ranks device_list (S3):
          Rank 1: new devices awaiting distribution
          Rank 2: devices with shortened deadlines (on-time history)
          Rank 3: devices in session awaiting their deadline
    (16) Opportunistic FL_OPEN beacon?  If heard in-range, treat as
         an immediate transact (bonus) — else goto (6).

[POST-MISSION — at dock]
  (16.5) If multiple edge servers are reachable, FLScheduler calls
         TargetSelectorRL.select_server(reachable_servers, energy_budget)
         to pick the dock target; L1 picks channel for the dock link.
  (17) Mule docks; ClientCluster detects dock link.
  (18) HFLHostMission → ClientCluster (intra-NUC, dock-entry handoff):
         partial_aggregate + MissionRoundCloseReport + contact_history
  (19) ClientCluster → HFLHostCluster (UP bundle, dock link):
         partial_aggregate + MissionRoundCloseReport + contact_history
  (20) HFLHostCluster:
        cross-mule FedAvg of partial aggregates
        produce cluster Amendments
        push to Tier 3 for θ_gen refinement (via ClusterCloudClient)
        build next MissionSlice per mule
  (21) Goto (1) for next mission.
```

### 4.1 Exception paths

| Event | Handler | Action |
|---|---|---|
| Gradient receipt fails checksum / TTL | HFLHostMission | discard, mark outcome=`partial`, device gets missed-count bump |
| Mule disconnects mid-round | HFLHostMission | write `partial_round_state` checkpoint → resume on reconnect |
| Cross-mule race on same device | HFLHostMission | device busy-flag (TTL-bounded) wins; losing mule marks `timeout` |
| Dock link drops mid-UP | ClientCluster | retain `partial_aggregate` on-NUC; retry on next dock; emit stale-bundle warning to HFLHostCluster |
| Dock DOWN bundle fails verification | ClientCluster | refuse intra-NUC handoff; request re-dispatch; HFLHostMission re-uses prior θ_disc until resolved |
| No min_participation_threshold | HFLHostCluster | **deadline-aware aggregation** — aggregate whatever arrived, never stall |
| Mission slice collision across mules | HFLHostCluster | disjoint slicing enforced at dispatch; no runtime reconciliation needed |

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
| `Δθ_disc` | tensor | Discriminator gradient / delta; flows UP. |
| `θ_gen` | tensor | Generator weights; **never leaves Tier 2**. |
| `SynthBatch` | tensor[] | Fresh synthetic samples generated per round at Tier 2. |
| `SpectrumSig` | struct | Per-device RF fingerprint (bands, last-good SNR priors). |
| `MissionSlice` | list[DeviceID] | Disjoint per-mule subset of the cluster registry. |

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
| `partial_round_state` | tensor | Intermediate mission-scope FedAvg accumulator (FlightFramework). |
| `MissionRoundCloseReport` | list[Line] | Authoritative participation ledger; per-device outcome ∈ {clean, partial, timeout}. |
| `gradient_receipt` | struct | `{device_id, checksum, bytes_received, ttl_ok}`. |
| `device_busy_flag` | map[DeviceID, TTL] | Cross-mule race arbitration. |
| `RoundCloseDelta` | struct | Streamed to Scheduler: `{device_id, outcome, ts}`. |

### 6.4 `TargetSelectorRL` state *(sub-model of FLScheduler S3.5)*

| Name | Type | Definition |
|---|---|---|
| `candidate_set` | list[DeviceID \| ServerID] | Admitted members of one S3 bucket (or reachable servers at dock). |
| `features(j)` | vector | `{last_known_pos, SpectrumSig, distance, mule_energy, rf_prior_snr, on_time_rate}`. |
| `bucket_tag` | {new, scheduled, beacon-active, server} | Which decision context the selector is running in. |
| `action` | DeviceID \| ServerID | argmax over `candidate_set`. |
| `replay_buffer` | list | Offline — rewards from mission-scope metrics (completion, energy, throughput). |
| `actor_weights` | tensor | Deployed on NUC; trained CTDE on AERPAW digital twin. |

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

1. **Deadline clock** — wall-clock or mission-logical time? Slide 42 shows *Current Deadline Time for Device*; slide 41 shows *next round base timestamp*. Resolve which is `Time` in `Deadline(j)`.
2. **`FL_Threshold` tuning** — static value or adaptive (e.g. learned per cluster)?
3. **Beacon channel** — reuse one of the 3 RL-managed bands (3.32 / 3.34 / 3.90 GHz) or a dedicated narrow beacon band?
4. **Cross-cluster θ_gen refinement cadence** — on every Tier-2 round, or on a slower rhythm orchestrated by Tier 3?
5. **Min-participation threshold default** — fraction of slice vs. absolute count?
6. **`TargetSelectorRL` algorithm** — now that trajectory is discrete target selection, MA-P-DQN's hybrid-action justification is gone. Options: (a) plain DDQN over candidate set, (b) pointer-network style selector, (c) keep MA-P-DQN as a legacy wrapper with a dummy continuous head. Decide during RL training bring-up.
7. **Selector reward shaping** — what exact signals drive `TargetSelectorRL` training? Proposed: `−time_to_complete − w·energy + completed_session_bonus`; verify this doesn't bias against beacon-active devices.
8. **Does L1 keep a shared encoder?** — if L1 is DDQN-only, the slide-21 "shared encoder captures channel↔position co-dependency" claim no longer applies. Confirm L1 is a standalone channel DDQN.

These do not change the architecture; they are parameters to fix during implementation.

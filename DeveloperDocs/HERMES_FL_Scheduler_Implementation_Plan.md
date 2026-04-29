# HERMES FL Scheduler — Implementation Plan

**Companion to** [HERMES_FL_Scheduler_Design.md](HERMES_FL_Scheduler_Design.md).
**Audience:** engineer executing the build. Read the design doc first — this plan assumes its terminology.

---

## 0. Guiding Principles for the Build

1. **Deterministic before learned.** Ship every gate, deadline formula, and bucket rule as hard-coded logic first. Plug `TargetSelectorRL` in only after the deterministic path is green.
2. **Stubs at every tier-boundary.** Each of the four information flows (intra-NUC, RF, dock, cloud) gets a loopback stub before it gets a real transport. A missing radio never blocks a software milestone.
3. **One boundary at a time.** Build Mule↔Device (RF link) → Mule↔Server (dock) → Server↔Cloud. Do not start the next boundary until the previous passes an integration test.
4. **Reuse over rebuild.** Extend `HFLHost.py`, `TrainingClient.py`, and the `FlightFramework` strategy hooks wherever the design maps onto them. New code goes in new files — don't rewrite working Flower plumbing.
5. **Every new program owns one file at first.** Split into subpackages only after the interface contracts are stable.

---

## 1. Current-State Inventory

| Exists today | Path | Role in new design |
|---|---|---|
| `HFLHost.py` (141 LOC) | `App/TrainingApp/HFLHost/` | Becomes **`HFLHostCluster`** after dock-side endpoints are added. Today it runs a Flower server over the whole round; post-split it runs only cluster-scope cross-mule FedAvg. |
| `TrainingClient.py` (159 LOC) | `App/TrainingApp/Client/` | Becomes **`ClientMission`** — add S2B utility, FL state, beacon driver, `FL_READY_ADV` payload. |
| `FlightFramework/flight/` | strategies, jobs, nn, runtime | Reuse `strategies` for partial FedAvg on mule; reuse `jobs` for round-close report emission; `partial_round_state` checkpoint lives here. |
| `Config/SessionConfig/` | arg parsing, model load, strategy wiring | Add new arg-group functions for `HFLHostMission`, `ClientCluster`, `FLScheduler`, selector. |
| AERPAW channel-switch scripts | (deck slide 29) | Becomes the execution backend for L1's channel DDQN. |

**Not yet present** (to be created): `FLScheduler`, `HFLHostMission`, `ClientCluster`, `TargetSelectorRL`, L1 channel DDQN wrapper, dock transport module, mule-NUC process supervisor.

---

## 2. Target Repo Layout (after the build)

```
App/
  TrainingApp/
    HFLHost/
      HFLHostCluster.py          (was HFLHost.py, narrowed)
      HFLHostMission.py          NEW — mission-scope FL server
      ClientCluster.py           NEW — dock-handoff client on mule
    Client/
      ClientMission.py           (was TrainingClient.py, extended)
  SchedulerApp/                  NEW
    FLScheduler.py
    stages/
      s1_eligibility.py
      s2a_readiness.py
      s2b_flag.py
      s3_deadline.py
      s35_selector.py            wraps TargetSelectorRL
    selector/
      target_selector_rl.py      NEW — DDQN actor + features
      selector_train.py          training harness (offline, CTDE)
  L1App/                         NEW
    channel_ddqn.py              channel-only RL (was MA-P-DQN)
    channel_switch_runner.py     wraps AERPAW switch scripts
  MuleApp/                       NEW — process supervisor on NUC
    mule_main.py                 launches L1 + FLScheduler + HFLHostMission + ClientCluster
Config/
  SessionConfig/
    ArgumentConfigLoad.py        extend: parse_mission_args, parse_cluster_args, parse_scheduler_args
Transport/                       NEW
  intra_nuc.py                   in-process queues + callbacks
  rf_link.py                     Flower-over-RF wrapper (real or sim)
  dock_link.py                   bulk transfer for mule↔cluster bundles
  cloud_link.py                  Tier 2 ↔ Tier 3
DeveloperDocs/
  HERMES_FL_Scheduler_Design.md
  HERMES_FL_Scheduler_Implementation_Plan.md  (this file)
tests/
  unit/
  integration/
  e2e/
```

---

## 3. Phased Milestones

Each phase ends with a **demo** and a **Definition of Done**. Demos run on a single laptop — no mule, no radios — until Phase 6.

### Phase 0 — Foundation & Types (1 sprint)

**Goal:** skeleton + shared types + stub transports so every phase after this can import cleanly.

**Deliverables**
- `types/` module: `DeviceID`, `DeviceRecord`, `MissionSlice`, `FL_READY_ADV`, `RoundCloseDelta`, `MissionRoundCloseReport`, `ClusterAmendment`, dock bundle types.
- `Transport/intra_nuc.py`: typed in-process pub/sub queues for every intra-NUC edge listed in §6.9 of the design doc.
- `Transport/rf_link.py`: loopback implementation — one process pretends to be device, one pretends to be mule. Real RF replaces this in Phase 6.
- `Transport/dock_link.py`: loopback implementation.
- CI: pytest scaffolding, lint, typecheck. Every module starts with `from __future__ import annotations`.

**Definition of Done**
- A no-op "hello from each tier" round-trip passes through all four transports.
- `pytest tests/unit` green with coverage gating ≥ 80 %.

### Phase 1 — `HFLHostCluster` Split (1 sprint)

**Goal:** narrow existing `HFLHost.py` to cluster-scope responsibilities and expose dock endpoints.

**Tasks**
1. Move current `HFLHost.py` body into `HFLHostCluster.py`; preserve existing arg-parsing and strategy wiring (`fitOnEnd`, FedAvg).
2. Add:
   - `DeviceRegistry` (in-memory dict with persistence hook).
   - `make_mission_slice(mule_id) → MissionSlice` — disjoint per-mule.
   - `cross_mule_fedavg(partials: list) → aggregate`.
   - `dispatch_down_bundle(mule_id) → DownBundle`.
   - `ingest_up_bundle(bundle)`.
3. Dock endpoint: wire `HFLHostCluster` to `Transport/dock_link.py` (loopback).
4. Keep θ_gen + synth generator hosted here; reuse existing AC-GAN code.

**Definition of Done**
- Two fake mules (test doubles) dock one after the other, upload disjoint partials, and receive disjoint slices.
- Cross-mule FedAvg unit test: weighted merge matches hand-computed reference.
- Existing `HFLHost.py` regression tests (if any) still pass.

### Phase 2 — `HFLHostMission` + `ClientMission` + RF FL Session (2 sprints)

**Goal:** in-field FL round works end-to-end between one mule-side server and one device-side client, over the RF loopback.

**Tasks — `HFLHostMission`**
1. Flower server role, reusing `FlightFramework/flight/strategies` for FedAvg plumbing.
2. `fl_open_solicit()` / `fl_ready_adv()` handshake.
3. `push_model(θ_disc, SynthBatch)` → `submit_gradient(Δθ_disc, meta)`.
4. Gradient receipt verification (checksum, byte count, TTL).
5. Partial FedAvg accumulator → `partial_round_state` (reuse FlightFramework checkpoint idiom).
6. `MissionRoundCloseReport` writer.
7. Emit `RoundCloseDelta` per device to `intra_nuc.scheduler_bus`.
8. Device busy-flag (TTL-bounded dict) for cross-mule race arbitration.

**Tasks — `ClientMission`** (extend `TrainingClient.py`)
1. Add `FL_state ∈ {busy, unavailable, FL_OPEN}` state machine.
2. Post-round utility computation:
   - `Performance_score = f(Acc, AUC, Loss)`
   - `diversity_adjusted = cosine(θ_local, θ_global) · perf_discount`
   - `utility(i) = w₁·Performance_score + w₂·diversity_adjusted`
3. `FL_READY_ADV` payload builder.
4. RF beacon emitter (stubbed — calls a `radio.beacon(payload)` shim; real hardware later).
5. Keep existing discriminator training loop; wrap it in the new state machine.

**Definition of Done**
- Happy-path test: one mule, one device, full pre-round → in-field → post-round sequence. Model weights measurably update.
- Failure tests pass:
  - Checksum-fail path → outcome `partial`, missed-count bump.
  - TTL expiry → outcome `timeout`.
  - `FL_READY = False` → device skipped, no state mutation.
- Mid-round disconnect → `partial_round_state` checkpoint restored on reconnect (not restart).

### Phase 3 — `ClientCluster` Dock Handoff (1 sprint)

**Goal:** clean symmetric client on the mule-NUC side of the dock link.

**Tasks**
1. Dock-link detector (polls `dock_link.py` for availability).
2. State machine per §5.4 of the design doc: `AWAIT_DOCK → COLLECT → UP → DOWN → VERIFY → DISTRIBUTE → signal`.
3. Bundle integrity verifier (checksum + version sig).
4. Intra-NUC fan-out:
   - `MissionSlice + ClusterAmendments` → `FLScheduler` (slow-phase trigger).
   - `θ_disc + SynthBatch` → `HFLHostMission` (next-round model state).
5. Retry queue for UP failures; persist `partial_aggregate` across dock attempts.
6. Surface error states up to `mule_main.py`.

**Definition of Done**
- End-to-end: fake mule runs a full FL round in-field (Phase 2) then docks, uploads, downloads, and distributes the next bundle intra-NUC. Verifiable with a round-2 run that consumes the amendments.
- Failure tests:
  - Dock link drops mid-UP → retry on second dock.
  - DOWN verification fail → refuse handoff, request re-dispatch; `HFLHostMission` keeps prior θ_disc.

### Phase 4 — `FLScheduler` Deterministic Core (2 sprints)

**Goal:** full scheduler with all four stages as hard-coded logic; **no RL yet**.

**Tasks**
1. `stages/s1_eligibility.py`:
   - `eligible(i) = has_active_deadline(i) ∨ beacon_heard(i)`
2. `stages/s2a_readiness.py`:
   - On-contact `FL_READY` gate; never trusts remote state.
3. `stages/s2b_flag.py`:
   - Thresholding incoming `FL_READY_ADV` against `FL_Threshold`.
   - Note: S2B is *verified* on mule but *computed* on device — scheduler consumes the pre-computed utility.
4. `stages/s3_deadline.py`:
   - `Deadline(j) = Time + Deadline_Fulfilment − Idle_Time`
   - Fast-phase: consume `RoundCloseDelta` from `HFLHostMission`.
   - Slow-phase: consume `ClusterAmendments` from `ClientCluster` at dock.
   - Bucket classifier → `{new, scheduled-this-round, beacon-active}`.
5. `stages/s35_selector.py`:
   - **Placeholder**: returns `head(sorted(bucket, key=last_known_distance))`. RL replaces this in Phase 5.
6. `FLScheduler.main_loop`:
   - Wire stage pipeline, subscribe to `intra_nuc.scheduler_bus`.
   - Emit `target_waypoint` to L1.

**Definition of Done**
- Unit tests per stage with hand-computed expected values.
- Integration test: scheduler correctly re-ranks a 5-device list after an injected round-close delta.
- End-to-end with Phase 3: two-round mission runs, deadlines visibly adapt between rounds.

### Phase 5 — `TargetSelectorRL` + L1 Channel DDQN (2–3 sprints)

**Goal:** replace the S3.5 placeholder with a trained selector; stand up the L1 channel DDQN.

**Tasks — `TargetSelectorRL`**
1. Feature extractor per §6.4 of the design: `{last_known_pos, SpectrumSig, distance, mule_energy, rf_prior_snr, on_time_rate, bucket_tag}`.
2. Small DDQN actor (Keras/PyTorch — match what `FlightFramework.nn` expects).
3. Training harness (`selector_train.py`):
   - Offline CTDE training on AERPAW digital twin.
   - Reward: `−time_to_complete − w·energy + completed_session_bonus`.
4. Two inference modes:
   - `select_target(bucket, env)` — per-round, per-bucket.
   - `select_server(reachable_servers, energy_budget)` — end-of-mission.
5. Scope guard (principle #12): selector **cannot** promote gated-out devices, cannot re-order buckets. Add runtime assertion.

**Tasks — L1 Channel DDQN**
1. Wrap existing AERPAW channel-switch scripts under `L1App/channel_switch_runner.py`.
2. `channel_ddqn.py`: small DDQN, state vector ≈ 8 features (SNR per BS, distance, position). Discrete action ∈ {3.32, 3.34, 3.90 GHz} — reuse existing IDQN setup from slide 26.
3. Inference-only on NUC; training stays on GPU/AERPAW.
4. Expose `read_rf_prior()` as a read-only API for the selector (do not allow writes).

**Definition of Done**
- Replay-buffer replay shows selector converging on digital twin.
- A/B test on sim: selector-enabled scheduler beats distance-sorted placeholder on mission completion rate by a measurable margin (choose threshold: ≥ 5 %).
- Runtime assertion fires when any test tries to have the selector admit a gated-out device.
- L1 channel DDQN picks measurably better channels than a fixed-channel baseline on recorded AERPAW traces.

### Phase 6 — Full Integration on AERPAW Digital Twin (2 sprints)

**Goal:** all seven programs co-run on the digital twin with real Flower traffic, not loopback. **All modifications to `HFLHost.py` and `TrainingClient.py` are gated behind a CLI mode flag** (see §3.6.1 below) — legacy behavior remains the default and is bit-for-bit unchanged unless the new mode is explicitly selected.

**Tasks**
1. Replace `Transport/rf_link.py` loopback with Flower-over-real-radio.
2. Replace `Transport/dock_link.py` loopback with wired/high-bw mule↔server link.
3. `Transport/cloud_link.py`: outbound-only AERPAW→AWS pattern per slides 30–32 (reverse SSH or HTTP polling).
4. `MuleApp/mule_main.py`: process supervisor launches `L1 channel DDQN`, `FLScheduler`, `HFLHostMission`, `ClientCluster` with a single config.
5. Observability: structured logs for every state transition in each state machine; Grafana-ready metrics exporter.
6. Run the full 5-slide flow on the twin: registry pull → distribution round → mission round → dock → next round.
7. **Mode-gated cutover** (see §3.6.1): wire the new `--mode hermes` branch in both `HFLHost.py` and `TrainingClient.py`; verify legacy mode and hermes mode are independently runnable from the same binary.

**Definition of Done**
- Full mission runs end-to-end on twin with ≥ 2 mules, ≥ 6 devices, 1 cluster server, **invoked with `--mode hermes`**.
- The exact same binaries, invoked **without** `--mode hermes` (or with `--mode legacy`), reproduce the pre-Phase-6 Flower-only behavior with no observable diff in logs or model output.
- All five happy-path steps from design doc §4 reproduce.
- All exception paths from design doc §4.1 reproduce under fault injection.
- CI runs both modes on every PR (see §6.5).

#### 3.6.1 Compat Mode Switch — `--mode {legacy,hermes}`

**Rule:** every line of code added by Phase 6 to `HFLHost.py` and `TrainingClient.py` lives inside an `if mode == "hermes":` branch. The default value is `"legacy"`. Choosing the legacy path must execute the **identical** code path that exists today in `main()`.

**Why this matters**
- Zero-risk rollback. If `--mode hermes` misbehaves on AERPAW night-before-demo, omit the flag and the mules go back to running stock Flower.
- Bisectable bugs. A regression that appears only in hermes mode is provably caused by Phase 6 plumbing, not by something the legacy code already did.
- Parallel acceptance. Existing Flower-based experiments keep running unchanged for as long as anyone needs them; the new cluster-mode path graduates on its own schedule.

**Argparse contract** (added to `Config/SessionConfig/ArgumentConfigLoad.py`):

```python
# in parse_HFL_Host_args() and the matching client-side parser
parser.add_argument(
    "--mode",
    choices=["legacy", "hermes"],
    default="legacy",
    help="legacy = run the pre-Phase-6 Flower server/client unchanged; "
         "hermes = run via hermes.cluster.HFLHostCluster / ClientMission shims.",
)
```

**Shim pattern in `HFLHost.py`** (last ~20 lines of `main()`):

```python
def main():
    args = parse_HFL_Host_args()
    display_HFL_host_opening_message(args)
    model = build_model(args)            # unchanged
    dataset = load_dataset(args)         # unchanged

    if args.mode == "legacy":
        # ─── existing path, byte-for-byte unchanged ──────────────────
        _run_standard_federation_strategies(model, dataset, args)
        return

    # ─── new cluster-mode path, only reachable with --mode hermes ────
    from hermes.cluster import HFLHostCluster, DeviceRegistry
    from hermes.transport import build_dock_link_from_args
    from hermes.adapters import RealGeneratorHost   # ~15-LOC AC-GAN bridge

    registry = DeviceRegistry.load(args.registry_path)
    cluster = HFLHostCluster(
        registry=registry,
        generator=RealGeneratorHost(model),
        dock=build_dock_link_from_args(args),
        synth_batch_size=args.synth_batch_size,
        min_participation=args.min_participation,
    )
    cluster.serve_forever()
```

**Shim pattern in `TrainingClient.py`** (federated branch only):

```python
def main():
    args = parse_client_args()
    if args.mode == "legacy":
        # ─── existing Flower client path, untouched ──────────────────
        modelFederatedTrainingConfigLoad(args)
        return

    # ─── new mission-mode path, only with --mode hermes ──────────────
    from hermes.client import ClientMission
    from hermes.transport import build_rf_link_from_args
    mission = ClientMission(
        device_id=args.device_id,
        rf=build_rf_link_from_args(args),
        local_train=lambda θ: modelFederatedTrainingConfigLoad(args, init_weights=θ),
    )
    mission.serve_forever()
```

**Hard rules**
- No code outside the `if args.mode == "hermes":` block is allowed to import `hermes.*` from these two files. (Enforced by a CI grep test — see §6.5.)
- No silent default change. A future PR that flips `default="legacy"` to `default="hermes"` is its own decision, reviewed separately, and lands no earlier than Phase 7.
- Both modes must launch from the same wheel/conda env — no separate install path.

**Definition of Done for the mode switch**
- `python HFLHost.py <usual-args>` (no `--mode`) launches the legacy Flower server, identical logs to pre-Phase-6 builds.
- `python HFLHost.py <usual-args> --mode hermes` launches `HFLHostCluster` and accepts a dock UP from a fake mule.
- Same two assertions for `TrainingClient.py`.
- CI matrix runs the existing Flower smoke test under `--mode legacy` and the new cluster smoke test under `--mode hermes`, both green.

### Phase 7 — Hardening, Ops, Handoff (1 sprint)

**Tasks**
- Chaos tests: kill -9 each program, verify recovery paths.
- Configuration freeze: document every weight (`w₁, w₂, w₃, w₄`, `FL_Threshold`, selector reward weights).
- Deployment scripts for NUC + edge server.
- Runbook: first-boot, cluster registry seeding, dock secret rotation.
- Retire loopback transports from the build path (keep under `tests/`).

**Definition of Done**
- Cold-start on clean hardware completes a mission within 30 min of runbook steps.
- All 12 design-doc principles have a test asserting them.

---

## 4. Dependency Graph

```
  Phase 0 (types + stubs)
       │
       ├──► Phase 1  (HFLHostCluster split)
       │         │
       │         └──► Phase 3 (ClientCluster)  [needs Phase 1 dock endpoints]
       │
       └──► Phase 2  (HFLHostMission + ClientMission)
                 │
                 └──► Phase 4  (FLScheduler deterministic)  [needs round-close deltas from Phase 2]
                           │
                           └──► Phase 5  (TargetSelectorRL + L1 DDQN)
                                     │
                                     └──► Phase 6  (AERPAW integration)
                                               │
                                               └──► Phase 7 (harden)
```

**Critical path:** 0 → 2 → 4 → 5 → 6 → 7. Phases 1 and 3 can run in parallel to 2 and 4 respectively if staffed.

---

## 5. File-by-File Task Breakdown

| File | Phase | Action | Size target |
|---|---|---|---|
| `types/*.py` | 0 | new | small |
| `Transport/intra_nuc.py` | 0 | new | small |
| `Transport/rf_link.py` | 0, 6 | new (loopback → real) | medium |
| `Transport/dock_link.py` | 0, 6 | new (loopback → real) | medium |
| `Transport/cloud_link.py` | 6 | new | small |
| `HFLHostCluster.py` | 1 | extract from `HFLHost.py` + extend | ~400 LOC |
| `HFLHostMission.py` | 2 | new | ~500 LOC |
| `ClientMission.py` | 2 | extend `TrainingClient.py` | ~300 LOC |
| `ClientCluster.py` | 3 | new | ~300 LOC |
| `SchedulerApp/FLScheduler.py` | 4 | new | ~200 LOC |
| `SchedulerApp/stages/s1_eligibility.py` | 4 | new | ~80 LOC |
| `SchedulerApp/stages/s2a_readiness.py` | 4 | new | ~80 LOC |
| `SchedulerApp/stages/s2b_flag.py` | 4 | new | ~100 LOC |
| `SchedulerApp/stages/s3_deadline.py` | 4 | new | ~150 LOC |
| `SchedulerApp/stages/s35_selector.py` | 4 → 5 | new (placeholder → RL) | small wrapper |
| `SchedulerApp/selector/target_selector_rl.py` | 5 | new | ~300 LOC |
| `SchedulerApp/selector/selector_train.py` | 5 | new | ~400 LOC |
| `L1App/channel_ddqn.py` | 5 | new (reuse IDQN) | ~200 LOC |
| `L1App/channel_switch_runner.py` | 5 | wrap AERPAW scripts | small |
| `MuleApp/mule_main.py` | 6 | new supervisor | ~200 LOC |
| `Config/SessionConfig/ArgumentConfigLoad.py` | 1–5 | extend per phase | incremental |
| `Config/SessionConfig/ArgumentConfigLoad.py` | 6 | add `--mode {legacy,hermes}` to `parse_HFL_Host_args` and client parser; default `legacy` | +20 LOC |
| `App/TrainingApp/HFLHost/HFLHost.py` | 6 | wrap new path in `if args.mode == "hermes":` shim (see §3.6.1); leave legacy branch untouched | +25 LOC, 0 deletions |
| `App/TrainingApp/Client/TrainingClient.py` | 6 | wrap new path in `if args.mode == "hermes":` shim (see §3.6.1); leave legacy branch untouched | +20 LOC, 0 deletions |
| `hermes/adapters/real_generator_host.py` | 6 | new AC-GAN ↔ `GeneratorHost` Protocol bridge | ~50 LOC |

---

## 6. Test Strategy

### 6.1 Unit tests (per phase)
- Each stage (S1/S2A/S2B/S3) has its own test file with hand-computed expected values.
- Formulas (`utility(i)`, `Deadline(j)`) tested against reference tables.
- Transport loopbacks tested for ordering and loss semantics.

### 6.2 Integration tests
- **Phase 2 end:** one-mule one-device FL round.
- **Phase 3 end:** mule runs round, docks, round 2 consumes amendments.
- **Phase 4 end:** scheduler re-ranks after injected round-close delta.
- **Phase 5 end:** selector A/B beats deterministic placeholder on sim.

### 6.3 Fault-injection suite
Map every row in design-doc §4.1 to a test:
| Fault | Test |
|---|---|
| Bad gradient receipt | inject corrupted checksum → assert `outcome=partial`, missed count+1 |
| Mule mid-round disconnect | kill `HFLHostMission`; restart; assert `partial_round_state` restored |
| Cross-mule race | two mules target same device; assert busy-flag TTL wins |
| Dock UP drops mid-transfer | assert bundle held, retry succeeds next dock |
| DOWN bundle verify fail | assert handoff refused, re-dispatch requested |
| Under-threshold at aggregation | assert cluster aggregates anyway (deadline-aware) |

### 6.4 Principle-assertion tests
One test per design-doc §7 principle. Example skeletons:
- Principle 5: attempt to pass SNR to `FLScheduler.score(...)` → static type check fails.
- Principle 9: inspect any code path that persists θ_gen on mule → must not exist (repo-wide grep test).
- Principle 12: selector test harness tries to select a gated-out device → `SelectorScopeViolation` raised.

### 6.5 Mode-switch tests (Phase 6)
The `--mode {legacy,hermes}` gate is a load-bearing rollback path. It gets its own test bundle, all of which must stay green from Phase 6 onward through Phase 7.

| # | Test | What it pins down |
|---|---|---|
| M1 | `parse_HFL_Host_args([])` → `args.mode == "legacy"` | Default never silently flips. |
| M2 | `parse_HFL_Host_args(["--mode", "bogus"])` → `SystemExit` | Argparse `choices=` is enforced. |
| M3 | Subprocess: `python HFLHost.py <smoke-args>` → exits 0, log line `"running Flower server"` present, no `hermes.` import in process trace | Legacy path is byte-for-byte the pre-Phase-6 path. |
| M4 | Subprocess: `python HFLHost.py <smoke-args> --mode hermes` → exits 0, log line `"HFLHostCluster ready on dock"` present | New path is reachable end-to-end. |
| M5 | Same as M3 + M4 for `TrainingClient.py`. | Symmetric guarantee on the client side. |
| M6 | Repo-wide grep: `import hermes` may only appear inside an `if args.mode == "hermes":` guarded block within `HFLHost.py` and `TrainingClient.py` | Prevents accidental coupling that would make legacy mode pull in the new stack. |
| M7 | CI matrix: smoke tests run under both `--mode legacy` and `--mode hermes` on every PR | Catches mode-specific regressions immediately. |

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Slide 41/42 wording conflict on `Time` in `Deadline(j)` | Silent off-by-one in deadline adaptation | Fix before Phase 4 starts (design doc §9 Q1). |
| Selector over-fits to digital twin, behaves badly on real RF | Mission completion drops on first AERPAW run | Keep deterministic-placeholder path live behind a feature flag for Phase 6 rollout. |
| AERPAW inbound-traffic restriction (slide 31) blocks cluster ↔ cloud | Tier 3 refinement never runs | Use reverse-SSH or HTTP-polling backchannel per slide 32; don't design for direct inbound. |
| `HFLHost.py` split breaks current `fitOnEnd` workflow | Regression in AC-GAN training pipeline | Keep thin shim `HFLHost.py` that re-exports `HFLHostCluster` entrypoints for existing callers. |
| Beacon channel contention with FL data channel | Devices dropped mid-round when beacon burst collides | Early decision in Phase 2: dedicated beacon band vs. time-multiplexed on one of the 3 FL bands (design doc §9 Q3). |
| MA-P-DQN → DDQN refactor erases prior training work | Lost time | Keep MA-P-DQN code as a legacy wrapper; drop only after Phase 5 A/B confirms DDQN-only is sufficient. |
| Cross-mule slicing bug dispatches overlapping slices | Model corruption from double-submitted gradients | Property-based test in Phase 1: random registry sizes, assert `⋃ slices == registry` and `∀ i≠j: slice_i ∩ slice_j = ∅`. |
| Phase 6 cutover regresses existing Flower workflow | Stops AC-GAN training pipeline mid-experiment; existing users blocked | All cutover code lives behind `--mode hermes` (§3.6.1). Default stays `legacy`; CI matrix M7 in §6.5 keeps both paths green every PR. Rollback = drop the flag. |
| Operator forgets to pass `--mode hermes` on AERPAW demo and mules silently run legacy Flower | Demo looks broken (no scheduler/L1 activity) but no error raised | `HFLHostCluster.serve_forever()` logs a banner `MODE=hermes; cluster_id=...` on startup; legacy path logs `MODE=legacy`. Runbook (Phase 7) requires grepping for the banner before trusting a run. |
| Default flips to `hermes` prematurely (someone changes `default="legacy"` in argparse) | Silent breakage of legacy users | Test M1 in §6.5 pins the default; CODEOWNERS on `ArgumentConfigLoad.py` requires explicit reviewer for changes to that line. |

---

## 8. Open Decisions Blocking Code (from Design §9)

These must be resolved before the phases that depend on them:

| # | Decision | Blocks | Owner | Due |
|---|---|---|---|---|
| 1 | `Time` in `Deadline(j)` — wall-clock vs mission-logical | Phase 4 | — | before Phase 4 start |
| 2 | `FL_Threshold` — static vs adaptive | Phase 2 (ClientMission) | — | before Phase 2 mid |
| 3 | Beacon band — dedicated vs shared | Phase 2 (ClientMission beacon) | — | before Phase 2 mid |
| 4 | θ_gen refinement cadence (Tier 2 ↔ Tier 3) | Phase 6 | — | before Phase 6 start |
| 5 | Min-participation threshold — fraction vs absolute | Phase 1 (cluster FedAvg) | — | before Phase 1 end |
| 6 | Selector algorithm (DDQN / pointer-net / legacy MA-P-DQN) | Phase 5 | — | before Phase 5 start |
| 7 | Selector reward shape | Phase 5 | — | before Phase 5 start |
| 8 | L1 shared-encoder — keep or drop | Phase 5 | — | before Phase 5 start |

---

## 9. Suggested Staffing & Timeline

Assume 2 engineers, 2-week sprints.

| Phase | Sprints | Who |
|---|---|---|
| 0 | 1 | both |
| 1 | 1 | Eng A |
| 2 | 2 | Eng B + Eng A (parallel to 1 tail) |
| 3 | 1 | Eng A (after 1) |
| 4 | 2 | Eng B (after 2) |
| 5 | 3 | both (selector training is sequential but L1 channel can parallel) |
| 6 | 2 | both |
| 7 | 1 | both |

**Calendar estimate:** ~13 sprints = ~26 weeks (~6 months) with 2 engineers. Critical path is 9 sprints if Phases 1/3 run fully parallel to 2/4.

---

## 10. Demos per Phase

| Phase | Demo script |
|---|---|
| 0 | Print-statement trace of a message round-tripping across all four transports. |
| 1 | Two fake mules → distinct slices → cross-mule FedAvg converges on a toy dataset. |
| 2 | One mule, one device, one FL round; show updated `θ_disc` weights diff. |
| 3 | Two mission rounds with dock in between; show amendments observed in round 2's deadline shift. |
| 4 | Inject three `RoundCloseDelta`s; watch re-ranked output change live. |
| 5 | Side-by-side: placeholder vs. selector on digital twin; completion-rate bar chart. |
| 6 | Full 2-mule 6-device mission on AERPAW twin; Grafana dashboard live. |
| 7 | Cold-start runbook timed; fault-injection suite green. |

---

## 11. Exit Criteria (Program-wide)

The implementation is **Done** when:

1. All 7 programs run unattended through a full mission on the AERPAW digital twin **under `--mode hermes`**.
2. The same binaries, invoked **without** `--mode hermes`, still pass the legacy Flower smoke test (proves rollback is real).
3. All 12 design principles have passing assertion tests.
4. All exception paths in design §4.1 have passing fault-injection tests.
5. Selector-enabled path beats deterministic placeholder on completion-rate A/B.
6. Runbook cold-start completes inside 30 min on clean hardware **and explicitly documents the `--mode` choice**.
7. All eight open decisions in §8 of this plan are resolved, documented, and reflected in config.
8. Mode-switch test bundle (§6.5, M1–M7) is green on the release commit.

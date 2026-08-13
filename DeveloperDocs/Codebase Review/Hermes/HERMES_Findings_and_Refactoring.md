# HERMES — Findings and Refactoring Plan

**Status:** Review only. No project code has been modified.
**Scope:** `hermes/` (69 modules, 14,147 LOC)
**Companions:** [`../00_Critical_Problem_Areas.md`](../00_Critical_Problem_Areas.md) ·
[`../01_Refactoring_Strategy.md`](../01_Refactoring_Strategy.md) ·
[`HERMES_Production_Code.md`](HERMES_Production_Code.md) ·
[architecture](../../architecture%20documents/Hermes/HERMES_Architecture.md)

---

## 0. Verdict

`hermes/` is the strongest code in the repository and should be treated as the reference
standard for everything else. Its transport abstraction, pure-function scheduler stages,
dependency injection, and observability-by-construction are all correct choices, and its
512-test suite runs in 12 seconds because of them.

Its defects fall into three clean groups:

1. **The transport layer was hardened asymmetrically.** The dock link got the right
   socket-lifetime treatment and the RF link did not. This is the only defect in the
   subsystem that produces silent, total mission failure in the field.
2. **Concurrency was written for N≈3 and never revisited.** Unbounded thread spawning,
   per-thread joins, and a single-threaded cluster loop are all fine at the validated
   1×2×5 topology and all fail at 10×.
3. **Persistence and layering were deferred with explicit "Phase 6" markers**, and Phase 6
   never arrived. The markers are still in the code, two sprints past the sprints that
   were supposed to remove them.

Nothing here needs a rewrite. Every item below is a bounded, testable change.

---

## 1. Findings by severity

Cross-cutting findings ([A-01](../00_Critical_Problem_Areas.md#a-01),
[A-03](../00_Critical_Problem_Areas.md#a-03), [S-01](../00_Critical_Problem_Areas.md#s-01))
are detailed in the main register. This document adds HERMES-specific mechanism and the
per-item plan.

### H-01 (P0) — RF link socket lifetime

**Mechanism.** `tcp_rf_link.py:375` sets `conn.settimeout(self._send_timeout_s)` — a
single timeout governing *both* directions on the device socket. The reader
(`_reader_loop` → `recv_message` → `recv_exactly`) therefore inherits a 30 s read
deadline. `socket.timeout` subclasses `OSError`, so `recv_exactly`'s handler converts it
to `WireError`, the reader breaks, and `_drop_device` closes the socket.

The client side is symmetric at 60 s (`:497`). Once the client's reader breaks it sets
`_closed`, and `ClientMission.serve_once` short-circuits through `_raise_if_closed()` →
`RFLinkError` → `return None` on every subsequent call. `DeviceService.run` has no sleep
on the `None` path, so the process spins a core indefinitely with no reconnect.

**Why the sibling transport is right.** `tcp_dock_link.py:283-299`:

```python
# Reader stays blocking-forever — mules can sit idle between
# missions on a long-lived dock connection.
conn.settimeout(None)
# S2-M3: bound the dock SEND-timeout via SO_SNDTIMEO …
conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, tv)
```

The same reasoning applies verbatim to devices between contacts. The RF link's docstring
even lists `send_timeout_s` as *"bound on per-message `sendall`"* — the intent was
send-only; the implementation is both.

**Secondary defect.** The `SO_SNDTIMEO` value is packed as `struct.pack("ll", …)`, a
Linux `timeval`. Windows expects a 4-byte DWORD of milliseconds, so `setsockopt` raises
and is swallowed by `except OSError` — the dock send timeout is silently absent on
Windows, the platform this repository is developed on.

**Plan → [R-10](../01_Refactoring_Strategy.md#r-10).** Extract a platform-correct
`_sockopt.py`; `settimeout(None)` on RF reader sockets; heartbeat frame; bounded
reconnect with backoff on the client; sleep on the device's idle path. Regression test:
45-second inter-contact gap.

---

### H-02 (P1) — Contact fan-out is unbounded and joined per-thread

**Mechanism.** `host_mission.py:519-527` and `:696-703`:

```python
for did in contact_devices:
    t = threading.Thread(target=_device_worker, args=(did, adv), daemon=True)
    t.start(); threads.append(t)
for t in threads:
    t.join(timeout=self.session_ttl_s * 2.0)
```

Three independent problems:

| | Problem | Consequence |
|---|---|---|
| a | One OS thread per device, no cap | S3a contact size is a tunable (`rf_range_m=120` → "most of a slice in one contact"); thread count follows it |
| b | Per-thread timeout, not a wall-clock deadline | worst case `N × 2 × ttl`; at N=20, ttl=30 s → 20 minutes for one contact |
| c | Abandoned `daemon=True` workers keep references to `_report` / `_accepted` | a late worker can append a `MissionRoundCloseLine` or a `GradientSubmission` **after** `close_round` ran `partial_fedavg` — silently lost, or folded into the next round's ledger |

(c) is the subtle one. `_record_outcome` (`:810-848`) takes `self._lock` and appends to
`self._report`; `close_round` (`:166-209`) sets `self._report = None` under the same lock.
A worker arriving between those two moments returns early — its outcome vanishes with no
log. A worker arriving after the *next* `open_round` appends to the *new* round's report.

**Plan → [R-11](../01_Refactoring_Strategy.md#r-11).** Bounded `ThreadPoolExecutor`,
single wall-clock deadline via `concurrent.futures.wait`, and a monotonic round epoch
checked inside `_record_outcome` so a stale worker's write is dropped **with a warning**
rather than silently.

---

### H-03 (P1) — `run_contact` / `deliver_contact` are ~90 % duplicated

187 lines (complexity 35) and 140 lines (complexity 32) sharing an identical skeleton:
pass guard → non-empty guard → θ snapshot under lock → broadcast → drain
`_misrouted_advs` → gather loop → thread fan-out → join → return. Only the per-device
worker body differs.

This is why H-02 has to be fixed twice, and why the H1 "misrouted advertisement" fix
appears twice with identical comments (`:387-404` and `:611-624`).

**Root cause worth naming.** The duplication exists because `recv_ready_adv` is a
*single unaddressed queue* (`tcp_rf_link.py:135`, `self._ready_q: Queue[FLReadyAdv]`).
There is no way to ask for "the advert from device X", so each contact must drain the
shared queue, keep what it wants, and stash the rest for the next contact. The stash is
a correct workaround; a per-device advert queue (mirroring the existing per-device
`_gradient_q` / `_delivery_ack_q`) would remove the need for it entirely.

**Plan → [R-11](../01_Refactoring_Strategy.md#r-11)** for the immediate de-duplication.
Consider a follow-up adding `_ready_q` per device so `_misrouted_advs` can be deleted;
that is a transport-level change and should be separately scoped.

---

### H-04 (P0) — `wait_for_dock(timeout=None)` is uncancellable

`client_cluster.py:215-226` is a `while True` polling `dock.is_available()` with
`time.sleep(self.dock_poll_interval_s)` and no exit condition when `timeout is None`.
`MuleSupervisor` calls it that way at `mule_main.py:306` and `:425`.

`MuleService` owns a `threading.Event` stop flag and checks it between missions
(`mule.py:211`), but the supervisor never receives it. SIGTERM during an inter-pass dock
wait is therefore absorbed until `shutdown_all`'s 15 s timeout expires and the mule is
`SIGKILL`ed — discarding the Pass-1 aggregate and the delivery-report carryover that
were about to upload.

**Plan → [R-12](../01_Refactoring_Strategy.md#r-12).** Thread the stop event through;
wait on `Event.wait(interval)` rather than `sleep`; replace `timeout=None` with a
configured bound so an unreachable cluster produces a logged error.

---

### H-05 (P1) — Cluster service loop is the throughput ceiling

`cluster.py:297-375` handles one UP per iteration and performs, inline and serially:
ingest → cross-mule FedAvg → close round → **TensorFlow evaluation over the held-out
set** → M × (`slice_for` O(N) + per-device registry lookups + θ copy + synth batch +
SHA-256 over the whole model + socket send).

Everything else in the dock path is already concurrent — `TCPDockLinkServer` runs one
reader thread per mule. The serialization is purely the service loop's shape.

Also at `:300`: `except Exception: up = None` swallows every failure from `recv_up`,
including programming errors, and continues as though nothing arrived.

**Plan → [R-24](../01_Refactoring_Strategy.md#r-24).** Move DOWN dispatch and model
evaluation onto a bounded pool; keep aggregation serialized; index the registry by
`assigned_mule`; narrow the exception handler to the link's own error types.

---

### H-06 (P0) — No persistence anywhere

`DeviceRegistry.save()` returns `None`; `DeviceRegistry.load()` returns an empty
registry. Both are marked *"Phase-6 hook. Intentionally a no-op until then."* Phase 6 is
recorded as closed in the README.

The blast radius is larger than the registry: a cluster restart yields empty
`MissionSlice`s for every mule, empty contact queues, and permanent
`"no submissions to aggregate"` failures with no operator-visible cause. All scheduler
adaptation (`on_time_history`, `missed_history`, `delivery_priority`,
`deadline_fulfilment_s`) is lost.

**Plan → [R-23](../01_Refactoring_Strategy.md#r-23).** `RegistryStore` Protocol,
`InMemoryStore` default (today's behaviour), `SQLiteStore` for deployment, snapshot on
round close, `registry_restored` event on load.

---

### H-07 (P1) — Observability violates its own stated contract

`events.py:108` calls `json.dumps` outside the `try` that the module docstring promises
guards every emit. `_coerce` handles `tuple` and `Path` but not numpy scalars — the exact
case its own docstring claims is caught. A single `emit(..., auc=np.float64(...))`
anywhere takes down the emitting service loop.

**Plan → [R-13](../01_Refactoring_Strategy.md#r-13).** One-line move plus numpy-aware
coercion and a `default=repr` fallback.

---

### H-08 (P1) — `hermes.processes` imports `experiments.exp4`

`cluster.py:113,140,461` and `device.py:89` reach up into the paper harness for
`load_weights`, `load_xy`, `evaluate_theta`, `make_local_train_fn`. The comment at
`device.py:86-88` acknowledges the inversion and constrains it to "a spawned device
subprocess whose CWD is the repo root".

Consequence: `hermes/` cannot be packaged, installed, or deployed independently — and the
`GeneratorHost` / `LocalTrainFn` Protocols that exist precisely to avoid this are bypassed
on the cluster side.

**Plan → [R-17](../01_Refactoring_Strategy.md#r-17).** Resolve the provider from a
config-supplied entry-point string; move the concrete implementations into
`hermes_adapters/`.

---

### H-09 (P2) — Round-robin assignment ignores position and is unstable

`device_registry.py:157-165` sorts by `(not is_new, device_id)` and assigns
`mule_list[i % len(mule_list)]`. `DeviceRecord.last_known_position` is never consulted.

The sort key also makes assignment unstable: when one device's `is_new` flips to `False`,
its index changes and every device after it shifts a slot — reassigning roughly half the
fleet and invalidating each moved device's cached `DeviceSchedulerState`,
`delivery_priority` carryover, and on-time history on the mule.

Currently masked because `ClusterService` overrides `assigned_mule` from config after
seeding and never calls `rebalance` again — so `HFLHostCluster.rebalance_for` is dead in
production while being the only path that matters at scale.

**Plan → [R-25](../01_Refactoring_Strategy.md#r-25).** Spatial partitioning +
rendezvous-hash tie-break, behind a strategy flag defaulting to today's behaviour.

---

### H-10 (P2) — Small hygiene items

| Item | Location | Fix |
|---|---|---|
| `Sequence` used in annotations, never imported | `host_mission.py:344,568` | add to the `typing` import |
| `self.cluster._cluster_round` bypasses the locked public property | `cluster.py:346,350` | use `cluster.cluster_round` |
| `TCPRFLinkClient._lock` created and never used | `tcp_rf_link.py:492` | either guard `_send_with_emulator` with it (correct — concurrent `send_ready_adv` + `send_gradient` would interleave frames) or delete it |
| `orchestrator.start_devices` ends with `time.sleep(0.3)` | `orchestrator.py:336` | replace with a device-readiness barrier, as the mule already has via `wait_for_devices` |
| 4 × `except Exception: pass` in shutdown paths | `cluster.py:492,497,503`, `device.py:213,219`, `mule.py:304,308,314`, `orchestrator.py:368,380,402` | narrow to expected types; log the rest at `WARNING` |
| `TrialRunner._status_col` is an identity function | `experiments/runner/runner.py:182` | inline and delete |

The `_lock` one is worth a second look: `TCPRFLinkClient` defines `self._lock =
threading.RLock()` and never acquires it. `_send_with_emulator` writes a full frame with
`sock.sendall` from whatever thread calls it. Today `ClientMission` is driven from one
thread, so no interleaving occurs — but nothing enforces that, and an interleaved
`sendall` would corrupt the framing irrecoverably (the receiver would read a valid header
followed by another message's bytes). Adding the lock costs nothing and closes a latent
class of bug.

---

## 2. What not to change

| Keep | Why |
|---|---|
| `RFLink` / `DockLink` / `CloudLink` ABCs | The reason the subsystem is testable at all. |
| Pure-function scheduler stages + injected `now_fn` | Deterministic, independently testable, already the right shape. |
| `partial_fedavg` / `cross_mule_fedavg` as free functions | Validate up front, float64 accumulate, cast back. Correct. |
| The two-pass mission design | Structurally eliminates async-FL drift. The best idea in the project. |
| numpy DDQN | ~200 ops per bucket; a framework would dominate the cost and add a heavy dep to `hermes/`. |
| Wire framing (magic + version + length cap) | Correct defensive design; changing it is a compat break for no gain. |
| Pickle on the LAN transports | Justified in-comment, bounded threat model. Fix the *cloud* link only ([S-01](../00_Critical_Problem_Areas.md#s-01)). |
| Fix-provenance comments (`H1`, `S2-M3`, `L-H2`, `EX-4.2`) | Institutional memory. Carry them through any move. |

---

## 3. Ordered work plan for this subsystem

| # | Item | Fixes | Effort | Risk | New test |
|---|---|---|---|---|---|
| 1 | RF socket lifetime + heartbeat + reconnect + device backoff | H-01 | 2 d | Low | 45 s idle-gap integration test |
| 2 | Unify contact exchange; bounded pool; deadline; round epoch | H-02, H-03 | 3 d | Low | N=32 contact bounded-wall-clock test; stale-worker rejection test |
| 3 | Cancellable dock wait | H-04 | 0.5 d | Very low | SIGTERM-during-dock-wait test |
| 4 | Observability guard + numpy coercion; encapsulation hygiene | H-07, H-10 | 0.5 d | Very low | `emit(np.float32)` does not raise |
| 5 | Provider entry-point indirection → `hermes_adapters/` | H-08 | 3 d | Medium | `grep -rn "import experiments" hermes/` is empty |
| 6 | `RegistryStore` Protocol + SQLite backend | H-06 | 3 d | Medium | kill-and-restart cluster preserves history |
| 7 | Concurrent cluster loop + registry index | H-05 | 4 d | Medium | throughput scales to 8 mules |
| 8 | Locality-aware stable assignment | H-09 | 3 d | Medium | rebalance moves ≤1 device on `is_new` flip |

Items 1–4 (~6 days) remove every P0 in this subsystem and are all behaviour-preserving in
the currently tested regime.

**Reference implementations:** [`HERMES_Production_Code.md`](HERMES_Production_Code.md).

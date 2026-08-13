# HiFINS — Critical Problem Areas

**Status:** Review only. No project code has been modified.
**Date:** 2026-08-10 · **Baseline:** `main` @ `35e2eeb`
**Companion:** [`01_Refactoring_Strategy.md`](01_Refactoring_Strategy.md) ·
[architecture review](../architecture%20documents/System_Architecture_Overview.md)

---

## How to read this

Every finding below was verified against source. Each carries a file:line citation and
the specific mechanism of failure — not a code smell, a consequence. Findings are graded:

| Grade | Meaning |
|---|---|
| **P0** | Produces wrong results, data loss, or hangs in normal operation. Fix before the next paper run or deployment. |
| **P1** | Blocks scale, deployment, or reproducibility. Fix this quarter. |
| **P2** | Compounding maintenance cost. Fix opportunistically or during related work. |

**Category legend:** `ARCH` architecture decision · `DUP` duplicate logic ·
`PERF` performance · `SCALE` scalability · `MAINT` maintainability · `CORR` correctness ·
`SEC` security · `ENV` environment/deployment

---

## Summary

| ID | Grade | Cat | Finding | Where |
|---|---|---|---|---|
| [C-01](#c-01) | **P0** | CORR | 21 AC-GAN CLI hyperparameters are declared and never read | `ArgumentConfigLoad.py:76-96` |
| [P-02](#p-02) | **P0** | PERF/SCALE | RF link drops any device idle > 30 s; no keepalive, no reconnect; device then hot-spins | `tcp_rf_link.py:375` |
| [C-02](#c-02) | **P0** | CORR | 4 advertised `--model_type` values crash with `AttributeError: 'NoneType'` | `modelCentralTrainingConfigLoad.py:70-77` |
| [A-03](#a-03) | **P0** | SCALE | All authoritative state is in-process; `DeviceRegistry.save()` is a no-op | `device_registry.py:210-219` |
| [P-03](#p-03) | **P0** | PERF | `wait_for_dock(timeout=None)` is an uncancellable busy-poll — mule hangs forever on cluster loss | `client_cluster.py:215` |
| [C-03](#c-03) | **P1** | CORR/ARCH | `--mode hermes` is a stub on both binaries; README presents it as the HERMES path | `TrainingClient.py:192` |
| [A-01](#a-01) | **P1** | ARCH | Two disconnected stacks; the only real bridge points the wrong way | `cluster.py:113`, `device.py:89` |
| [E-01](#e-01) | **P1** | ENV | `requirements_core.txt` is a workstation `pip freeze`; 4 pins cannot install | `AppSetup/requirements_core.txt` |
| [A-06](#a-06) | **P1** | ARCH/ENV | No packaging metadata; `sys.path` hacks in 16 modules; CWD-dependent everything | repository root |
| [S-01](#s-01) | **P1** | SEC | `pickle.loads` of an unauthenticated remote HTTP body | `cloud_link.py:171,242` |
| [A-09](#a-09) | **P1** | SCALE | Single-threaded cluster service loop is the system throughput ceiling | `cluster.py:297-375` |
| [P-01](#p-01) | **P1** | PERF/SCALE | Unbounded thread-per-device fan-out with per-thread joins | `host_mission.py:519-527` |
| [A-04](#a-04) | **P1** | ARCH/DUP | Config-as-code: 47 trainer classes, 21.3 K LOC, one loss function copied 24× | `Config/ModelTrainingConfig/` |
| [M-02](#m-02) | **P1** | MAINT | CWD-relative dataset paths contradict the documented invocation | `Config/DatasetConfig/**` |
| [Q-02](#q-02) | **P1** | CORR | Observability can crash the process it instruments | `events.py:108` |
| [D-02](#d-02) | **P2** | DUP | `CANGANCentralTrainingConfig.py` is a 1,900-line byte-clone | `Config/…/FullModel/` |
| [D-01](#d-01) | **P2** | DUP | `run_contact` / `deliver_contact` ~90 % duplicated | `host_mission.py:342,566` |
| [A-05](#a-05) | **P2** | MAINT | 41–42 positional-parameter dispatchers | `modelCentralTrainingConfigLoad.py:49` |
| [A-07](#a-07) | **P2** | ARCH | 5.8 K LOC vendored `FlightFramework` with zero importers | `FlightFramework/` |
| [A-08](#a-08) | **P2** | SCALE | Round-robin device→mule assignment ignores geography | `device_registry.py:159-165` |
| [M-01](#m-01) | **P2** | MAINT | 863-line / complexity-170 function | `analysis/exp3.py:342` |
| [E-02](#e-02) | **P2** | ENV | `docker-compose.yml` hardcodes another project's absolute Windows paths | `DockerSetup/docker-compose.yml` |
| [E-03](#e-03) | **P2** | MAINT | Hardcoded `192.168.129.x` server table in two places | `TrainingClient.py:124-133` |
| [Q-01](#q-01) | **P2** | MAINT | `Sequence` used in annotations, never imported | `host_mission.py:344` |
| [Q-03](#q-03) | **P2** | MAINT | Private-attribute access across objects despite a public property | `cluster.py:346,350` |
| [T-01](#t-01) | **P2** | MAINT | Test suite is not green on `main`; one stale assertion | `test_experiments_calibration.py:36` |
| [G-01](#g-01) | **P2** | ENV | Untracked, un-ignored directories a `git add -A` would commit | `.claude/`, `hermes_rl/`, `.idea/` |

---

## P0 — Fix before the next paper run or deployment

### C-01 — 21 AC-GAN CLI hyperparameters are declared and never read {#c-01}

**Category:** CORR · **Grade:** P0

`Config/SessionConfig/ArgumentConfigLoad.py:76-96` declares 21 AC-GAN training flags:

```python
parser.add_argument("--AC_disc_learning_rate", type=float, default=0.00001, …)
parser.add_argument("--AC_gen_learning_rate",  type=float, default=0.00003, …)
parser.add_argument("--AC_d_to_g_ratio",       type=int,   default=1,       …)
# …18 more: decay_steps, decay_rate, staircase, beta_1, beta_2,
#   valid/fake smoothing factors, attack/benign/validity/class weights
```

Repository-wide search for any of these names returns **exactly one hit each — the
declaration itself.** Nothing downstream reads `args.AC_*`.

The values that actually train the model are literals inside the trainer:

```python
# ACGANCentralTrainingConfig.py:109-118
lr_schedule_gen  = ExponentialDecay(initial_learning_rate=0.00012, decay_steps=10000,
                                    decay_rate=0.98, staircase=False)
lr_schedule_disc = ExponentialDecay(initial_learning_rate=0.00007, decay_steps=10000,
                                    decay_rate=0.98, staircase=False)
self.gen_optimizer  = Adam(learning_rate=lr_schedule_gen,  beta_1=0.5, beta_2=0.999, clipnorm=1.0)
self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999, clipnorm=1.0)
```

The D:G ratio is worse: it is a **default argument of `fit`**
(`ACGANCentralTrainingConfig.py:816`, `def fit(self, X_train=None, y_train=None, d_to_g_ratio=3)`)
and `TrainingClient.main` calls `client.fit()` with no arguments — so the effective
ratio is always 3:1 while `--AC_d_to_g_ratio` advertises a default of 1.

**Failure scenario.** A researcher runs
`TrainingClient.py --model_type AC-GAN --AC_gen_learning_rate 0.001 --save_name lr_high`
and then `--AC_gen_learning_rate 0.00001 --save_name lr_low`. Both runs train with
`0.00012`. The two checkpoints differ only by dataset-sampling noise, and any
lab-notebook or paper table recording "generator LR = 0.001 / 0.00001" is wrong. There
is no warning, no log line, and no test that would catch it.

**Why this is P0.** It silently invalidates hyperparameter results. Nothing else in this
register can produce a wrong number that looks right.

---

### P-02 — RF link drops any device idle > 30 s, then the device hot-spins {#p-02}

**Category:** PERF / SCALE · **Grade:** P0

`hermes/transport/tcp_rf_link.py:375` sets **one** timeout for both directions on every
registered device socket:

```python
conn.settimeout(self._send_timeout_s)      # default 30.0
```

`_reader_loop` then calls `recv_message(conn)` with no explicit timeout, so the socket's
30 s applies to the **read**. After 30 s of silence:

`socket.timeout` → `OSError` → caught by `recv_exactly`'s `except OSError` → `WireError`
→ `break` in `_reader_loop` → `_drop_device(device_id)`.

The device side is the same with a 60 s timeout (`tcp_rf_link.py:497`). When its reader
breaks it sets `self._closed`, after which:

```python
# ClientMission.serve_once →
self.rf.recv_open_solicit(...)  # → _raise_if_closed() → RFLinkError, immediately
except RFLinkError: return None
```

and `DeviceService.run` (`hermes/processes/device.py:184-206`) loops on that `None` with
**no sleep, no backoff, and no reconnect** — a 100 % CPU spin on the edge device until
the process is killed.

**Failure scenario.** A mule flies a 90-second leg between two contact clusters. Every
device in the *next* cluster has been silent for 90 s, so the mule dropped their sockets
at t+30 s. The Pass-1 broadcast reaches nobody; `run_contact` records `TIMEOUT` for
every device; `close_round` raises `PartialFedAvgError("no submissions to aggregate")`;
the supervisor logs an empty round and carries θ forward. The mission produces zero
federated learning, and every device is now spinning a core.

**Why the tests do not catch it.** Every integration test completes in well under 30
seconds.

**The fix already exists in the sibling transport.** `tcp_dock_link.py:283-299` handles
exactly this case correctly and says so:

```python
# Reader stays blocking-forever — mules can sit idle between
# missions on a long-lived dock connection.
conn.settimeout(None)
# S2-M3: bound the dock SEND-timeout via SO_SNDTIMEO …
conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, tv)
```

The RF link was never updated to match.

*Secondary:* the `SO_SNDTIMEO` value is packed as `struct.pack("ll", …)` (a Linux
`timeval`). Windows expects a 4-byte DWORD of milliseconds, so the call raises and is
swallowed by `except OSError` — the dock send timeout is silently absent on Windows.

---

### C-02 — Four advertised `--model_type` values crash {#c-02}

**Category:** CORR · **Grade:** P0

`ArgumentConfigLoad.py:62-65` advertises eight model types. Four of them cannot run:

```python
# modelCentralTrainingConfigLoad.py:70-77
elif model_type == "NIDS-IOT-Binary":            client = None
elif model_type == "NIDS-IOT-Multiclass":        client = None
elif model_type == "NIDS-IOT-Multiclass-Dynamic": client = None
# …and CANGAN has no branch at all — lines 158-166 are entirely commented out
```

`client` stays `None`, is returned, and `TrainingClient.main` calls `client.fit()` →
`AttributeError: 'NoneType' object has no attribute 'fit'` after the full dataset load
and model construction have already run.

**Failure scenario.** A user follows `--help`, selects `--model_type CANGAN`, waits
several minutes for 400 K rows to load and preprocess, and gets an unhandled
`AttributeError` with no indication that the mode was never implemented. The `CANGAN`
choice is also offered for `--dataset` and `--dataset_processing`, and
`datasetLoadProcess.py:77` has a matching commented-out branch — so the failure is
consistent across three flags.

---

### A-03 — All authoritative state is in-process memory {#a-03}

**Category:** SCALE · **Grade:** P0

`DeviceRegistry` is described in its own module docstring as "single source of truth for
cluster-scope devices". Its persistence is:

```python
# hermes/cluster/device_registry.py:210-219
def save(self, path: str) -> None:
    """Phase-6 hook. Intentionally a no-op until then."""
    return None

@classmethod
def load(cls, path: str) -> "DeviceRegistry":
    """Phase-6 hook. Returns an empty registry until real persistence lands."""
    return cls()
```

Nothing else persists either — see the state-ownership table in the
[architecture overview §7.1](../architecture%20documents/System_Architecture_Overview.md#71-hermes-mission-flow-the-real-path).
`_pending` cluster rounds, `DeviceSchedulerState`, `_prepared_delta` on devices, and the
mule's `_pending_delivery_report` are all in-process only.

**Failure scenario.** The cluster process is restarted mid-experiment (OOM, node
maintenance, a deploy). On restart the registry is empty, so `slice_for` returns an
empty tuple for every mule, every `MissionSlice` is empty, every mule's contact queue is
empty, and every mission fails with `"no submissions to aggregate"` — permanently, with
no error indicating why. The only recovery is restarting the whole topology from the
orchestrator, which discards all accumulated `on_time_history` / `missed_history` /
`delivery_priority` and therefore all scheduler adaptation.

There is no HA story, no warm restart, and no operator-visible signal that the registry
is empty rather than idle.

---

### P-03 — `wait_for_dock(timeout=None)` is an uncancellable busy-poll {#p-03}

**Category:** PERF · **Grade:** P0

```python
# hermes/mule/client_cluster.py:215-226
def wait_for_dock(self, *, timeout: Optional[float] = None) -> bool:
    deadline = None if timeout is None else time.time() + timeout
    while True:
        if self.dock.is_available():
            return True
        if deadline is not None and time.time() >= deadline:
            return False
        time.sleep(self.dock_poll_interval_s)
```

`MuleSupervisor` calls it with `timeout=None` twice — `mule_main.py:306` (single-pass)
and `mule_main.py:425` (inter-pass dock). With `timeout=None` the loop has **no exit
condition** other than the dock becoming available. It does not consult
`MuleService._stop_event`.

**Failure scenario.** The cluster process dies during a mule's Pass 1. The mule finishes
Pass 1, calls `wait_for_dock(timeout=None)`, and polls forever. SIGTERM sets the
service's stop event, but the supervisor only checks it *between* missions, so
`shutdown_all` waits its full 15 s timeout and then `SIGKILL`s the mule — losing the
Pass-1 aggregate and the delivery-report carryover that were about to be uploaded. In a
long-running deployment (not orchestrated by the test harness) the mule simply hangs
indefinitely.

---

## P1 — Blocks scale, deployment, or reproducibility

### C-03 — `--mode hermes` is a stub on both binaries {#c-03}

**Category:** CORR / ARCH · **Grade:** P1

README, §"Legacy training pipeline":

> `python3 App/TrainingApp/Client/TrainingClient.py <usual-args> --mode hermes  # HERMES path`

What the flag actually does:

```python
# App/TrainingApp/Client/TrainingClient.py:192-197
def _stub_train(theta, synth):
    raise RuntimeError(
        "hermes-mode local_train not wired yet — Sprint 1.5 deliverable")
```

```python
# App/TrainingApp/HFLHost/HFLHost.py:155-189  (_run_hermes_main)
cluster = HFLHostCluster(registry=…, generator=StubGeneratorHost(disc_weights=[]),
                         dock=LoopbackDockLink(), synth_batch_size=1)
# No serve_forever yet …
print("MODE=hermes; … awaiting Sprint 1.5 + Sprint 2 wiring.")
```

The client constructs a `ClientMission` whose training callback raises on first use; the
host constructs a cluster over a loopback link and returns. Neither serves. Both print a
success-shaped banner. `tests/unit/test_mode_switch.py` asserts that the *banner* is
printed, which is why the gap is invisible to CI.

Sprint 1.5 and Sprint 2 are both marked closed in the README's status section. The
"awaiting" comments are stale by two sprints.

---

### A-01 — Two disconnected stacks; the only real bridge is inverted {#a-01}

**Category:** ARCH · **Grade:** P1

With `--mode hermes` inert ([C-03](#c-03)), the only executed link between the two
stacks runs the wrong way — the core library imports the experiment harness:

```python
# hermes/processes/cluster.py:113
from experiments.exp4.model_task import load_weights
# hermes/processes/cluster.py:140, 461
from experiments.exp4.model_task import load_xy, evaluate_theta
# hermes/processes/device.py:89
from experiments.exp4.model_task import load_xy, make_local_train_fn
```

`device.py:86-88` acknowledges it:

> *"Layer note: hermes is the core library; this reaches up into the experiments package
> only on the opt-in real-model path, and only in a spawned device subprocess whose CWD
> is the repo root."*

**Consequences.** `hermes/` cannot be packaged or deployed without `experiments/`. The
real-model path only works when CWD is the repository root. And the two Protocols
designed precisely for this (`GeneratorHost`, `LocalTrainFn`) are bypassed on the
cluster side — `StubGeneratorHost` still emits zero tensors as "synthetic samples" in
every live path, meaning the GAN half of "GAN-based NIDS" has never run inside HERMES.

---

### E-01 — `requirements_core.txt` is a workstation `pip freeze` {#e-01}

**Category:** ENV · **Grade:** P1

README: *"Core HERMES + test deps (always required)"* and *"If you only intend to run the
multi-process HERMES path, `requirements_core.txt` + pytest is enough — `flwr` is only
needed for the legacy path."*

The file contains **273 pinned packages** including `flwr==1.9.0`, `tensorflow==2.15.0`,
`torch==2.3.1`, `torchvision`, `transformers`, `PyQt5`, `ansible`, `boto3`,
`azure-storage-blob`, `pymavlink`, `hagrid`, `syft-proto`, `opendp`, `Sphinx`, and
`gps==3.19`. It differs from `requirements_edge.txt` by 45 lines out of ~300 — the
"lightweight core" is not lightweight and not different.

Four classes of pin cannot install:

| Pin | Problem |
|---|---|
| `uuid==1.30` | abandoned 2006 backport that **shadows the stdlib `uuid` module** |
| `zmq==0.0.0` | placeholder package; the real one (`pyzmq==25.1.0`) is also pinned |
| `serial==0.0.97` | unrelated package; the intended one (`pyserial==3.5`) is also pinned |
| `gps`, `distro-info`, `launchpadlib`, `wadllib`, `ssh-import-id`, `python-apt` | Debian/Ubuntu system packages, not PyPI distributions |

`hermes/` imports exactly two third-party names: `numpy` and (in tests) `pytest`.

**Failure scenario.** A collaborator or a fresh AERPAW node runs
`pip install -r AppSetup/requirements_core.txt`. It fails on `gps==3.19`. If they
`--no-deps` past it, `uuid==1.30` shadows the stdlib module and breaks imports in
unrelated libraries. The documented install path does not work.

---

### A-06 — No packaging metadata {#a-06}

**Category:** ARCH / ENV · **Grade:** P1

Missing from the repository root: `pyproject.toml`, `setup.py`, `pytest.ini`,
`conftest.py`, `LICENSE`. The README's repository-layout block lists `pytest.ini`; the
README's footer declares an MIT licence and links `LICENSE`. Neither file exists.

Consequences visible in the code:

- 16 modules carry `sys.path.append(os.path.abspath('../../..'))` — a fragile
  CWD-relative substitute for an installed package.
- `python -m hermes.processes.cluster` only resolves when CWD is the repository root,
  which is why `device.py`'s layer-note has to state that as a precondition.
- `@pytest.mark.slow` is used in `test_mode_switch.py` and `test_exp4_model_task.py` and
  registered nowhere, emitting `PytestUnknownMarkWarning` on every run. The README's
  documented `pytest tests/ -m slow` works only by accident and would fail under
  `--strict-markers`.
- There is no way to pin what `hermes/` needs separately from what the TensorFlow stack
  needs — which is the root cause of [E-01](#e-01).

---

### S-01 — `pickle.loads` of an unauthenticated remote HTTP body {#s-01}

**Category:** SEC · **Grade:** P1

```python
# hermes/transport/cloud_link.py:159-171
with urllib.request.urlopen(req, timeout=timeout_s) as resp:
    …
    body = resp.read()
    return pickle.loads(body)          # ← arbitrary code execution on malicious body
```

`base_url` comes from `ClusterConfig.tier3_url` and is fetched over **plain HTTP** with
no TLS requirement, no authentication header, and no signature check on the returned
`GeneratorRefinement`. Unpickling untrusted bytes is arbitrary code execution in the
cluster process.

`hermes/transport/wire.py:12-17` explicitly reasons about this for the LAN transports
and concludes pickle is acceptable *because* "the only senders are co-deployed mule /
cluster / device processes". That reasoning does not extend to an arbitrary remote URL,
and the cloud link inherited the codec without inheriting the caveat.

Note the asymmetry: dock bundles carry a SHA-256 `bundle_sig` verified on receipt
(`hermes/types/signatures.py`); cloud refinements carry no integrity field at all.

**Threat model note.** The exposure is real only when `tier3_url` is set (Chameleon /
AERPAW deployment), and the current deployment is a research testbed. That bounds the
urgency, not the severity.

---

### A-09 — Single-threaded cluster service loop {#a-09}

**Category:** SCALE · **Grade:** P1

`ClusterService.run` (`hermes/processes/cluster.py:297-375`) processes **one** UP bundle
per iteration, and does everything inline:

```
recv_up(timeout=1.0) → ingest_up_bundle → aggregate_pending (cross-mule FedAvg)
  → close_cluster_round → _emit_model_evaluation (a full TensorFlow forward pass
    over the held-out set) → for mid in registered_mules(): send_down(dispatch_down_bundle(mid))
```

`dispatch_down_bundle` per mule performs `slice_for` (O(N) over all devices), a
`registry.get` per slice member, a `get_global_disc_weights` copy, a `make_synth_batch`,
and a SHA-256 signature over the whole θ. With M mules that is M full model
serializations, serialized behind one thread, while every other mule's UP waits in the
socket queue.

Meanwhile `_emit_model_evaluation` runs TensorFlow **inside the same loop** on the
real-model path, adding hundreds of milliseconds to seconds per round during which no
dock traffic is processed.

**Scale ceiling:** ~5–10 mules before dock latency dominates mission time. The dock
transport itself is already multi-threaded (one reader per mule), so the bottleneck is
purely the service loop's structure.

Also here: `except Exception: up = None` at line 300 swallows every error from
`recv_up`, including programming errors, and the loop continues as if nothing arrived.

---

### P-01 — Unbounded thread-per-device fan-out {#p-01}

**Category:** PERF / SCALE · **Grade:** P1

```python
# hermes/mission/host_mission.py:519-527  (and 696-703 in deliver_contact)
threads: List[threading.Thread] = []
for did in contact_devices:
    t = threading.Thread(target=_device_worker, args=(did, adv), daemon=True)
    t.start()
    threads.append(t)
for t in threads:
    t.join(timeout=self.session_ttl_s * 2.0)
```

Three problems in nine lines:

1. **No bound.** One OS thread per device in the contact. S3a is documented to produce
   2–3 devices per contact at `rf_range_m=60` but "most of a slice falls into one or two
   large contacts" at `rf_range_m=120` — so contact size is a tunable, and thread count
   follows it unbounded.
2. **Per-thread joins, not a deadline.** N threads each joined with
   `timeout=2 × session_ttl_s` gives a worst case of `N × 2 × ttl`, not `2 × ttl`. At
   N=20 and the default 30 s TTL that is 20 minutes for one contact.
3. **Abandoned threads keep writing.** A worker that outlives its join is `daemon=True`
   and still holds a reference to `self._report` / `self._accepted`. It can append a
   `MissionRoundCloseLine` or a `GradientSubmission` *after* `close_round` has run
   `partial_fedavg` — silently contributing to nothing, or to the next round's ledger.

---

### A-04 — Config-as-code: 47 trainer classes, 21.3 K LOC {#a-04}

**Category:** ARCH / DUP · **Grade:** P1

`Config/ModelTrainingConfig/` holds 47 classes across 45 files, 21,278 LOC. Each
(model type × sub-model × training area) combination is a separate class carrying its
own full copy of the training loop.

Helper duplication across the stack:

| Helper | Copies |
|---|---|
| `discriminator_loss` | 24 |
| `generator_loss` | 17 |
| `evaluate_validation_disc` | 17 |
| `evaluate_validation_NIDS` | 14 |
| `setup_logger` | 12 |
| `probabilistic_fusion` | 10 |
| `log_epoch_metrics` | 10 |
| `gradient_penalty` | 7 |

Function sizes inside them: `ACGANCentralTrainingConfig.fit` 499 lines (complexity 34),
`.evaluate` 307 lines, `ServerACDiscBothFitOnEndConfig.aggregate_fit` 311 lines.

**Failure scenario — the one that already happened.** The September-2025 mode-collapse
work (3:1 D:G ratio, label-smoothing rebalance, generator-LR slowdown, discriminator
health monitoring) had to be applied by hand to each affected copy. `git log` shows
`ACGANCentralTrainingConfig.py` as the single most-churned file in the current tree (40
commits). Any copy that was missed still contains the pre-fix loop, and there is no test
anywhere in `Config/` that would reveal which.

**Test coverage of this subsystem: zero.**

---

### M-02 — CWD-relative dataset paths contradict the documented invocation {#m-02}

**Category:** MAINT · **Grade:** P1

Six loaders hardcode `'../../../../datasets/<NAME>'`
(`ciciot2023DatasetLoad.py:73`, `ciciot2023DatasetLoadV2.py:255`,
`iotbotnet2020DatasetLoad.py:68`, `loadLiveData.py:104`,
`datasetLoadProcess.py:73`, plus a `/root/datasets/...` variant in the vendored
framework). Four levels up resolves correctly only from `App/TrainingApp/Client/`.

The README instructs:

```bash
python3 App/TrainingApp/Client/TrainingClient.py <usual-args>   # from repo root
unzip $HOME/datasets/CICIoT2023.zip -d $HOME/datasets/CICIOT2023
```

From the repository root, `'../../../../datasets/CICIOT2023'` resolves three levels
*above* the repository's parent — not `$HOME/datasets/`. The documented setup and the
documented invocation are mutually incompatible; the code only works from an undocumented
working directory.

---

### Q-02 — Observability can crash the process it instruments {#q-02}

**Category:** CORR · **Grade:** P1

`hermes/observability/events.py` states its contract in the module docstring:

> *"**Best-effort emit**. A write that fails (disk full, file closed) logs at `DEBUG` and
> drops the event. Observability must never crash the process it's instrumenting."*

The implementation:

```python
line = json.dumps(record, separators=(",", ":"))   # ← line 108, OUTSIDE the guard
with self._lock:
    fp = self._fp
    if fp is None: return
    try:
        fp.write(line + "\n")
    except Exception:
        log.debug("event emit failed (event=%s)", event, exc_info=True)
```

Only the `write` is guarded. Serialization is not. `_coerce`'s own docstring says:

> *"If a caller hands us a numpy scalar, `json.dumps` will raise and the emit drops via
> the exception handler"*

It does not. `TypeError: Object of type float32 is not JSON serializable` propagates
into the caller.

**Failure scenario.** `cluster.py:476` does `self.metrics.observe("model_auc", float(m["auc"]))`
— correctly cast today. Any future `emit(..., some_metric=np_array.mean())` (which
returns `np.float64`) takes down the cluster service loop from inside the logging call.
The guard is one line away from where it belongs.

---

## P2 — Compounding maintenance cost

### D-02 — `CANGANCentralTrainingConfig.py` is a 1,900-line byte-clone {#d-02}

**Category:** DUP · **Grade:** P2

```
$ diff ACGANCentralTrainingConfig.py CANGANCentralTrainingConfig.py
53c53
< class CentralACGan:
---
> class CANGan:
```

Four lines of `diff` output across 3,800 lines of file. The clone has no importer — its
only reference is a commented-out import at `modelCentralTrainingConfigLoad.py:43`. Both
files contain the same 499-line `fit`, the same 307-line `evaluate`, and the same
hardcoded learning rates from [C-01](#c-01) — so every AC-GAN fix must be applied twice
or the clone silently diverges. It already has: 40 commits touch the AC-GAN file, and
the clone's history is shorter.

Adjacent, same category: `Config/…/CentralTrainingConfig/misc/Copy of ACGANCentralTrainingConfig.py`
— a third copy, with a filename containing a space.

---

### D-01 — `run_contact` / `deliver_contact` are ~90 % duplicated {#d-01}

**Category:** DUP · **Grade:** P2

`hermes/mission/host_mission.py:342-528` (187 lines, complexity 35) and `:566-705`
(140 lines, complexity 32) share an identical skeleton:

```
guard pass mode → guard non-empty devices → snapshot θ under lock
→ broadcast FLOpenSolicit → drain _misrouted_advs into `expected`
→ while expected_set - expected.keys(): recv_ready_adv / re-stash
→ def _device_worker(did, adv): …               ← the ONLY real difference
→ spawn thread per device → join each with 2×ttl → return outcomes
```

The 47 lines of solicit-and-gather logic are byte-equivalent apart from the `pass_kind`
enum. Both carry the same three thread-management defects from [P-01](#p-01), so every
fix must land twice.

---

### A-05 — 41–42 positional-parameter dispatchers {#a-05}

**Category:** MAINT · **Grade:** P2

| Function | Parameters |
|---|---|
| `_run_fit_on_end_strategies` (`HFLStrategyTrainingConfigLoad.py:69`) | 42 |
| `modelCentralTrainingConfigLoad` (`modelCentralTrainingConfigLoad.py:49`) | 41 |
| `modelFederatedTrainingConfigLoad` (`modelFederatedTrainingConfigLoad.py:45`) | 41 |
| `ServerNIDSFitOnEndConfig.__init__` | 37 |
| `nidsAdversarialModelCentralTrainingConfig.__init__` | 34 |
| …51 more functions with ≥12 parameters | |

`hyperparameterLoading` returns a **20-element positional tuple**, unpacked into 20 local
names at each of two call sites, then threaded through the 41-parameter calls positionally.
A single transposition anywhere in that chain is a silent wrong-value bug that no type
checker or test can catch.

---

### A-07 — 5.8 K LOC of vendored code with zero importers {#a-07}

**Category:** ARCH · **Grade:** P2

`FlightFramework/` is a copy of the third-party **flight / FLoX** project (Hudson &
Hayot-Sasson, University of Chicago) — 82 files, 5,853 LOC, with its own
`pyproject.toml`, `LICENSE`, `tox.ini`, and a `torch` dependency.

`grep -rn "from flight\|import flight"` outside `FlightFramework/`: **zero hits.**

Documentation says otherwise in four places, including
`HERMES_FL_Scheduler_Implementation_Plan.md:30` and `README.md:157`. The plan changed —
`partial_fedavg.py` is 105 lines of standalone numpy — but neither the vendored copy nor
the claims were removed.

Two distinct costs: 5.8 K LOC of unreviewed third-party code in the search/lint/CI
surface, and an unresolved licence-provenance question in a repository that has no
`LICENSE` of its own.

---

### A-08 — Round-robin device→mule assignment ignores geography {#a-08}

**Category:** SCALE · **Grade:** P2

```python
# hermes/cluster/device_registry.py:157-165
ordered = sorted(self._records.values(), key=lambda r: (not r.is_new, r.device_id))
for i, rec in enumerate(ordered):
    target = mule_list[i % len(mule_list)]
```

Devices are assigned to mules **round-robin by device ID**. `DeviceRecord` carries
`last_known_position` — `rebalance` never reads it. Two devices 500 m apart with adjacent
IDs go to different mules; two devices at the same location go to different mules.

The ordering key also makes assignment unstable: when a device's `is_new` flips to
`False` after its first round, its position in `ordered` changes and **every subsequent
device shifts one slot**, reassigning a large fraction of the fleet. `delivery_priority`
carryover, `on_time_history`, and the mule's cached `DeviceSchedulerState` all become
stale for the reassigned devices.

Currently masked: `ClusterService._seed_registry_from_config` overrides `assigned_mule`
from the config immediately after seeding, and the live service loop never calls
`rebalance` again. So `HFLHostCluster.rebalance_for` is dead in production while being
the only path that would be used at real scale.

---

### M-01 — 863-line, complexity-170 function {#m-01}

**Category:** MAINT · **Grade:** P2

`experiments/analysis/exp3.py:342` — `write_figures`, 863 lines, cyclomatic complexity
170. The largest and most complex function in the repository by a factor of ~2.

Runners-up: `ACGANCentralTrainingConfig.fit` (499 lines), `ACGANClientTrainingConfig.fit`
(427), `AC_DiscModelClientConfig.fit` (400), `modelCreateLoad` (375 lines, complexity 89),
`ACGANClientTrainingConfig.evaluate` (338), `ServerACDiscBothFitOnEndConfig.aggregate_fit`
(311). 122 functions exceed 80 lines; 34 exceed complexity 20.

`test_exp3_analysis.py::test_write_figures_smoke` can only assert that output files
appear — no branch inside a complexity-170 function is independently testable.

---

### E-02 — `docker-compose.yml` hardcodes another project's Windows paths {#e-02}

**Category:** ENV · **Grade:** P2

```yaml
volumes:
  - C:/Users/kskos/PycharmProjects/FLVision/ciciot2023_archive:/app/CICIOTDataset
  - C:/Users/kskos/PycharmProjects/FLVision/iotbotnet2020_archive:/app/iotbotnet2020_archive
```

Absolute paths on one developer's Windows machine, pointing at a **different project
directory** (`FLVision`) than this repository. The file also references `flwr-server` and
`flwr-client` images that no `Dockerfile` in `AppSetup/DockerSetup/` builds under those
tags. The README presents this as the containerized deployment path.

---

### E-03 — Hardcoded `192.168.129.x` server table, duplicated {#e-03}

**Category:** MAINT · **Grade:** P2

The same four-way if/elif mapping `--host 1..4` to `192.168.129.{3,6,7,8}:8080` appears
in `TrainingClient.py:124-133` (which is used) and `ArgumentConfigLoad.py:211-220`
(inside a banner-printing function, which only displays it). Two copies of the same lab
subnet, guaranteed to drift, both requiring a code edit for any topology change. Port
8080 is hardcoded in both.

---

### Q-01 — `Sequence` used in annotations but never imported {#q-01}

**Category:** MAINT · **Grade:** P2

`hermes/mission/host_mission.py:24` imports `Callable, Dict, List, Optional, Tuple`.
`Sequence` is used at line 344 (`contact_devices: Sequence[DeviceID]`) and line 568.

No runtime error, because `from __future__ import annotations` makes annotations strings.
But `typing.get_type_hints()` on `HFLHostMission` raises `NameError`, IDE resolution
fails, and any future runtime-annotation consumer (pydantic, `dataclasses` with
`eval_str`, a schema generator) breaks. This was the only unresolved-name finding in the
entire first-party codebase outside the optional-`scapy` guard in `LiveDataExtraction.py`
— which is a good sign for the rest, and worth fixing so it stays true.

---

### Q-03 — Private-attribute access across objects {#q-03}

**Category:** MAINT · **Grade:** P2

```python
# hermes/processes/cluster.py:346, 350
self.events.emit("cluster_round_closed", cluster_round=self.cluster._cluster_round)
self._emit_model_evaluation(self.cluster._cluster_round)
```

`HFLHostCluster` exposes a `cluster_round` property (`host_cluster.py:405-408`) that
reads the same field **under the lock**. The private access bypasses the lock, so the
value can be torn against a concurrent `close_cluster_round`. Low practical risk today
(the service loop is single-threaded — see [A-09](#a-09)), but it is exactly the kind of
access that becomes a race the moment A-09 is fixed.

69 cross-object private accesses exist repo-wide; most are in tests reaching into
internals, which is its own coupling problem — `test_l1_rf_prior.py` touches `._record`
8 times, `test_exp3_sim_env.py` touches `._upload_rate_at` 4 times.

---

### T-01 — Test suite is not green on `main` {#t-01}

**Category:** MAINT · **Grade:** P2

Local run: **512 passed, 8 failed, 1 skipped**. README claims "410 passed, 22 deselected".

Seven failures are environmental (TensorFlow-dependent `test_exp4_model_task`;
`flwr`-dependent `test_mode_switch` subprocess tests) — which is itself a finding, since
there is no marker or skip condition distinguishing "needs TF" from "broken", and no
`pytest.ini` to register one.

One is a genuine stale assertion:

```python
# tests/unit/test_experiments_calibration.py:36
def test_default_calibration_is_placeholder():
    """The shipped TOML must declare itself placeholder until paper run."""
    cal = load_calibration()
    assert cal.status == "placeholder"      # AssertionError: 'verified' == 'placeholder'
```

The calibration TOML was promoted to `verified`; the test guarding that transition was
never updated. A red suite trains people to ignore red suites.

---

### G-01 — Untracked, un-ignored directories in the working tree {#g-01}

**Category:** ENV · **Grade:** P2

`git status` shows three untracked directories that `.gitignore` does not cover:

| Path | Size | What it is |
|---|---|---|
| `.claude/` | 7.8 MB | tooling worktrees — contains a **complete duplicate copy of the repository** (330 `.py` files, ~34 K LOC) |
| `hermes_rl/` | 578 KB | a **nested git repository** (`hermes_rl/.git/`), 4 files, 2,251 LOC of drone-RL prototype |
| `.idea/` | — | IDE settings |

`git add -A` from the repository root would stage all three, committing a duplicate of
the entire source tree plus a foreign repository's contents. `hermes_rl/` being a nested
`.git` also means it is neither a submodule nor tracked — its history is invisible to
this repository and it will be silently lost on a fresh clone.

`.gitignore` is 7 lines and covers `__pycache__/`, `*.feather`, `*.pt`, `*.log`, `*.zip`,
`tmp/`, `datasets/`.

---

## Cross-cutting statistics

Measured across 357 first-party Python files (excluding `.claude/worktrees`, `__pycache__`):

| Metric | Count |
|---|---|
| Functions ≥ 80 lines | 122 |
| Functions with cyclomatic complexity ≥ 20 | 34 |
| Functions with ≥ 12 parameters | 56 |
| Broad `except Exception` handlers | 90 (14 of them silent `pass`) |
| Bare `except:` | 1 |
| `threading.Thread` spawn sites | 46 (15 in `hermes/`, 31 in tests) |
| `time.sleep` call sites | 18 |
| TODO / FIXME / HACK markers | 37 (20 of them in vendored `FlightFramework/`) |
| Cross-object private-attribute accesses | 69 |
| Unresolved name references | 5 (4 are guarded optional `scapy` imports) |
| Star imports | 1 |
| Mutable default arguments | 0 ✅ |
| `global` statements | 0 ✅ |
| Syntax errors | 0 ✅ |

The last three lines matter: the codebase has none of the classic Python footguns. Its
problems are structural, not idiomatic.

---

**Next:** [`01_Refactoring_Strategy.md`](01_Refactoring_Strategy.md) sequences these into
a phased plan with effort, risk, and definitions of done.

# HERMES Operations Runbook

**Status:** Phase 7 living document. Companion to
[HERMES_FL_Scheduler_Implementation_Plan.md](HERMES_FL_Scheduler_Implementation_Plan.md).

This runbook is the **practical first-boot guide** for an engineer who
has the codebase but has never run it. Read end-to-end before the first
attempt; once you've done a successful cold-start, the §1.5 / §3 / §4
quick-refs are enough for daily work.

The Phase 7 DoD says cold-start on clean hardware completes a mission
within 30 minutes of these steps.

> **All commands assume `cwd = /path/to/FL-DNN-GAN-IDS`.** Paths are
> shown relative to the repo root. Where Windows and Unix invocations
> differ, both are listed; Windows-shell-only commands are explicitly
> labelled **(Windows / PowerShell)**.

---

## 0. Environment setup

### 0.1 Create or activate the venv

The dev environment uses a Python 3.10 venv at `.venv310/`.

```bash
# Create (one-time):
python3.10 -m venv .venv310

# Activate — Unix / Git Bash:
source .venv310/Scripts/activate     # Windows path inside Git Bash
# or
source .venv310/bin/activate         # Linux / macOS

# Activate — PowerShell:
.venv310\Scripts\Activate.ps1

# Activate — cmd.exe:
.venv310\Scripts\activate.bat
```

After activation, `python` and `pip` resolve into the venv. To check:

```bash
which python                         # → .../FL-DNN-GAN-IDS/.venv310/Scripts/python.exe
python --version                     # → Python 3.10.x
```

### 0.2 Install dependencies

```bash
# Core HERMES + test deps (always required):
pip install -r AppSetup/requirements_core.txt
pip install pytest

# Edge / training deps (legacy Flower path, AC-GAN, dataset loaders):
pip install -r AppSetup/requirements_edge.txt
```

If you only intend to run the multi-process HERMES path (no
`--mode legacy` smoke), `requirements_core.txt` + pytest is enough.
The `flwr` dependency is only needed for the legacy path and the
M3–M5 mode-switch tests.

### 0.3 Verify the install

```bash
python -m pytest tests/unit -q
```

Expected: ~370+ tests pass in under 15 seconds. If anything errors on
import, the venv is incomplete — re-run §0.2.

---

## 0.5 TL;DR — already done it once

```bash
# Activate venv + smoke test:
source .venv310/Scripts/activate          # Unix-style path on Windows works in Git Bash
pytest tests/integration/test_e2e_topology.py -v

# Run the full fast suite:
pytest tests/ -q --deselect tests/integration/test_contact_selector_ab.py \
                 --deselect tests/unit/test_mode_switch.py

# Run the slow integration suite (real subprocesses, ~30s):
pytest tests/ -m slow

# Run a single test:
pytest tests/integration/test_e2e_faults.py::test_pass_2_unreachable_device_marked_undelivered -v
```

The deselects above skip the two pre-existing flaky tests
(stochastic A/B at rf=60 m, and Flower-mode subprocess timeouts).
Both are documented in the implementation plan's exit-criteria notes.

---

## 1. Cold start — local emulation

This brings up the full topology on a single machine via the
multi-process orchestrator. End-to-end runtime: under 30 seconds for
the smallest viable topology (1 cluster + 1 mule + 1 device).

### 1.1 Verify the artefact

```bash
pytest tests/integration/test_e2e_topology.py -v
```

A passing run means the system can complete one full §4 mission cycle
through real subprocesses + TCP. If this fails, do not proceed —
something is broken in the build, not in your config. Inspect the
stderr block in the failure message; chunk L-L8 prints the last 60
lines from whichever subprocess died.

### 1.2 Run the smallest viable topology

The chunk-N e2e test is the smallest functional demo. Save the snippet
below to `scripts/run_smoke.py` (create the `scripts/` directory if
needed) and run it.

```python
# scripts/run_smoke.py
from hermes.processes import (
    ClusterConfig, DeviceConfig, MuleConfig,
    MultiProcessOrchestrator, TopologyConfig,
)


def main() -> None:
    topo = TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="c-smoke",
            dock_host="127.0.0.1", dock_port=0,  # 0 = ephemeral
            synth_batch_size=2, min_participation=1,
        ),
        mules=[MuleConfig(
            mule_id="m-smoke",
            rf_host="127.0.0.1", rf_port=0,
            rf_range_m=60.0, session_ttl_s=3.0,
            n_missions=1,                          # exit after one mission
        )],
        devices=[DeviceConfig(
            device_id="d-smoke",
            position=(0.0, 0.0, 0.0),
        )],
    )

    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=20.0)
        orch.mule_handles["m-smoke"].proc.wait(timeout=30.0)
    finally:
        # Keep tmpdir so we can read JSONL after the run.
        orch.shutdown_all(timeout=5.0, cleanup_tmpdir=False)
        print(f"\nrun dir = {orch.tmpdir}")


if __name__ == "__main__":
    main()
```

```bash
python scripts/run_smoke.py
```

The script prints the run-dir path on exit. JSONL event logs land
under that path as `cluster-c-smoke.jsonl`, `mule-m-smoke.jsonl`,
`device-d-smoke.jsonl`. Either `tail -f` them while running or inspect
post-hoc per §1.4.

> **No `--mode hermes` flag is needed** for the multi-process
> orchestrator path — that flag gates the legacy `App/TrainingApp/`
> Flower binaries. The orchestrator + entry points under
> `hermes/processes/` are hermes-mode-only.

### 1.3 Expected event sequence

For a clean run you should see, in order across the three log files:

1. `cluster_ready` — dock listening on the bound port.
2. `mule_ready` — RF server up; dock client connected.
3. `device_ready` — RF client connected to the mule.
4. `mule_bootstrapped` (cluster log) — first DOWN dispatched.
5. `dock_bootstrapped` (mule log) — bootstrap DOWN consumed.
6. `mission_started` — Pass 1 begins.
7. `device_served` × N (Pass 1 collects).
8. `up_bundle_ingested` — mule docks back, ships UP.
9. `cluster_round_closed` — cross-mule FedAvg fired.
10. `device_served` × N (Pass 2 delivers).
11. `mission_completed` — full cycle done with `delivered`/`undelivered` counts.
12. `metrics_snapshot` + `service_stopped` (mule log) — graceful exit.

Anything missing usually means a step *before* it is the real failure
point — search backwards from the last event you saw.

### 1.4 Inspect the JSONL output

```bash
RUN=$(ls -td /tmp/hermes_orch_* | head -1)        # or the printed tmpdir on Windows
ls "$RUN"
# cluster-c-smoke.jsonl  device-d-smoke.jsonl  mule-m-smoke.jsonl  ...

# Pretty-print all events from the mule:
python -c "import json,sys; [print(json.dumps(json.loads(l), indent=2)) for l in open(sys.argv[1])]" \
       "$RUN/mule-m-smoke.jsonl"

# Just the event names + ts:
python -c "import json,sys; [print(f\"{json.loads(l)['ts']:.3f}  {json.loads(l)['event']}\") for l in open(sys.argv[1])]" \
       "$RUN/mule-m-smoke.jsonl"

# Pull the metrics snapshot (mule):
python -c "import json,sys
for line in open(sys.argv[1]):
    e = json.loads(line)
    if e['event'] == 'metrics_snapshot':
        print(json.dumps(e.get('metrics', {}), indent=2))" \
       "$RUN/mule-m-smoke.jsonl"
```

If you have `jq` installed:

```bash
jq -c '{ts, event, mission_round, delivered}' "$RUN/mule-m-smoke.jsonl"
jq 'select(.event=="metrics_snapshot").metrics' "$RUN/mule-m-smoke.jsonl"
```

In Python with pandas:

```python
import pandas as pd
events = pd.read_json("/tmp/hermes_orch_xyz/mule-m-smoke.jsonl", lines=True)
events[["event", "mission_round", "delivered", "undelivered"]]
```

---

## 2. Configuring a real-size topology

The Sprint-2 DoD target is **1 cluster + 2 mules + 5 devices** (3
stationary + 2 mobile). Build the topology by extending the
`mules` and `devices` lists. The orchestrator validates the topology
before launching anything; an invalid config (duplicate IDs, slice
collision, dangling expected_devices) raises `TopologyValidationError`
synchronously.

### 2.1 Slice assignment

Each device must belong to exactly one mule. Two ways to declare it:

* **Implicit (round-robin):** leave each `MuleConfig.expected_devices`
  empty. `TopologyConfig.validate()` distributes devices round-robin
  across mules in declaration order. Deterministic across runs.
* **Explicit:** populate `MuleConfig.expected_devices` with the device
  IDs each mule owns. The validator rejects collisions and dangling
  references.

For the Sprint-2 target topology:

```python
mules = [
    MuleConfig(mule_id="m1", expected_devices=["d1", "d2", "d3"],
               rf_host="127.0.0.1", rf_port=0, rf_range_m=60.0),
    MuleConfig(mule_id="m2", expected_devices=["d4", "d5"],
               rf_host="127.0.0.1", rf_port=0, rf_range_m=60.0),
]
devices = [DeviceConfig(device_id=f"d{i+1}", position=(i*10.0, 0.0, 0.0))
           for i in range(5)]
```

### 2.2 Cluster pre-seeding

If the registry must hold device positions before the first DOWN
dispatches (i.e., the very first mission's slice should be populated),
populate `ClusterConfig.seed_devices`:

```python
ClusterConfig(
    cluster_id="c1",
    seed_devices=[
        {"device_id": "d1", "position": [10.0, 0.0, 0.0],
         "assigned_mule": "m1"},
        # ...
    ],
)
```

The orchestrator's `start_cluster` populates `seed_devices` automatically
from the topology when the field is empty — explicit population is only
needed for non-default positions or per-device `spectrum_sig` priors.

### 2.3 Capacity vs. RF-range trade-off

`MuleConfig.rf_range_m` controls how aggressively S3a clusters slice
members into contact events:

* `rf_range_m = 30` — most contacts cover N=1 device; mule does
  per-device visits.
* `rf_range_m = 60` (default) — typical clusters hold 2–3 devices per
  contact at slice sizes ≤ 10.
* `rf_range_m = 120` — most of the slice falls into one or two large
  contacts.

For local emulation development, 60 is the validated default.
For AERPAW, re-calibrate from real RF link tests at deployment.

### 2.4 Inspect a topology before launch

```python
# python -c "..." or interactive:
from hermes.processes import TopologyConfig, ClusterConfig, MuleConfig, DeviceConfig
topo = TopologyConfig(
    cluster=ClusterConfig(cluster_id="c1"),
    mules=[MuleConfig(mule_id="m1"), MuleConfig(mule_id="m2")],
    devices=[DeviceConfig(device_id=f"d{i+1}") for i in range(5)],
)
topo.validate()                      # raises TopologyValidationError on conflict
print(topo.device_to_mule)           # → {'d1':'m1','d2':'m2','d3':'m1','d4':'m2','d5':'m1'}
```

---

## 3. Mode switching (`--mode legacy` vs `--mode hermes`)

The legacy `App/TrainingApp/HFLHost/HFLHost.py` and
`App/TrainingApp/Client/TrainingClient.py` binaries default to
`--mode legacy` for zero-risk rollback. The flag is irrelevant for the
multi-process orchestrator path under `hermes/processes/` — that path
is hermes-mode by design.

### 3.1 Run in hermes mode

```bash
python App/TrainingApp/HFLHost/HFLHost.py <usual-args> --mode hermes
python App/TrainingApp/Client/TrainingClient.py <usual-args> --mode hermes
```

`<usual-args>` are whatever the legacy training pipeline already takes
(model_type, dataset, save_name, etc.). See
`Config/SessionConfig/ArgumentConfigLoad.py` for the full list.

### 3.2 Roll back to legacy

Just omit the flag:

```bash
python App/TrainingApp/HFLHost/HFLHost.py <usual-args>
python App/TrainingApp/Client/TrainingClient.py <usual-args>
```

The legacy path runs the original Flower server / client byte-for-byte
unchanged — no hermes imports cross the mode-gate (enforced by the M6
grep test).

### 3.3 Mode-switch test contract

`tests/unit/test_mode_switch.py` enforces M1–M7:

* M1, M2 — argparse defaults.
* M3, M4 — subprocess smoke under both modes (slow tests; need `flwr`).
* M5 — same for TrainingClient.
* M6 — repo-wide grep that hermes imports stay inside the
  `if args.mode == "hermes":` branch.
* M7 — both modes share one venv.

```bash
pytest tests/unit/test_mode_switch.py -v
```

---

## 4. Diagnostics

### 4.1 A subprocess crashed during start

The orchestrator's `OrchestratorError` includes the last 60 stderr lines
of the crashed subprocess (chunk L-L8). The error message contains a
`--- last stderr from <role> ---` block — read that **first** before
grepping further.

If the orchestrator is still alive but a child crashed silently:

```python
# Inside Python, with `orch` still in scope:
print(orch.cluster_handle.stderr_tail(120))
for mid, h in orch.mule_handles.items():
    print(f"=== mule {mid} ===")
    print(h.stderr_tail(120))
```

### 4.2 The mule isn't completing missions

Tail the mule's JSONL and look at the last event:

```bash
# Adjust the run-dir path:
RUN=/tmp/hermes_orch_xyz
tail -f "$RUN/mule-m-smoke.jsonl" | python -c \
  "import json,sys
for l in sys.stdin:
    e=json.loads(l); print(e['event'], {k:v for k,v in e.items() if k not in ('ts','schema_version','role','id','event')})"
```

Diagnostic table:

* No `mule_ready` → RF port bind failed; check firewall / port conflict.
  ```bash
  netstat -an | grep LISTEN              # Linux / Git Bash
  netstat -ano | findstr LISTENING       # Windows cmd / PowerShell
  ```
* No `dock_bootstrapped` → cluster never sent the initial DOWN; verify
  the cluster's dock port matches `MuleConfig.dock_port`. Inspect the
  cluster log:
  ```bash
  jq 'select(.event=="cluster_ready").dock_port' "$RUN/cluster-c-smoke.jsonl"
  ```
* `mission_started` but no `mission_completed` → look at `device_served`
  events. If the device served zero times, S1/S2A/S2B is rejecting it;
  confirm `FLState.FL_OPEN` on the device side and `FL_Threshold`
  matches the device's reported utility.

### 4.3 Cluster isn't aggregating

```bash
# Count up_bundles ingested vs cluster_rounds closed:
jq -c 'select(.event=="up_bundle_ingested" or .event=="cluster_round_closed") | .event' \
   "$RUN/cluster-c-smoke.jsonl" | sort | uniq -c
```

* `up_bundle_ingested` but no `cluster_round_closed` → `min_participation`
  is set higher than the number of mules actually reporting.
* Neither event → the mule never docked back. Check the mule's last
  event in §4.2.

### 4.4 Tier-3 refinement isn't applying

```bash
# Did any refinement land?
jq 'select(.event=="tier3_refinement_applied")' "$RUN/cluster-c-smoke.jsonl"

# Pull the cluster's metrics snapshot (counters):
jq 'select(.event=="metrics_snapshot").metrics' "$RUN/cluster-c-smoke.jsonl"
```

Counters to look at:

* `tier3_refinements_applied` — successful folds.
* `tier3_poll_failures` — transport errors (Tier-3 unreachable, timeout).
* `tier3_refinement_fold_failures` — fold-side errors (e.g. shape
  mismatch with the local θ_gen).

### 4.5 Stuck or leaked subprocesses

If the orchestrator dies and leaves children behind:

```bash
# Find stragglers (Linux / Git Bash):
ps -ef | grep -E 'hermes\.processes\.(cluster|mule|device)'

# Kill them gracefully (SIGTERM = code 15):
pkill -15 -f 'hermes\.processes\.(cluster|mule|device)'
```

```powershell
# Find + kill (Windows / PowerShell):
Get-Process python | Where-Object { $_.CommandLine -match 'hermes.processes' } | Stop-Process
```

Cleanup the lingering tmpdir:

```bash
rm -rf /tmp/hermes_orch_*                          # Linux / Git Bash
```

```powershell
Remove-Item -Recurse "$env:TEMP\hermes_orch_*"     # Windows / PowerShell
```

---

## 5. Registry seeding from a config file

For repeatable runs, seed the cluster's `DeviceRegistry` from a JSON or
TOML file rather than hard-coding positions in Python.

### 5.1 Seed file format (JSON)

```json
{
  "devices": [
    {
      "device_id": "d1",
      "position": [10.0, 0.0, 0.0],
      "assigned_mule": "m1",
      "spectrum_sig": {
        "bands": [0],
        "last_good_snr_per_band": [22.0]
      }
    },
    {
      "device_id": "d2",
      "position": [25.0, 5.0, 0.0],
      "assigned_mule": "m1"
    }
  ]
}
```

`spectrum_sig` is optional; without it the cluster falls back to a
20 dB single-band placeholder (chunk L-L3).

### 5.2 Loading from a Python script

```python
# scripts/load_seed.py
import json
from pathlib import Path

from hermes.processes import ClusterConfig, MuleConfig, DeviceConfig, TopologyConfig

seed_data = json.loads(Path("seeds/topology.json").read_text(encoding="utf-8"))

cluster_cfg = ClusterConfig(
    cluster_id="c1",
    seed_devices=seed_data["devices"],
    expected_mules=["m1", "m2"],
)
mules = [MuleConfig(mule_id=m) for m in ["m1", "m2"]]
devices = [DeviceConfig(device_id=d["device_id"],
                        position=tuple(d["position"]))
           for d in seed_data["devices"]]

topo = TopologyConfig(cluster=cluster_cfg, mules=mules, devices=devices)
topo.validate()
```

```bash
python scripts/load_seed.py
```

### 5.3 Round-trip a topology to disk for reproducibility

```python
from pathlib import Path
Path("topo.json").write_text(topo.to_json(), encoding="utf-8")

# Later:
from hermes.processes import TopologyConfig
topo2 = TopologyConfig.from_file(Path("topo.json"))
topo2.validate()
```

---

## 6. AERPAW deployment (when the testbed returns)

The orchestrator + per-process entry points already accept routable
host strings. Deployment is a config swap, not a refactor.

### 6.1 Per-AVN host wiring

Replace `127.0.0.1` in `TopologyConfig` with the AERPAW AVN IPs:

* Cluster AVN → `cluster.dock_host` = its routable IP.
* Each mule AVN → its `mule.rf_host` = the mule's IP; `mule.dock_host`
  = the cluster's IP.
* Each device AVN → `device.mule_rf_host` = the assigned mule's IP.

Generate per-AVN config files locally first:

```python
# scripts/emit_avn_configs.py
import json
from pathlib import Path
from dataclasses import asdict

from hermes.processes import ClusterConfig, DeviceConfig, MuleConfig, TopologyConfig

# ... build `topo` with real AVN IPs ...
topo.validate()

out = Path("avn_configs"); out.mkdir(exist_ok=True)
(out / "cluster.json").write_text(json.dumps(asdict(topo.cluster), indent=2))
for m in topo.mules:
    (out / f"mule-{m.mule_id}.json").write_text(json.dumps(asdict(m), indent=2))
for d in topo.devices:
    (out / f"device-{d.device_id}.json").write_text(json.dumps(asdict(d), indent=2))
```

### 6.2 Process placement

```bash
# On the cluster AVN:
ssh cluster-avn 'cd ~/hermes && source .venv310/bin/activate && \
  python -m hermes.processes.cluster \
    --config /home/avn/cluster.json \
    --port-out /tmp/cluster.port \
    --run-dir /tmp/hermes_run'

# On each mule AVN (substitute mule_id):
ssh mule1-avn 'cd ~/hermes && source .venv310/bin/activate && \
  python -m hermes.processes.mule \
    --config /home/avn/mule-m1.json \
    --port-out /tmp/mule.port \
    --run-dir /tmp/hermes_run'

# On each device AVN (substitute device_id):
ssh dev1-avn 'cd ~/hermes && source .venv310/bin/activate && \
  python -m hermes.processes.device \
    --config /home/avn/device-d1.json \
    --run-dir /tmp/hermes_run'
```

On AERPAW, the `MultiProcessOrchestrator` (which assumes co-located
processes) doesn't apply — each AVN runs its single role under its
own systemd unit / `tmux` session. A minimal `tmux` pattern:

```bash
ssh cluster-avn 'tmux new -d -s hermes "source .venv310/bin/activate && python -m hermes.processes.cluster --config ~/cluster.json --port-out /tmp/cluster.port --run-dir /tmp/hermes_run"'
ssh cluster-avn 'tmux capture-pane -p -t hermes'   # peek at output later
```

### 6.3 Tier-3 wiring

If a Tier-3 endpoint is reachable, set `ClusterConfig.tier3_url` to
its base URL. The cluster polls every 5 s (per
`_TIER3_POLL_INTERVAL_S`) and folds returned refinements
(chunk P2). If Tier-3 is unreachable, leave the field None.

```python
ClusterConfig(
    cluster_id="c1",
    tier3_url="http://tier3.aerpaw.example/api",
    # ...
)
```

### 6.4 Telemetry collection

Per-process JSONL files land in `--run-dir`. Aggregate them
post-experiment by `scp`-ing each AVN's log directory back to the
analysis machine; the JSONL format is line-delimited so it
concatenates trivially across nodes.

```bash
# Pull every AVN's JSONL back to ./runs/<timestamp>/:
RUN=runs/$(date +%Y%m%d-%H%M%S); mkdir -p "$RUN"
for host in cluster-avn mule1-avn mule2-avn dev1-avn dev2-avn dev3-avn dev4-avn dev5-avn; do
  scp "$host:/tmp/hermes_run/*.jsonl" "$RUN/"
done

# One combined view sorted by timestamp:
cat "$RUN"/*.jsonl | jq -s 'sort_by(.ts)' > "$RUN/combined.json"
```

---

## 6.5 Running Experiment 1 (FL vs Centralized)

Experiment 1 has its own server + client scripts under
`experiments/exp1/`. They're **testbed-agnostic**: identical commands
work on a single Linux box, AERPAW (5 AVNs), or Chameleon (5 KVMs);
only the IPs in the topology change.

### 6.5.1 Local smoke (one Linux box, no shaping)

```bash
# Smallest viable run — 1 cell × 2 arms × 2 trials = 4 trials, ~2s
N_TRIALS=2 FILTER='Dpd=10MB,R=5' \
  bash experiments/exp1/setup/launch_local.sh

# Inspect the CSV:
head results/exp1.csv
```

The launcher spawns the server + 4 clients in the background under
`logs/exp1_*.log` and waits for the server to finish. Tail those logs
during the run if anything looks stuck.

### 6.5.2 Local with link shaping (paper-grade variance)

```bash
# Apply 10 Mbps cap on the loopback interface (Linux + sudo only):
sudo bash experiments/exp1/setup/shape_link.sh apply

# Run the full grid:
N_TRIALS=20 bash experiments/exp1/setup/launch_local.sh

# Single-cell jittery ablation (|D|pd=100MB, alpha=1.0):
sudo bash experiments/exp1/setup/shape_link.sh apply --jittery
N_TRIALS=20 FILTER='Dpd=100MB,alpha=1.0' OUT=results/exp1_jittery.csv \
  bash experiments/exp1/setup/launch_local.sh

# Always remove the qdisc when done so other workloads aren't capped:
sudo bash experiments/exp1/setup/shape_link.sh remove
```

### 6.5.3 AERPAW or Chameleon deployment

```bash
# 1. Edit configs/exp1_aerpaw.json with the 5 AVN IPs (server + 4 clients).
# 2. SSH-launch:
ssh server-avn 'cd ~/hermes && source .venv310/bin/activate && \
  python -m experiments.exp1.server \
    --topology ~/configs/exp1_aerpaw.json \
    --output ~/results/exp1.csv'

# 3. On each client AVN (parallel via SSH-batch or separate terminals):
for i in 1 2 3 4; do
  ssh "dev${i}-avn" "cd ~/hermes && source .venv310/bin/activate && \
    python -m experiments.exp1.client \
      --client-id d${i} \
      --server <server-avn-ip>:9000 \
      --data-partition $((i-1))" &
done
wait

# 4. Pull the CSV back:
scp server-avn:~/results/exp1.csv results/
```

The server's three topology modes (`--topology JSON`, repeated
`--client` CLI flags, or `--discover`) are all available — pick the
mode that matches your provisioning workflow. Defaults are explicit
(JSON or `--client`); `--discover` is opt-in.

### 6.5.4 Diagnostic checks

```bash
# Verify a CSV row's energy components match the calibration:
python -c "
from experiments.calibration import load_calibration, exp1_energy_proxy
import csv
cal = load_calibration().exp1
with open('results/exp1.csv') as f:
    row = next(csv.DictReader(f))
e = exp1_energy_proxy(
    T_proc_s=float(row['Tproc_s']),
    B_pw_bytes=float(row['Bpw_bytes']),
    cal=cal,
)
print(f'idle={e.idle_J:.4f}J  tx={e.tx_J:.4f}J  total={e.total_J:.4f}J')
"

# Confirm calibration provenance before a paper run:
python -c "
from experiments.calibration import load_calibration
cal = load_calibration()
print(f'status={cal.status}  source={cal.source}  verified={cal.last_verified}')
assert cal.is_paper_grade, 'calibration is still placeholder!'
"
```

---

## 7. Phase 7 verification commands

Use these after a code change to confirm the Phase 7 invariants haven't
regressed.

```bash
# 1. Design-principle assertion suite (chunk P1):
pytest tests/unit/test_design_principles.py -v

# 2. Tier-3 refinement-fold path (chunk P2):
pytest tests/unit/test_tier3_refinement_fold.py -v

# 3. Loopback-retirement invariant — no production module uses Loopback (chunk P3):
pytest tests/unit/test_loopback_retirement.py -v

# 4. Full slow-test suite (chaos / fault injection / e2e):
pytest tests/ -m slow -v

# 5. Everything except the two pre-existing flakes:
pytest tests/ --deselect tests/integration/test_contact_selector_ab.py \
              --deselect tests/unit/test_mode_switch.py
```

Expected counts at Phase 7 closeout: **410 passed, 22 deselected**.

---

## 8. Cleanup

### 8.1 Local emulation

```bash
# Normal exit:
# (orchestrator's shutdown_all() handles SIGTERM + tmpdir cleanup)

# After a crashed run, find + kill leaks:
pkill -15 -f 'hermes\.processes\.(cluster|mule|device)'   # Linux / Git Bash
rm -rf /tmp/hermes_orch_*

# Windows / PowerShell:
Get-Process python | Where-Object { $_.CommandLine -match 'hermes.processes' } | Stop-Process
Remove-Item -Recurse "$env:TEMP\hermes_orch_*"
```

### 8.2 AERPAW

* Each AVN's per-process script catches SIGTERM and exits cleanly,
  flushing its JSONL file. Use **`kill -15`**, not `-9`, so the metrics
  snapshot fires.
  ```bash
  ssh cluster-avn 'pkill -15 -f hermes.processes.cluster'
  ```
* On Windows the equivalent is to send `Ctrl-C` to each terminal; hard
  kill via `taskkill /F` skips the JSONL flush.

### 8.3 Reset to a clean checkout

```bash
# Discard local changes + reinstall deps from scratch:
git status --short
# (review, then if desired:)
git stash -u
rm -rf .venv310
python3.10 -m venv .venv310
source .venv310/Scripts/activate
pip install -r AppSetup/requirements_core.txt -r AppSetup/requirements_edge.txt pytest
pytest tests/integration/test_e2e_topology.py -v
```

---

## 9. Where to look when this runbook is wrong

* [HERMES_FL_Scheduler_Design.md](HERMES_FL_Scheduler_Design.md) —
  what each component is supposed to do.
* [HERMES_FL_Scheduler_Implementation_Plan.md](HERMES_FL_Scheduler_Implementation_Plan.md)
  — what's actually built and what's pending.
* [HERMES_Configuration_Reference.md](HERMES_Configuration_Reference.md)
  — every tunable, with rationale.
* [HERMES_Experiments_Implementation_Plan.md](HERMES_Experiments_Implementation_Plan.md)
  — how to drive the system for paper experiments.
* `tests/integration/` — every behaviour pinned by tests; if the runbook
  step claims X but a test expects Y, trust the test.

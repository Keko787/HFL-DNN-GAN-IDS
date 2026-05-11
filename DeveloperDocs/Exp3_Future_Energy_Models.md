# Experiment 3 — Future propulsion-energy models

This document describes two extensions to the Experiment-3 propulsion-energy
model that are NOT currently implemented but are documented here so they can be
added later if a reviewer requires them or if a deployment study calls for
them. The currently-shipping model treats the mule as a rigid waypoint
follower — it visits each admitted cluster's centroid once, takes whatever
yield falls out, and never revisits or repositions. Under that model, jittery
cells consume *less* propulsion energy than clean cells (because slow uploads
truncate Pass 1), which is counterintuitive but mathematically correct for the
model. The current paper handles this by reporting **Option A** —
`propulsion_energy_per_completed_device` (a normalized panel, fig0g) — which
inverts the comparison and exposes the true operational cost.

This document covers the two more-realistic alternatives that we kept in our
back pocket:

- **Option B**: retry / revisit on failed contacts.
- **Option C**: adaptive in-cluster positioning (mule micro-flies between
  individual devices when RF is poor).

Either option can be added incrementally without breaking the existing
metrics or trial-grid harness.

---

## Option B — retry / revisit on failed contacts

### Motivation

In the current simulator, every contact is visited at most once per mission.
If the mule arrives at a cluster and the per-contact uplink fails (2 % packet
loss in jittery cells), every device in that cluster's batch is lost
together — and the mule moves on to the next cluster without ever returning.
In a real deployment, a mule that detects a failed UP transmission would be
expected to retry, either immediately or by revisiting later in the mission.
Under jittery network conditions this would **add** flight time and
propulsion energy, matching the intuition that a hostile network *costs* more
energy rather than less.

### What changes in the simulator

#### 1. Detect `contact_uplink_kept = False` outcomes

[`Exp3Sim.step`](../experiments/exp3/sim_env.py) already computes
`contact_uplink_kept` as a per-contact Bernoulli. Currently the result is
folded into per-device completion (`completed = contact_uplink_kept and …`).
For retry support, surface that outcome in `Exp3StepResult` so the arm driver
can decide what to do:

```python
@dataclass(frozen=True)
class Exp3StepResult:
    ...
    contact_uplink_kept: bool          # NEW
    retry_eligible: bool               # NEW — True if uplink failed but
                                       # transit + collect succeeded
```

#### 2. Maintain a per-mission retry queue in `arm_mule.run_mule_trial`

```python
retry_queue: list[ContactWaypoint] = []
max_retries_per_contact = 1   # config knob
retry_count: dict[ContactWaypointID, int] = {}

while not sim.done:
    if retry_queue and policy_chooses_retry_over_new(...):
        chosen = retry_queue.pop(0)
        result = sim.step(chosen)
    else:
        ranked = policy.rank_contacts(...)
        if not ranked:
            break
        chosen = ranked[0]
        result = sim.step(chosen)

    if (not result.contact_uplink_kept
        and retry_count.get(id(chosen), 0) < max_retries_per_contact):
        retry_queue.append(chosen)
        retry_count[id(chosen)] = retry_count.get(id(chosen), 0) + 1
```

#### 3. New config knobs

Add to `Exp3SimConfig`:

```python
@dataclass(frozen=True)
class Exp3SimConfig:
    ...
    max_retries_per_contact: int = 0       # 0 = current no-retry behaviour
    retry_policy: str = "deferred"          # "immediate" | "deferred"
```

`max_retries_per_contact = 0` preserves the current behaviour. Bumping it to
1 or 2 enables the new model. `retry_policy = "immediate"` re-runs `step`
straight away (cheaper transit since the mule is still at the cluster);
`"deferred"` queues the contact for later (more transit cost, but lets a
smarter policy interleave retries with new visits).

#### 4. Cost-model effects

Each retry adds:

- One full transit segment (mule has to fly back to the cluster).
- One `session_time_s` collect (assuming a fresh exchange).
- One `upload_s` per regime — which is the variable that drove the retry in
  the first place.
- All of the above charged against `mission_budget_s`.

So jittery cells, where retries fire 2 % per contact at the contact-uplink
level, would expend extra flight kilometres trying to recover those losses.
At a mean ~3 contacts per jittery mission, expected retries are ~0.06 per
mission — small in absolute terms but with a multiplier on transit
(potentially 50–200 m extra flight × 10 J/m = 500–2000 J more propulsion).

#### 5. Metric panel implications

- `propulsion_energy_J` rises in jittery (matches intuition: hostile network
  costs more energy).
- `path_length_m` rises in jittery.
- `mission_completion_rate` *also* rises slightly in jittery (some retries
  succeed and recover their cluster's contributions).
- `update_yield` per round rises slightly in jittery.

The fig0e raw-energy panel would invert: jittery > clean. Fig0g (energy per
completion) becomes less dramatic because both numerator and denominator move
in the same direction.

### Implementation cost

- ~50 lines in `Exp3Sim` and `arm_mule.py`.
- ~3 new unit tests (retry budget exhaustion, retry-success bookkeeping,
  immediate-vs-deferred behaviour).
- Re-train A4 selector? Probably yes — the reward landscape shifts when
  retries become viable, so the trained DDQN should be re-fit on the new
  dynamics for paper-grade A4 numbers.
- Trial-grid sweep cost: same number of trials, slightly longer per-trial
  walltime (a few μs).

### Reviewer angle

A reviewer asking *"why does jittery use less energy?"* is satisfied the
moment they see the energy-per-completion panel (fig0g) plus a sentence
acknowledging that the model treats every contact as one-shot. Adding Option
B closes the loophole completely but isn't necessary for paper-publication
unless the reviewer explicitly asks for retry-resilient propulsion modeling.

---

## Option C — adaptive in-cluster positioning

### Motivation

The current model parks the mule at the cluster centroid for the entire
collect window and computes per-device completion via:

```python
rf_factor = max(0.4, 1.0 - d_dist / (3.0 * world_radius))
p_complete = reliability * rf_factor
```

`d_dist` is the distance from the centroid to each device. A device near the
edge of a wide cluster gets `rf_factor ≈ 0.8`, a device near the centroid
gets `rf_factor ≈ 1.0`. There is **no compensation** — the mule does not
move closer to peripheral devices or sweep through the cluster's footprint.

In practice, a real mule with an actuated antenna or a steerable beamformer
*would* trade some transit cost for better RF positioning when serving
peripheral cluster members. Under jittery conditions where every device-side
contribution is precious, this trade-off would presumably push the mule to
take more micro-flights — adding `path_length_m` (and propulsion energy) in
exchange for higher `mission_completion_rate`.

### What changes in the simulator

#### 1. Decompose the per-cluster collect into per-device sub-steps

Instead of a single `session_time_s` block at the centroid, the collect phase
becomes a sequence of mini-flights. For each device in the cluster:

```python
mini_transit = _euclid(current_mule_pose, dev.pos) / cruise_speed_m_s
sub_session = session_time_s / len(contact.devices)  # split window equally
# OR adaptive:
#   sub_session = session_time_s * (1.0 - rf_factor_at_centroid_for_this_device)
```

Each mini-step charges its own transit + sub-session against the budget.

#### 2. Recompute `rf_factor` per-position

Once the mule is closer to the device, the per-device `rf_factor` improves:

```python
for dev in contact.devices:
    micro_pose = optimal_position_for(dev, current_mule_pose)
    transit_to_dev = _euclid(current_mule_pose, micro_pose) / cruise_speed_m_s
    # ... charge transit, advance mule ...
    rf_factor = compute_rf_factor(micro_pose, dev.pos)
    p_complete = dev.reliability * rf_factor
    ...
```

This significantly raises per-device completion probability in clusters
where peripheral devices currently fail.

#### 3. New config knob

```python
@dataclass(frozen=True)
class Exp3SimConfig:
    ...
    in_cluster_positioning: str = "centroid"   # "centroid" | "per_device"
    per_device_session_s: float = 6.0          # if per-device: time per dev
```

`"centroid"` preserves current behaviour. `"per_device"` enables Option C.

#### 4. Cost-model effects

- `path_length_m` rises substantially in *all* regimes (extra micro-flights
  per cluster).
- `propulsion_energy_J` rises proportionally.
- `mission_completion_rate` rises (peripheral devices that previously failed
  now succeed).
- `time_total_s` is approximately the same (per-device sessions sum to about
  the same as one centroid session, plus mini-transit overhead).
- Fewer contacts fit in the mission budget (because of the extra mini-
  transits) — so `coverage` and `update_yield` may *drop* even as
  per-cluster completion rises.

This is a genuine trade-off the paper could explore: *"per-device adaptive
positioning improves per-cluster completion at the cost of cluster reach."*

### Open design questions

- **What's `optimal_position_for(dev, current_pose)`?** Trivially the
  device's own position, but that ignores the cost of returning to the
  cluster centroid before moving to the next device. A real implementation
  would solve a TSP-lite over the cluster members. For the simulator, a
  greedy nearest-next sweep is probably good enough.
- **How do we split `session_time_s`?** Equal-weight split is the simplest;
  proportional-to-rf-deficit is more realistic but introduces config
  complexity.
- **Does this break A3's feasibility filter?** The
  `EdfFeasibilityPolicy.FeasibilityModel` currently estimates per-cluster
  cost as `transit + session + upload`. With Option C, the session has a
  variable inner cost. The filter's projected cost would need to include
  an `expected_in_cluster_overhead_s` term per cluster.
- **A4 retraining?** Definitely. The state-action dynamics change
  significantly because contacts now consume more time per yield.

### Implementation cost

- ~150 lines in `Exp3Sim` and `arm_mule.py`.
- ~5 new unit tests.
- A3 feasibility model needs a new term and a new test.
- A4 must be retrained against the new dynamics.
- Trial-grid sweep cost: same number of trials, ~1.5–2× per-trial walltime
  (more sub-steps).

### Reviewer angle

This is the change to make if the paper claims to model RF-adaptive mule
behaviour. Without it, the paper's energy comparison is honest only under
the assumption of a fixed-pose mule — which is a reasonable simplification
for a scheduling-strategy ablation but is worth flagging in the methods
section. The doc's recommendation: ship the current paper without Option C,
note the assumption explicitly, and reserve Option C for a follow-on
deployment study.

---

## Summary of the three options

| Option | Status | Effect on raw energy panel | Effect on intuition | Implementation cost |
|---|---|---|---|---|
| **A** Energy-per-completion normalization | **Shipped** (fig0g) | unchanged | inverted to match intuition | ~30 lines, no simulator changes |
| **B** Retry on failed contact | Future work | jittery > clean (matches intuition directly) | matches | ~50 lines, A4 retrain |
| **C** Adaptive in-cluster positioning | Future work | rises in all regimes; trade-off between coverage and completion | matches; new trade-off surface | ~150 lines, A3+A4 retrain, methods rewrite |

For the current Experiment-3 paper:

- **Lead with Option A** — the normalized panel is honest and exposes the
  operational cost without altering the simulator's physics.
- **Document the model's one-shot-contact assumption** in the methods
  section so reviewers know the absolute energy comparison is conditional
  on that assumption.
- **Hold Option B for a revision round** if a reviewer specifically asks
  about retry behaviour.
- **Hold Option C for a follow-on deployment paper** — it's the right
  model for an actual operational claim, but it's a larger experiment
  scope than the current scheduling-strategy ablation.

---

## Pointers

- Current propulsion model: [Eq. 5 in the paper, implemented in
  `experiments/calibration.py::exp3_energy_proxy`](../experiments/calibration.py).
- Current step physics:
  [`Exp3Sim.step`](../experiments/exp3/sim_env.py).
- Current arm driver:
  [`run_mule_trial` in `arm_mule.py`](../experiments/exp3/arm_mule.py).
- The energy-per-completion derived metric:
  [`Exp3Row.propulsion_energy_per_completion` in
  `experiments/analysis/exp3.py`](../experiments/analysis/exp3.py).

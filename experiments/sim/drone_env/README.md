# drone_env — the hermes_rl prototype, vendored

**Source.** [github.com/FyneappleJuice/hermes_rl](https://github.com/FyneappleJuice/hermes_rl),
commit `a8a453f` ("Initial commit", 2026-04-28), authored by FyneappleJuice. Copied into this
repository on 2026-09-28 as Phase 0 of the FeRRy build plan, which settles the `hermes_rl/` part
of finding G-01 (`DeveloperDocs/Codebase Review/00_Critical_Problem_Areas.md`). The original
checkout was left untouched and still holds its git history. The source repository has no
license file; confirm reuse terms with its author before this code is published anywhere.

## What is here

| File | What it is | Needs |
|---|---|---|
| `drone_env.py` | The environment: a drone moves between waypoints, collects from sensors in range and uploads to base stations over channels whose rate is `max(0, α·sin(ωt + φ) + β/(1 + γd))`. Gymnasium-compatible, and runs without Gymnasium. | numpy |
| `train_dqn.py` | Hybrid trainer (a Double DQN picks the job, a heuristic picks waypoint, base station and channel) and a standalone DQN mode over the full action. | torch, matplotlib |
| `drone_demo.py` | Pygame visualisation of the heuristic agent. | pygame |
| `dqn_results_a8a453f.png` | The results plot as committed at `a8a453f`. | — |

Tests: `tests/unit/test_drone_env.py`.

```python
from experiments.sim.drone_env import DroneDataRelayEnv, ScenarioRandomization, overloaded_config

cfg = overloaded_config()                        # the 5-job training scenario
cfg.randomization = ScenarioRandomization(       # optional; off by default
    position_jitter=5.0, random_phases=True, deadline_jitter=0.2, rate_noise_std=0.05,
)
env = DroneDataRelayEnv(config=cfg)
obs, info = env.reset(seed=11)                   # one seed, one world
```

```bash
python -m experiments.sim.drone_env.train_dqn --episodes 1500   # from the repository root; needs torch
python -m experiments.sim.drone_env.drone_demo                  # needs pygame
```

## Changes from a8a453f

- **Package imports.** `from .drone_env import …`; the trainer and the demo run as modules.
- **`overloaded_config()`** is the training scenario, moved unchanged from
  `train_dqn.make_env_config` (which now returns it) so that it builds without torch.
- **`ScenarioRandomization`**, an opt-in, seeded variation of each episode: waypoint and sensor
  jitter, random channel phases, deadline jitter, and rate noise drawn as a table over time so
  that querying rates does not change the world. Every field defaults to off.
- **The trainer's plot** goes to `results/drone_env/dqn_results.png` by default instead of the
  working directory.
- **`drone_demo.py` did not compile under Python 3.** Two bytes (`0xB7`) were Latin-1 inside
  otherwise UTF-8 source; they are transcoded to UTF-8 (`A·`). Nothing else in it changed.
- **Not carried over:** the committed `__pycache__/*.pyc` and a `.gitignore` saved as UTF-16,
  which git could not read. This repository's `.gitignore` already covers `__pycache__/`.

With default settings the environment is the original. Three rollouts recorded from the
original files — a random-action episode in the default scenario, one in the training scenario,
and a scripted collect-and-upload episode that exercises collection, upload, the completion
bonus and the deadline penalty — are reproduced exactly by the tests.

## Relation to SEC'26

SEC'26 Table VI (HERMES RL 76.40 return, 3 jobs done, 2 failed; EDF −4.85, 2 done, 3 failed;
five jobs on a 100 × 100 plane, "overloaded"), Fig. 7 and Observation 2 match the hybrid run of
this code in the training scenario: `dqn_results_a8a453f.png` shows the hybrid policy's
evaluations at about +76 with 3 jobs done and 2 failed early in training, against the
heuristic's −5 with 2 done and 3 failed. That match is by numbers, scenario and metrics; the
training has not been re-run here. `SEC26_Code_Audit.md` (question H4) records this provenance
as unknown.

## Known issues, as found (not changed)

- **Evaluation runs on the training instance.** The environment ignores seeds by default, so the
  trainer's periodic evaluations (seeds 2000+) and its final evaluation (seeds 9000+) all play
  the same deterministic episode. The reported model is the best of about 30 evaluations on
  that one instance, re-measured on the same instance. `ScenarioRandomization` makes a held-out
  evaluation possible.
- **The default scenario cannot be completed.** At 0.05 MB per step its jobs need 600 to 1,600
  steps of collection against a 300-step cap, and sensor 1 sits 15.6 units from the nearest
  waypoint, outside its 15-unit radius. Use `overloaded_config()` or a scenario of your own.
- **The hybrid heuristic has oracle channel knowledge:** it reads the exact rate at the predicted
  arrival time. The observation itself carries no channel state; the phase can only be inferred
  from the clock.
- **Without Gymnasium,** `reset(seed=…)` seeds NumPy's global generator (the fallback base class
  calls `np.random.seed`), and `action_space.sample()` is not seeded at all.

## Where FeRRy uses it

The time-axis channel here is the prototype for Phase 3's seconds-axis channel model, and the
environment is the starting point for the flight-clock experiments: the transit-time-over-channel-
period sweep (test c), the γ-sweep (test b), and the DQN over the contact graph after Chen et
al. (arm E3). Mapping it onto HERMES — contact waypoints, a band on the contact link, seconds at
the cruise speed, S3b as an action mask, rewards in FL units — is Phase 5 work; see
`DeveloperDocs/FeRRy_Build_Plan.html`.

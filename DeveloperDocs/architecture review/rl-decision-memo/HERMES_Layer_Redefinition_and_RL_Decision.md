# Layer redefinition and the RL decision — record of the analysis

**Status:** decision memo, not a freeze. Nothing here changes recorded sweeps. It specifies what E3
should be and why, and it is the permanent copy of an interactive memo built over 2026-08-26 →
2026-09-16 (artifact: *Does HERMES Need RL?*, `claude.ai/artifact/EXCburbTHNsGmrA58GGY8V`, private;
standalone HTML copy: [`HERMES_Layer_Redefinition_and_RL_Decision.html`](HERMES_Layer_Redefinition_and_RL_Decision.html), artifact v12).

**Prompted by** the team's own diagnosis in `HEREMES Methodology REDEFINED.docx`: the problem is
novel (FL for edge devices that cannot reach the FL host), the solution space is somewhat novel
(drones as FL relay hosts), but **the solution method is too simple to justify RL**. The proposal
on the table was a Trajectory Navigation layer plus joint optimisation with RF band selection, with
the open worry that a cross-heuristic would be just as good.

**Traced against:** [`HERMES_Matrix_Results.md`](../../HERMES_Matrix_Results.md) (640 trials,
2026-08-13) · [Exp4 L1 record](../../HERMES_Experiment4_L1_RF_Layer.md) · [Exp4 L2 record](../../HERMES_Experiment4_L2_Scheduling_Layer.md)
· [Related Work notes](../../HERMES_Related_Work_Notes.md) · [Baseline candidates](../../HERMES_SOTA_Baseline_Candidates.md)
· Holistic Revision Plan §7.3 / Freeze D4 · `hermes_rl/drone_env.py` · `experiments/exp4/{driver,channel}.py`
· Chen et al., GLOBECOM Wkshps 2023 + `github.com/Cirrick/Multi_UAV_Data_Harvesting`.

> **The one-sentence version.** Adding a trajectory layer will not justify reinforcement learning.
> Adding a *coupling* will. The distinction is testable, and the test is an ablation that can run on
> code that already exists.

---

## 0. Why the layer question is downstream of a measurement problem

Before any redesign, there is a reason no policy inside the mule arm can currently demonstrate
anything, and it is already in the matrix results — stated as a strength, which it is, without the
consequence drawn:

> H1's `final_auc` sits between **0.922 and 0.933 in every one of the twelve cells** — flat. H0's
> falls from 0.943 to **0.587** as the infrastructure degrades. *(Matrix results, sweep A2)*

That flatness is the headline finding (the mule never uses the backhaul that is failing). It is
also why **H2 ties H1**: the mule saturates the outcome and leaves almost no variance for a scheduler
to explain. The C-sweep found the same thing on its own when its prediction failed — more missions
*halved* the L1 effect (+0.046 at 4 → +0.023 at 6), which falsifies "L1 reaches a better model" in
favour of "L1 reaches the same model sooner." **These mechanisms buy convergence speed, not a better
endpoint, so an endpoint metric cannot see them.** No architecture change fixes that.

Two fixes, both already identified in the matrix results, neither needing new code:

| Fix | Currently | Change to | Why |
|---|---|---|---|
| Convergence target τ | τ = 0.90 — **5.9 %** of trials reach it | **τ = 0.82** — 50.2 % reach it (report 0.85 as sensitivity) | 0.90 is above the p90 of `final_accuracy`; it measures almost nothing. At the median the metric has resolution both ways. |
| Operating point | `--mission-budget-s 120`, deliberately slack | **~60 s** (the S3b knee) plus a 30 s stress cell | Decisions only matter where something is scarce. At 120 s the budget barely binds; below ~30 s round closure collapses to ≈0. |

Do both and the existing system may already show an effect. Skip them and every redesign inherits an
unmeasurable outcome.

---

## 1. The mechanism — a layer is additive, a coupling is not

The reason "trajectory navigation" reads as *just another layer* is that, as specified, it is one.
Three modules that each optimise their own objective and hand a result downstream compose into a
**chain**; a chain decomposes; a cross-heuristic solves a decomposed problem; therefore RL is
unjustified. That reasoning is correct.

But the physics is not a chain, and it is not one loop either. It runs on **two clocks**:

```
PLAN TIME — committed once, before takeoff
  demand → contact band → range → S3a clustering → route + order → S3b feasibility → COMMIT
                                                                                        │
                                                                            committed plan ↓
FLIGHT TIME — decided again at every stop                                              │
  round closes ← rate → dwell ← band at arrival ← observed phase ← arrive at stop ◄────┘
       │              └──────── drift → re-plan (today: abort only) ────────►  route
       └──────────────── next mission's demand ─────────────────────────────► demand
```

**The entry point is demand** — the FL job scheduler's "who needs service, and by when." It is the
only exogenous input; band and route are both responses to it. *(An earlier draft put band choice at
the origin; that inverted the causality and was corrected.)*

**Band is not one decision.** It is two, on two clocks:

* **Range is a commitment.** The band class flown sets contact range, range sets S3a clustering,
  clustering fixes the stops and therefore the route. Settled before takeoff; not revisable mid-hop
  without re-planning.
* **Rate is revisable.** The band actually transmitted on at a stop depends on the phase found on
  arrival. Settled per hop — and it is where `hermes_rl/drone_env.py` already operates.

The hard part, and the reason a decomposed heuristic struggles: **the slow commitment bounds the fast
decision, while the payoff of the slow commitment is only observable through the fast one.** You
commit to a clustering under an assumed range, then discover stop by stop whether the rate you
actually get pays for the route that range implied. A two-timescale problem with a commitment stage
is a far better home for a value function than a permutation of contacts that will all be visited
anyway — which is what the current `target_selector` slot is (L2 record §3: "the same work in a
different order").

### 1.1 The three cuts — where the coupling is severed in code

Half of the picture above is built. The half that couples the two clocks is not. Three cuts, one per
region; each is where a fix lands.

| # | Region | Cut | Where |
|---|---|---|---|
| **1** | Plan time · the commitment | **There is no contact-band decision to make.** Today's L1 selects a band for the *backhaul* (mule → BS, once per mission, at the dock). The *device-contact* link is a different radio with no band variable at all — `rf_range_m` is a fixed 60 m constant. The fix is not "wire L1's band to range"; it is **introduce a second band decision on the contact link** and let that set range. Without it the plan-time loop has nothing to commit to. | `experiments/exp4/driver.py:75,155`; L1 record §0 (backhaul only) |
| **2** | Flight time · the reaction | **The band decision sits on the wrong clock.** L1 re-selects per *mission*; the phase that matters is the one at *arrival*. A once-per-sortie commitment cannot respond to a channel that moves during transit (§2 below). The prior reaching the selector is `mean_chosen_snr_db` — one mission-mean scalar, identical across every candidate in a batch, which is one reason four of the eleven feature slots are constant. | `channel.py` → `MuleConfig.rf_prior_snr_db` → `selector/features.py:49` slot 10; L2 record §4.6 |
| **3** | The return path · flight → plan | **What flight time learns never reaches the plan.** Two halves of one broken return. S3b *does* model travel at `cruise_speed_m_s` and *does* test the mission budget — but dwell is a constant `session_time = 1.0 s`, so a slow band and a fast band cost the same at a stop; the clock carried back is an assumption, not a measurement. And the one edge that does carry realised state upward, `_remaining_is_feasible()`, only **aborts**; it never re-plans. Q2's re-planning and this cut are the same piece of work. | `stages/s3b_feasibility.py`; `mule/mule_main.py::_remaining_is_feasible`; Freeze D2 |

> **Correction to an earlier framing of cut 3.** It was first described as "no travel-time model in
> Exp 4." That is wrong: S3b models travel. What is constant is *dwell*. The precise statement
> matters because it names the exact constant (`session_time`) that blocks rate from feeding back
> into feasibility — and it is an uncited platform placeholder besides.

### 1.2 The closed loop, as a workflow

§1.1 says what is severed. This says what the system does once it is not — which subsystem acts
when, where the three decisions are made *together* rather than handed downstream, and how the loops
close. Rendered as swimlanes in
[`figures/joint_optimisation_workflow.svg`](figures/joint_optimisation_workflow.svg) (standalone,
literal colours, usable in slides) and in the HTML companion. In text:

```
                 PLAN — once per mission                         │   FLY — once per stop
                                                                 │
SCHEDULER   Demand ─────────────────────► S3b feasibility ──┐    │  Collect update       Round closes
 (L2)         │  candidates · urgency          ▲  predicted │    │       ▲                    ▲
              │                                │  cost      │    │       │ served             │ queue empty → dock
            ┌─┼────────────────────────────────┼──────────┐ │    │  ┌────┼────────────────────┼──────────────────┐
RF          │ ▼                                │          │ │    │  │  Select band ──dwell·clock──┐              │
 (L1)       │ Contact band class ◄─ iterate ─ Phase at    │ │    │  │  → rate → dwell             │              │
            │   → range              arrival → rate·dwell │ │    │  │       ▲ observed phase      ▼              │
            │   │ range                 ▲ arrival times   │ │    │  │       │              Still feasible? ─yes─► Next stop
TRAJECTORY  │   ▼                       │                 │ │    │  │  Arrive at stop k ◄────────────────────────┘ k+1
 (NAV)      │ S3a clustering ──stops──► Route + order     │ │    │  │       ▲     JOINT · (band, next stop)  │ no
            │ JOINT · search over (band class, route)     │ └────┼──┼───────┘ committed plan                │
            └─────────────────────────────────────────────┘      │  └─────────────────────────────────────────┘
                                          ▲                      │                                            │
                                          └──── re-plan the remaining queue (today: abort only) ◄────────────┘
                     ◄──────────────────────────── next mission's demand ──────────────────────────────────────
```

**PLAN.** Band class, clustering and route form a search cycle: each candidate band class implies a
range, the range a clustering, the clustering a route, the route a set of arrival times, and the
arrival times a predicted phase — and therefore a predicted rate and dwell — at every stop. That
predicted cost is what S3b gates before commit, so **the gate sees a route priced by the band that
will serve it, not by a constant.**

**FLY.** The band chosen at each arrival sets the dwell, the dwell moves the clock, and the clock
decides whether the committed queue is still feasible. If not, control returns to the plan.

**Heuristic vs learned.** A heuristic fills each JOINT enclosure with a fixed rule — mean-SNR band
class and nearest-neighbour route; best band now and follow the order. A learned policy fills each
with a value function over the joint choice, anticipating what the other enclosure will find. §5
decides which is warranted.

**The cuts on this picture.** Cut 1 is the absence of the *Contact band class* box. Cut 2 is
*Select band* running once per mission instead of once per arrival. Cut 3 is the dashed re-plan edge
that today only aborts, plus *Phase at arrival* reading a constant instead of a rate.

---

## 2. The sharpest case for learning — computed from `drone_env.py`

`hermes_rl/drone_env.py` already has the joint action `(waypoint, base station, channel)` chosen
*at every step*, transit consuming time with no transfer, and `alpha > beta` so the time-varying term
dominates distance. That is the flight-time clock, and it is already the right one. Put its constants
against a real sortie and the cost of running the band decision on the plan-time clock is arithmetic.

Three bands at evenly spaced phases (as in `channel.py`), `omega = 0.15` → period ≈ 41.9 steps,
`drone_speed = 1.0`, tour W0→W3→W4→W1→W2:

| Stop | t | Committed band (best at t = 0) | Best available at arrival | Loss |
|---|---|---|---|---|
| W3 | 39 | +0.91 | +0.91 (same band) | 0 |
| W4 | 64 | **−0.98** | +0.64 | 1.63 |
| W1 | 118 | +0.41 | +0.59 | 0.18 |
| W2 | 150 | **−0.87** | +0.86 | 1.73 |
| **mean** | | **−0.14** | **+0.75** | |

Hops are 25–54 steps against a ≈42-step period — **0.6 to 1.3 of a full cycle each** — so the band
that is genuinely best at takeoff lands in a trough at two of four stops.

Deliberately simplified (equal amplitudes, no per-band gain `g(c)`, no switch penalty `λ(c,t)`), and
note what the simplification costs rather than flatters: with equal amplitudes *every* fixed band
averages ≈0 over a full cycle, so **none of this gap comes from committing to the wrong band. All
of it comes from committing at all.**

The figure also shows why the two clocks cannot simply be merged: the arrival times that set the
phase are themselves a product of the plan-time route. Moving the band decision to flight time does
not remove the commitment — it makes the commitment *consequential*.

**Make transit-time / channel-period an explicit swept parameter.** It is the knob that turns a
learned policy's advantage on and off (transit → 0: the committed band is still right on arrival and
the heuristic wins; transit ≫ period: the takeoff observation carries no information). A sweep over it
gives a crossover curve rather than a contested number — the same shape as the A1/A2 surface.

<details>
<summary>Reproduce the table</summary>

```python
import math
W={'W0':(20,70),'W1':(50,60),'W2':(80,70),'W3':(35,35),'W4':(15,20)}
tour=['W0','W3','W4','W1','W2']; om=0.15; ph=[0,2*math.pi/3,4*math.pi/3]
t=0; arrivals=[]
for a,b in zip(tour,tour[1:]):
    t+=max(1,math.ceil(math.dist(W[a],W[b])/1.0)); arrivals.append((b,t))
for b,tt in arrivals:
    v=[math.cos(om*tt+p) for p in ph]
    print(b, tt, f"committed={v[0]:+.2f}", f"best={max(v):+.2f}")
```
</details>

---

## 3. The five questions

| # | Question | Answer |
|---|---|---|
| **Q1** | Include a trajectory navigation layer? | **Yes — half of it.** Distance and order: add (travel cost is the binding constraint; the machinery exists in Exp 3 `sim_env` and S3b and is disconnected, not missing). Environment and obstacles: **don't** — solved motion planning, adds path length you could sample, buys complexity without coupling. |
| **Q2** | Make it dynamic? | **Yes — in one sense only.** Re-plan when the world diverges from the plan, not react to terrain. `_remaining_is_feasible()` is the seed; today it aborts, make it re-plan. Keep the L2 record's boundary: running out of *time* is foreseeable, a *random link failure* is not — a property of the model, not a gap. |
| **Q3** | Jointly optimise RF band and navigation? | **Yes, and this is the contribution** — but the joint variable must carry the coupling. Not one chain but two, hinged: `demand → contact band → range → clustering → route → commit` at plan time; `arrive → observed phase → band → rate → dwell → clock` per stop. **Test for system vs stack:** cut the band→range edge and see whether the result moves. |
| **Q4** | Connect RF/nav back to the FL job scheduler? | **Bilevel, three edges — on both clocks.** See §4. |
| **Q5** | Enough to justify RL over a cross-heuristic? | **It justifies testing, not asserting.** The cross-heuristic is the baseline, not the threat. Aim for *"here is where a decomposed heuristic provably stops working, and we locate that boundary empirically"* — the A1/A2 crossover surface is already a paper in that voice. |

---

## 4. The architecture — three edges, twice

The scheduler is the outer problem (which devices are worth acquiring), RF-plus-navigation the inner
problem (can we reach them, at what cost), the coupling bidirectional because value depends on cost
and feasibility depends on which devices you selected — **and the inner problem is solved twice,
once as a commitment before takeoff and again as a reaction at every stop.**

| Edge | Plan time — once, before takeoff | Flight time — again at every stop |
|---|---|---|
| **Demand** ↓ sched → RF/nav | Who needs service, how urgently, what each update is worth. *S3 deadline + bucket classify — works today.* | The committed queue: which stop is next, how much clock is left. *`MuleConfig.mission_budget_s` — works today.* |
| **Coupling** ↔ RF ↔ nav | Contact band → range → clustering → stops. **A commitment.** *Cut 1 — `rf_range_m` is a constant; there is no band variable here.* | Route → arrival → phase → rate → dwell. **Revisable.** *Cut 2 — the decision runs per mission; the prior is a mission mean.* |
| **Return** ↑ RF/nav → sched | Not a route — a *feasible set plus a cost vector*, consumed as a hard gate. *S3b gate · `features.py` slot 10 — works today, on assumed costs.* | Realised clock vs planned. Should trigger a re-plan. *Cut 3 — dwell is constant so the cost is wrong; the edge can only abort.* |

Read down the first column: **the plan-time interface is complete.** Everything missing is in the
flight-time column or the plan-time coupling. Both sockets for the return edge already exist (S3b;
slot 10), so this is a rewiring, not a rewrite. It also preserves the architectural guarantee
untouched — the hard gates still run before anything learned ranks anything.

---

## 5. The RL test — three properties, and which clock each lives on

Design the experiment to *measure* these, not assert them. Two split by clock and need not agree.

What the properties are choosing between is drawn in
[`figures/heuristic_vs_learned_fillings.svg`](figures/heuristic_vs_learned_fillings.svg): the two
JOINT enclosures from §1.2, each filled two ways. Enclosures, inputs and outputs are identical across
each row — that is what makes the two fillings comparable *arms* rather than two architectures.

| Slot | Heuristic — a fixed rule | Learned — a value function |
|---|---|---|
| **PLAN** | Band class ← argmax mean SNR. Route ← nearest-first at that range. Cost ← Σ travel + dwell at mean rate. **One pass:** band before route, route before phases. The route and the demand are *not consulted* when the band is chosen. | `V(class, route \| demand)`, trained on realised round closure, scores each **(band class, route) pair together**; argmax over pairs; **iterates**. Consults the route, the demand and mean SNR. |
| **FLY** | Band ← argmax rate at the phase *now*. Next stop ← the committed order. Feasible? → serve, else abort. The remaining queue and clock left are *not consulted*. A good band now can burn the clock stop k+1 needs. | `Q(s, band, next stop)`, trained on realised round closure, chooses **band and next stop jointly**. Consults phase now, remaining queue, clock left. Will take a worse band now to reach k+1 at a better phase — the trade §2 priced. Feasible? → serve, else re-plan. |

The three properties map onto visible features of that figure: **(a)** asks whether the iterate
loop is worth having; **(b)** whether the extra inputs carry temporal credit; **(c)** whether any
single fixed rule could have consulted them correctly across regimes.

| Property | Plan time | Flight time |
|---|---|---|
| **(a) Decomposition** — *the coupling itself; does not split* | *(spans both)* Planning against a nominal rate and reacting later is measurably worse than planning that anticipates what flight time will find. **Measure:** offline oracle on N ≤ 6, exhaustive over band class × clustering × route × per-stop band, against each decomposition; report the optimality gap of each. **Falsified by:** gap ≈ 0. A large gap on the cross-clock split but a small one *within* plan time is the likely and useful outcome: keep a heuristic planner, learn only the flight-time policy. | |
| **(b) Delayed consequence** — *two horizons* | Does a clustering commitment at mission *m* change the demand at *m+1, m+2* through the devices it starved? **Measure:** sweep γ across the mission horizon. **Falsified by:** flat curve → a greedy planner suffices. | Does a band/dwell choice at stop *k* change whether *k+1, k+2* stay reachable? **Measure:** same sweep, γ across stops within one sortie (4–6 steps, not 4–6 missions). **Falsified by:** flat curve → a one-step rule at each arrival suffices. |
| **(c) Non-stationarity** — *two sources* | Does the winning weight vector *change* across the clean/jittery × tight/slack grid? **Measure:** tune the cross-heuristic per regime; compare the vectors, not just scores. **Falsified by:** one vector wins everywhere → publish the heuristic. | Already answered by §2: the channel moves *within* a sortie, so no per-mission vector can track it in principle. **Measure** not *whether* but *how much* — sweep transit/period. **Falsified by:** nothing, and say so; settled by construction, earns no credit on its own. |

Keep the claim rule: bootstrap CI excluding 0 **and** paired Wilcoxon p < 0.05, multiplicity
position stated.

> **Read the (b) row as a decision, not a result.** Its two cells decide what gets built, and the
> outcomes are different projects: flat on **both** → no temporal credit anywhere, build the joint
> heuristic, write the negative result. Live on **flight time only** → learn the per-arrival policy
> under a deterministic planner (small, well-scoped). Live on **both** → hierarchical policy,
> plan-time options over a flight-time policy — substantially bigger, not to be entered without
> this evidence.

---

## 6. Baselines — which capability earned the gain

### 6.1 Why "static vs dynamic" is the wrong axis

The to-do doc frames FedCS/Oort as *static* against a *dynamic* comparator. Oort is already
adaptive (per-round utility with a staleness bonus); grouping it with a learned trajectory policy
groups two things that share nothing. Worse, if HERMES wins and the win is attributed to being
"dynamic," nobody can tell whether it came from **choosing where to fly**, **knowing about FL**, or
**learning**. A spectrum cannot answer that. A factorial can.

### 6.2 The 2 × 2

Two axes genuinely separate the candidates. **Decision scope:** does the policy choose the trajectory,
or only the order of service along a route it is handed? **Objective denomination:** FL units
(update utility, deadlines, round closure) or physical units (bytes, age)?

| | **FL-blind** — bytes, age | **FL-aware** — updates, rounds, deadlines |
|---|---|---|
| **Selection-only** — permutes within a given route | **MAX-AoI greedy** — B1, exists | **Oort statistical utility** — B2, exists (direct rival to Φ-widening) · **FedCS, degraded** — capability contrast, last-known state substituted for its Resource Request |
| **Trajectory + selection** — chooses the next stop | **DQN over the contact graph, after Chen et al.** — new, after cuts 1–3; optimises bytes, has no notion a second update from a served device is worthless | **Cross-heuristic** — new; two clocks, deadline-aware, fixed rule · **HERMES** — new, redefined; same architecture, learned flight-time policy |

**Three tests fall out, and the third is the paper.** *Scope effect* — read down the columns.
*Awareness effect* — read across the rows. **Interaction** — is having both worth more than the sum
of having each? That is the two-clock coupling claim in statistical form: if the coupling is real,
trajectory control is worth *more* when the objective is FL-aware, because FL-awareness is what
tells the trajectory which stops matter and when. Paired seeds and `experiments/analysis/stats.py`
test it directly (difference of differences).

**The RL question is nested inside the bottom-right cell, not the headline.** HERMES vs the
cross-heuristic holds clock structure constant (both two-clock) and isolates learned-vs-fixed; §5
applies there. Nesting is what protects the paper: if the fixed rule wins the nested test, the result
is still *"the architecture is right, and a heuristic suffices to exploit it."*

**One data point already exists.** Old H2 — the DDQN selector — is (selection-only, FL-aware,
*learned*) and it tied H1. Learning did not help when the policy could only reorder. The factorial
tests whether it helps once the policy can fly. A published null becomes a falsifiable prediction,
consistent with §1's diagnosis that the old action had no consequence to learn from.

**The other axis — *when the ranking signal is obtained* (pre-selection reporting vs
retrospective) — is a capability axis, not a performance axis.** It decides who is *admitted* to the
grid (why FedCS runs degraded and Oort runs faithfully). It belongs in Related Work as the argument
for the arm list, per [`HERMES_SOTA_Baseline_Candidates.md`](../../HERMES_SOTA_Baseline_Candidates.md) §3.
It is not a column in the results.

### 6.3 Chen et al. (GLOBECOM Wkshps 2023) — what it is, what transfers

*Model-aided Federated Reinforcement Learning for Multi-UAV Trajectory Planning in IoT Networks* —
Chen, Esrafilian, Bayerlein, Gesbert, Caccamo. Six pages; code BSD-3 at
`github.com/Cirrick/Multi_UAV_Data_Harvesting` (PyTorch, QMIX/IQL after `starry-sky6688`, SMAC-style
env). Three UAVs, ten static devices, 3D city grid (RBM 600×800 m, RDM 1000×1200 m). Same group as
Mestoukirdi et al., already cited as the closest architectural prior.

Three separable ideas: a **learned digital twin** (channel NN from real measurements + PSO
localisation of unknown devices, train in the twin); **QMIX** for multi-UAV coordination;
**parameter averaging** between UAVs every 50 episodes, which the title calls federated.

| Piece | Verdict | Why |
|---|---|---|
| **Trajectory DQN as a baseline** | **Yes — in E3** | Against the *current* harness it fails the test the baseline doc applied to Mestoukirdi (a joint method dropped into `target_selector` is their metric inside our router). Against the *redefined* HERMES it becomes the flight-time competitor — a trajectory policy *is* what the flight-time loop is — and the gap to HERMES isolates deadlines, readiness, the plan-time commitment and starvation handling. Nothing FedCS or Oort can give, because they cannot fly. |
| Learned twin | Later — journal scope | HERMES already trains offline (Exp 3 `sim_env`) and evaluates on the real stack. What they add is learning the channel from real traces instead of `channel.py`'s sinusoid — exactly what cut 2's predicted-SNR-at-arrival wants. A genuine, separate contribution. |
| QMIX | Not now | One mule. QMIX with one agent *is* DQN, which exists. Relevant only if multi-mule enters scope (L2 record §4.7). |
| Federated averaging | Skip | Every UAV trains in the *same* twin; nothing is heterogeneous. Their text concedes diversity comes from ε-greedy. Parallel exploration, not FL over distinct data. |

**Fidelity caveats that must be stated if used.** Their action space is grid moves; HERMES's is
waypoint-to-waypoint, so the code cannot run as-is. The honest port is a single-agent DQN over
HERMES's contact graph with their observation design (per-device SNR, distance, remaining data for
reachable devices — all mule-visible), labelled **"DQN over the contact graph, after Chen et al.,"
never "FedQMIX."** Their objective is bytes; a harvesting policy will over-collect from good-channel
devices. That is the expected failure mode and what makes it meaningful — report `update_yield` and
`round_close_rate`, never raw bytes.

**As a paper.** Three random runs, no statistics; "three orders of magnitude" is read off a log-scale
plot. The 1000× is by construction (N = 1000 simulated episodes per real one; simulator cost never
reported). Baselines are QMIX, IQL, model-aided QMIX — **RL against RL only; no heuristic baseline
at all.** Once positions are known this is prize-collecting orienteering with a rate function; a
greedy partition is the obvious comparator and is not run. That is the omission step 5 below exists
to prevent. For the to-do doc's compute-cost line: Oort is a formula, MAX-AoI is a sort, this needs a
thousand simulated episodes per real one.

Their **safety controller** (mask actions that leave the UAV unable to reach its terminal, then let
the policy choose among what is left) is the same gate-then-learn pattern as S3b and the "learning
cannot override hard constraints" guarantee — an independent convergence worth citing. Cite alongside
Mestoukirdi et al. and **Bayerlein et al., IEEE OJCOMS 2021** (the multi-UAV DDQN harvesting work
Chen builds on; not yet in the Related Work notes). Together they fill the empty "Adaptive:" line in
the to-do doc, relabelled *trajectory-and-selection*.

---

## 7. Sequence

Ordered by dependency, grouped by kind of work. Step 1 touches no architecture and could settle the
question before anything is built; steps 2–4 close one cut each, in the order the diagram reads.

**Instrument — before touching architecture**

1. **Fix the measurement.** *(no new code)* τ = 0.82 time-to-accuracy; run where S3b binds (~60 s,
   not 120 s). May reveal signal the current setup structurally cannot see.

**Close the loop — one cut per region**

2. **Plan time · the commitment — introduce a contact-band decision that sets `rf_range_m`.**
   *(small · cut 1)* A second band variable on the device link, distinct from L1's backhaul band.
   Immediately testable by ablation: cut the edge and see whether the result moves.
3. **Flight time · the reaction — move the band decision from per-mission to per-arrival.**
   *(small · cut 2)* Re-select from the phase actually found; make slot 10 a per-contact predicted SNR
   at estimated arrival. Closes the anticipation edge and the constant-feature problem together.
4. **The return path — rate-dependent dwell, and re-plan instead of abort.** *(medium · cut 3)*
   Port Exp 3's cost model into the Exp-4 path; replace constant `session_time` with a function of
   achieved rate; extend `_remaining_is_feasible()` from abort to re-plan. Q2 and cut 3 are one piece
   of work; either alone leaves the loop open.

**Compare, then decide**

5. **Fill the 2 × 2.** *(medium)* B1/B2 occupy the selection-only row. Add the cross-heuristic
   (trajectory, FL-aware, fixed) and the DQN over the contact graph after Chen et al. (trajectory,
   FL-blind), sharing transport, realism and paired seeds. Report compute cost per arm. The
   interaction is the headline; HERMES vs cross-heuristic is the nested RL test.
6. **Then decide on RL, using (a)–(c).** *(gated on 1–5)* Not before. Every step above is worth
   doing whichever way the decision goes — which is what makes the ordering safe.

---

## 8. Scope — this is E3, not a new front

The holistic revision plan records E3 as *"the largest untouched block of work in this plan,"* and
Freeze D4 settles that E4 makes no RL claim and the RL question belongs to E3. This redefinition
does not open a new thread; it finally specifies the experiment the plan reserved for exactly this
question. **The ICNC submission does not have to carry it.** E4's crossover surface stands alone as
an availability result; the RL question ships with E3 on its own timeline.

Protect the framing that already makes E4 credible — *HERMES is not a general improvement on
federated learning; it is an availability mechanism, and it costs throughput when availability is
not the problem* — and apply the same discipline to RL.

---

## 9. How the analysis evolved — corrections log

Recorded because several of these overturned an earlier framing, in the same spirit as the
"retracted claim" and "correction to earlier drafts" notes elsewhere in these docs.

| Date | Change | What it corrected |
|---|---|---|
| 08-26 | Initial memo: diagnosis (flat reward), coupling as a six-node ring, five answers, (a)–(c), sequence. | — |
| 08-26 | **Cycle redrawn as two clocks with demand as entry.** | The ring put *band choice* at the origin and drew the *current* per-mission band decision while arguing for the prototype's per-stop one — the two halves of the argument were inconsistent. Team review caught it. |
| 08-26 | **Cut 1 restated:** L1 is backhaul-only; the contact link has *no* band variable. | "Wire L1's band to range" understated the work; a second band decision must be introduced. |
| 08-26 | **Cut 3 restated:** S3b *does* model travel; dwell is the constant. | "No travel-time model in Exp 4" was false. Names `session_time` as the blocking constant. |
| 08-26 | Cut list, sequence, Q4 table, test matrix regrouped by clock; cut 3 reframed as the *return path* and merged with Q2's re-planning. | Three structures had shared no spine with the diagram. Regrouping exposed that Q2 and cut 3 are one task and that the plan-time interface is complete today. |
| 08-26 | Anticipation figure rebuilt from a single hop to a full sortie with both clocks on one axis (§2 table). | Single-hop version showed no clock structure; sortie version produced the −0.14 vs +0.75 result. |
| 09-15 | Chen et al. + repo assessed against the baseline criteria. | Filled the "Adaptive:" slot in the to-do doc; identified Bayerlein 2021 as a missing citation. |
| 09-16 | **"Static vs dynamic" replaced by the 2 × 2** (§6). | Static/dynamic conflated three capabilities and could not attribute a win. |
| 09-17 | Joint-optimisation workflow added (§1.2; swimlane figure). | The two-clock diagram showed structure and cuts, not process. The workflow shows which subsystem acts when, and draws the two joint decisions as enclosures rather than as a chain. |
| 09-17 | Heuristic-vs-learned pair added (§5; 2×2 figure). | Both fill the *same* enclosures; the figure makes the difference mechanistic — what reaches the decision, and whether the search iterates — and ties each of (a)–(c) to a visible feature. |

Rendering defects fixed along the way (grid auto-placement in the sequence list; circled-digit
glyphs absent from IBM Plex Mono) are recorded in the artifact's version labels and do not affect
the content.

---

## 10. Open items

- [x] Standalone HTML copy alongside this file:
      [`HERMES_Layer_Redefinition_and_RL_Decision.html`](HERMES_Layer_Redefinition_and_RL_Decision.html)
      (artifact version 12, 2026-09-17). `hermes-rl-decision.html` is the **first** publish
      (2026-08-26) and is deliberately retained unchanged as the pre-correction baseline.
- [ ] Add Chen et al. 2023 and Bayerlein et al. 2021 to
      [`HERMES_Related_Work_Notes.md`](../../HERMES_Related_Work_Notes.md) §3 (UAV-specific thread) and §7
      (citation readiness — both need full-text verification marks).
- [ ] Relabel the to-do doc's "Static / Adaptive" split as *selection-only / trajectory-and-selection*.
- [ ] When step 1 runs, record whether τ = 0.82 alone surfaces an H2-vs-H1 effect — it changes how
      much of §1 is load-bearing.
- [ ] The (b)-row outcome (§5) decides the RL architecture; do not start step 6 without it.

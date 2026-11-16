# The simplified trip-planning problem

CMP_SC 8050 project, Phase 1 (steps 1.1 to 1.8 of `DeveloperDocs/FeRRy_CS8050_Project_Plan.md`), 16 November 2026. The plan checker is `experiments/cs8050/problem.py`, tested in `tests/unit/test_cs8050_problem.py`. Decisions 1.5 and 1.6 were settled on 16 November 2026.

Every proof and every solver in the project uses this one problem. It keeps the planner's core: one radio setting per trip, stops within reach, deadlines and a time limit. It leaves out the rules listed under step 1.8.

## Instance (step 1.1)

* **Devices.** $n$ devices. Device $j$ has a position $x_j \in \mathbb{R}^2$, an importance $w_j > 0$, a deadline $D_j \in [0, \infty]$ in seconds after takeoff, and a data size $s_j \ge 0$ in bytes.
* **Radio settings.** A set $B$ of settings; FeRRy has three (wide, medium, narrow). Setting $b$ has a reach $r_b > 0$ and a rate $\rho_b(d) > 0$ in bytes per second for a device $d \le r_b$ metres from the stop. The rate is a step function that never increases with $d$. A device's send time at a stop $p$ is $\sigma_b(j, p) = s_j / \rho_b(\lVert p - x_j \rVert)$. The proposal's model, one rate per setting, is the one-step case.
* **Flight.** A dock $o$, a speed $v > 0$, a time limit $T$, and an upload time $u \ge 0$ charged when the drone lands with data.
* **Sites.** A finite set $P$ of places where the drone may stop (decision 1.5).
* **Travel time.** $\tau(p, q)$, the straight-line distance over $v$. The proofs round it up to a multiple of a fixed quantum $\delta$; the experiments keep it exact (decision 1.6).

## Plan (step 1.2)

A plan picks one setting $b$ and $m \ge 0$ stops $(p_1, S_1), \dots, (p_m, S_m)$ in flying order:

* each $p_k$ is a site in $P$, and a site may appear more than once;
* each $S_k$ is a non-empty set of devices, and the sets are pairwise disjoint, so no device is served twice;
* every device $j \in S_k$ is in reach: $\lVert p_k - x_j \rVert \le r_b$.

With $m = 0$ the plan is empty: the drone never takes off and pays no upload.

## Rules (step 1.3)

The drone takes off at time 0. It reaches stop $k$ at $a_k$ and finishes there at $f_k$:

$$a_1 = \tau(o, p_1), \qquad f_k = a_k + \sum_{j \in S_k} \sigma_b(j, p_k), \qquad a_{k+1} = f_k + \tau(p_k, p_{k+1}).$$

A plan is valid when, besides the reach condition above:

1. **Deadlines.** Each stop finishes by the deadline of every device it serves: $f_k \le D_j$ for all $j \in S_k$.
2. **Time limit.** For $m \ge 1$, the drone is home with the upload done in time: $f_m + \tau(p_m, o) + u \le T$.

The deadline binds the stop's finish, not the drone's arrival. This follows the code (step 1.7). The proposal and the first wording of step 1.3 said "reached before its deadline".

The code checks the time limit at every stop, as if the drone flew home right after it. That is the same as checking the last stop: when $\tau$ obeys the triangle inequality, $f_k + \tau(p_k, o)$ never decreases along the route. Rounding up keeps the triangle inequality, because $\lceil x + y \rceil \le \lceil x \rceil + \lceil y \rceil$.

## Goal (step 1.4)

Maximize the served importance $\sum_{k} \sum_{j \in S_k} w_j$ over valid plans. The empty plan is always valid, so the best value is at least 0.

**Decision version (PLAN).** Given an instance and a number $W$, is there a valid plan worth at least $W$? The hardness proofs of Phase 2 are about this question.

Ties do not matter here. The planner breaks them by mission time, energy and link loss, but the project measures served importance only.

## Decision 1.5: stops at a finite list of sites (settled 16 November 2026)

The site list $P$ is part of the input, so "the best plan" means the best over the listed sites.

* **The planner works this way.** It only searches the stops it is offered, so its `exact` mode is exact over a finite list too.
* **The solver stays exact.** With stops anywhere in the plane, where to stop becomes a continuous problem of the "traveling salesman with neighbourhoods" kind [5]. No exact method for it scales.
* **The reductions stay clean.** The hardness proofs put the sites at the device positions.

**Two lists in the experiments** (step 3.6):

| List | Sites, per setting | What the gap against it measures |
| --- | --- | --- |
| Planner's stops | The stops the planner is offered | What the route search and the choice of devices lose: the headline gap |
| Full list | The planner's stops, the device positions, and every point where two devices' reach circles cross | Also what stop placement loses, on the sizes the solver reaches |

The full list loses no group of devices. If some point serves a group, so does a device position or a crossing point: slide the reach circle until two of the devices sit on its edge. It can lose flying distance, because the best spot for the route may lie between listed points. With up to about $n^2$ sites per setting, it is the list most likely to make the solver too slow (Risk 1).

**Kept as defined above:** a device may be served at any listed site in reach, and a site may be visited twice. The planner fixes each device to its own group's stop and never repeats a stop, so the gap also counts what fixed groups lose. A repeat visit can only pay off through deadlines: without them, merging two visits to one site never makes the trip longer.

## Decision 1.6: rounded in the proofs, exact in the experiments (settled 16 November 2026)

* **Why round in the proofs.** Claiming the problem is in NP needs a polynomial-time check of a plan. With exact Euclidean times, that check compares sums of square roots, and no polynomial-time method for that is known. Rounding each leg up to a multiple of a fixed rational $\delta$ makes every time a rational number that is computed exactly in polynomial time.
* **What the report claims.** NP-complete for rounded times, for any fixed $\delta$. NP-hard for exact times, with membership in NP left open.
* **Why exact in the experiments.** The planner computes exact float distances and stays unchanged, so the solver does the same. The checker supports both (`travel_quantum_s`); every experiment leaves it at None.
* **Rounding changes tight verdicts.** In the tests, a valid plan turns late at $\delta = 3$ s. Measured results never use rounding, so this never touches them.
* **One hardness proof covers both.** See the grid-graph route under Leads for Phase 2.

## What the code does (step 1.7)

| Question | Answer in the code | Where |
| --- | --- | --- |
| Does a deadline bind the arrival or the end of the transfer? | The end of the whole stop: arrival plus the dwell of every device served there, against each served device's deadline. A stop carries the earliest deadline of its served devices. | `FeasibilityModel.admit` with the default `deadline_bounds = "collection"` (`stages/s3b_feasibility.py`); `stop_deadline` (`stages/s3d_age_cap.py`) |
| Does the sending rate depend on distance? | Yes. The rate is a step function of the mean signal-to-noise ratio (CQI 1 to 15), which falls with distance. Inside reach the mean is at least 5.13 dB above the floor, so every device in reach sends at a positive rate. | `ContactLink.rate_bps` and `mean_snr_db` (`l1/contact_link.py`); `FerryRuntime.member_dwell_s` (`mule/ferry.py`) |
| Do devices send different amounts of data? | No. In a pass every device sends the same bytes (the push plus the update), so send times differ only through distance. | `session_bytes` (`mule/ferry.py`) |
| What does the time limit cover? | The collection pass: flying, dwell, the return leg and the upload at the dock. The delivery pass and the dock turnaround are not in it; they enter the score only. | `admit`'s budget clause; Pass 2 in the notes of `plan/plan_search.py` |
| How is travel priced? | Straight line in the plane at the cruise speed. | `FeasibilityModel.cost` |
| How are plans ranked? | After the age cap's key, by served importance first under the default `coverage_rank = "lexicographic"`. Mission time, energy and link loss only break ties. | `plan_key` (`plan/plan_score.py`) |

## Rules the simplified problem leaves out (step 1.8)

Each row says how to turn the rule off in the planner for the comparison in step 3.8.

| Planner rule | What it does | How to turn it off |
| --- | --- | --- |
| Age cap | Devices unserved for too long must be served, are exempt from their own deadline, and a plan leaving one out ranks last | `AgeCapSpec(s_missions=None)`, i.e. `age_cap_missions=None` (the default) |
| Hover stops | Adds a one-device stop for a capped device its group cannot serve alone | `PlanSearchParams(hover_stops=False)`; it already does nothing once the cap is off |
| Energy limit | A battery limit in the check | `FerryPhysics(energy_capacity_j=None)` (the default) |
| Delivery deadlines | Other readings of what a deadline bounds | Keep `FerryPhysics(deadline_bounds="collection")` (the default) |
| Score terms: mission time (with the delivery pass and the turnaround), energy, link loss | Make up the score V | Nothing needed: under the lexicographic rank they only break ties between plans of equal served importance. To silence them anyway: `PlanScoreParams(c_energy=0, c_link=0)` |
| Weighted rank | Ranks by V alone, trading importance against time | Keep `coverage_rank="lexicographic"` (the default) |
| Whole stops | Serves a stop whole or not at all | Keep `member_admission="subset"` (the default) |
| Speed factors (arm F-pref) | Scale each weight by Oort's speed factor | Off by default (`demand_weights(speed=None)`) |

### Differences in shape, not rules

The planner searches a smaller set of plans than this problem allows:

* **Fixed groups.** The planner groups devices once per setting and can only drop devices from a group. A device is served only at its own group's stop. Here a device may be served at any site in reach.
* **Stop positions.** The planner's stops are its groups' stops (a group's centre, or its anchor device when the centre loses someone). Here the stops are the site list.
* **No repeats.** The planner never visits a stop twice. Here a site may repeat.
* **Weights.** The planner computes $w_j$ from age and missed visits (`demand_weights`). Here they are given.

So the best plan here is at least as good as the planner's `exact` answer. Step 3.8 makes the like-for-like comparison on the planner's own set of plans: each device at its own stop, no repeats.

## Leads for Phase 2

* **The knapsack reduction (step 2.3) needs unequal data sizes.** The proposal's model has them; the code's instances do not. In the code every device sends the same bytes and a setting has at most 15 rates. With one stop and that few distinct send times, the problem may be solvable in polynomial time. Step 2.6 should settle this, and the report should say which model each hardness result covers.
* **The traveling salesman reduction (step 2.2), through grid graphs.** Hamiltonian cycle stays NP-complete on grid graphs: finite sets of integer grid points, joined when exactly 1 apart (Itai, Papadimitriou and Szwarcfiter, SIAM J. Computing, 1982; a source to add, and to check in Phase 2). Put a device on each of the $n$ points with weight 1, no data and no deadline. Put the sites at the devices and the dock at one of them, with reach below 1, $v = 1$, $u = 0$ and $T = n$. Serving all $n$ devices then means a closed trip through all $n$ points: at least $n$ legs between different points, each at least 1 long. So a plan worth $n$ flies exactly $n$ legs of length 1, which is a Hamiltonian cycle. Two grid points that are not joined are at least $\sqrt{2}$ apart, and rounded up to whole units $\sqrt{2}$ becomes 2, so the same proof holds for exact and for rounded times.

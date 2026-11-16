# FeRRy CS8050 Project Plan

16 November 2026. Snapshot of the tracked plan at https://claude.ai/artifact/DdghhyU4qGSFP6oW8dNfFS, which is the copy where steps are ticked and statuses are set. Source: the CMP_SC 8050 final project proposal deck (October 2026).

## Summary

The project answers one question: is FeRRy's trip-planning problem NP-hard, and how close does FeRRy's plan get to the best possible plan for 7 to 20 devices? It is a solo, six-week final project for CMP_SC 8050. The work is four theory pieces (T1 to T4) and one comparison study, run side by side so each experiment can check a proof as it lands.

| Question | Answered by | Work that answers it |
| --- | --- | --- |
| Q1. How hard is the problem? | A proof | T1: NP-hardness by converting the traveling salesman and knapsack problems into it, plus the simpler versions that stay fast |
| Q2. How close does FeRRy get? | Proof and experiments | T2: exact solver. T3: limits on the shortcuts. Gap study on 7 to 20 devices |
| Q3. When does long reach win? | A formula and experiments | T4: tipping-point formula, checked against the setting the planner picks |

The project is done when six things exist: written proofs for T1 to T4, a repeatable layout generator, an exact solver, a test harness that runs FeRRy's planner unchanged, the charts, and the dataset with the final report and presentation.

## Starting point

The planner and everything needed to run it already exist in the repo; no course-project code or proof exists yet. The planner stays unchanged throughout. All new code goes in a new package that only imports from it (proposed: `experiments/cs8050/`, tests in `tests/unit/test_cs8050_*.py`, outputs in `results/cs8050/`).

| Already built | Where | What the project uses it for |
| --- | --- | --- |
| Plan search | `hermes/scheduler/plan/plan_search.py` | The thing under study. Three searches by size: `exact` up to 6 devices, `stop_subsets` up to 6 stops, `local` above that (at most 50 passes or 2,000 walks per setting) |
| Stop placement | `hermes/scheduler/stages/s3a_cluster.py`, `cluster_by_rf_range` | The first shortcut. Anchor on the most urgent device, group every device within reach, repeat |
| Route start | `hermes/scheduler/routing/two_opt.py`, `order_contacts` | The second shortcut. A short loop that the local search then improves |
| The one check | `hermes/scheduler/stages/s3b_feasibility.py`, `FeasibilityModel.fold` | Decides whether a stop is allowed. The simplified rules must match it |
| Radio settings | `hermes/l1/contact_link.py` | Wide 20 MHz reaches 60 m, medium 5 MHz about 120 m, narrow 1.4 MHz about 232 m |
| Layout draw | `experiments/exp4/topology_builder.py`, `device_positions` | The spread-out layouts, drawn exactly as recorded trials drew them |
| Planning worlds | `experiments/analysis/age_cap_s_star.py`, `ClassWorld` | Builds the planner's inputs for a layout without flying a mission. The harness starts from this |
| Brute-force reference | `tests/unit/_p4_brute.py` | An independent check of the planner's own search family, used to test the new solver's wiring |
| Full-mission simulator | `experiments/ferrysim/` | Not used for the gap study. It stays the simulator for whole missions |

New work for the course: the simplified problem and its plan checker, the four proofs, the exact solver, the clustered and tricky layouts, the test harness, the upper bound, the tipping-point formula, the charts, the dataset and the report.

## Phases at a glance

Each coding phase starts a week after the proofs it tests, and week 6 is writing only. Weeks are counted from the project start.

| Phase | Weeks | Kind | Exit check | Status |
| --- | --- | --- | --- | --- |
| 1. Write the problem down precisely | 1 | Proofs | The plan checker is right on five hand-built plans, and the stop and travel-time choices are written down | Done |
| 2. Hardness proofs (T1) | 1 to 2 | Proofs | Both conversions are written in both directions | In progress |
| 3. Exact solver and test layouts (T2) | 2 to 3 | Code and runs | The solver equals enumeration, the planner's `exact` search equals the solver with the extra rules off, and the largest solvable size is known | Not started |
| 4. Shortcut limits and tipping point (T3, T4) | 3 to 4 | Proofs | The placement limit is proved or countered, one route-search bad case is measured, and the formula has a test | Not started |
| 5. Experiments and parallel test | 4 to 5 | Code and runs | One command reruns the grid, and every chart is rebuilt from the saved tables | Not started |
| 6. Report and presentation | 6 | Writing | A fresh clone reproduces the tables, and Q1 to Q3 each have a stated answer | Not started |

## Steps by phase

Each phase lists its steps as tick boxes, what it produces, and the check that closes it. Step numbers are for use in commits and notes.

### Phase 1 · Write the problem down precisely (week 1)

Goal: one fixed definition of the simplified problem that every proof and every solver uses.

- [x] **1.1 Instance.** Write what is given: each device's position, importance, deadline and data size; three radio settings, each with a reach and a sending rate; the drone's speed, the dock and the time limit.
- [x] **1.2 Plan.** Write what is chosen: one radio setting, an ordered list of stops, and the devices each stop serves, all within that setting's reach of the stop.
- [x] **1.3 Rules.** Write the three rules. A stop lasts the sum of data size over rate for the devices it serves. Each stop finishes (arrival plus dwell) by the deadline of every device it serves, as the code does (step 1.7). Flying, stops and the trip home fit in the time limit.
- [x] **1.4 Goal.** Maximize the total importance of the devices served. Add the yes/no version the proofs need: is there a plan worth at least W?
- [x] **1.5 Where stops may be.** Decide between a finite list of allowed stop positions given in the input and stops anywhere in the plane. Recommended: the finite list, because the planner itself only searches the stops it is offered, and the exact solver stays exact.
- [x] **1.6 How travel times are given.** Decide between exact Euclidean distances and travel times rounded to a fixed precision. This decides whether "a plan can be checked fast, so the problem is in NP" can be claimed: no fast exact method is known for comparing sums of square roots.
- [x] **1.7 Match the code.** Read `s3b_feasibility.py` and record whether a deadline binds the drone's arrival or the end of a device's transfer, and whether the sending rate depends on distance.
- [x] **1.8 Dropped rules.** List every rule the real planner has that the simplified problem leaves out, each with the switch that turns it off: the age cap (`AgeCapSpec.s_missions = None`), hover stops (`PlanSearchParams.hover_stops = False`), the energy and link-loss terms (`PlanScoreParams.c_energy`, `c_link`), the delivery pass in the mission time, and the ranking order (`coverage_rank`).
- [x] **1.9 Plan checker.** Write `problem.py`: the instance and the plan as data classes, and `check_plan(instance, plan)`, which returns the served importance or the first rule broken.

Produces: a one-to-two page problem statement for the report, `experiments/cs8050/problem.py`, and the dropped-rules table.

Exit check: the checker gives the right verdict on five hand-built plans (valid, out of reach, late arrival, over the time limit, a device served twice), and decisions 1.5 and 1.6 are written into the statement.

### Phase 2 · Hardness proofs, T1 (weeks 1 to 2)

Goal: a written proof that the problem is NP-hard, and a short list of simpler versions that can still be solved fast. This answers Q1.

- [ ] **2.1 In NP.** Show that a proposed plan is checked in time polynomial in the input, following the Phase 1 checker step by step. Claim "in NP" only under the travel-time choice of step 1.6; otherwise claim NP-hard only.
- [ ] **2.2 From the traveling salesman problem.** Shrink the reach below the smallest gap between devices, and set every data size to zero, every importance to 1 and every deadline to infinity. Then all n devices can be served within the time limit exactly when a tour no longer than the limit times the drone's speed exists. Cite Papadimitriou [7] for the Euclidean case.
- [ ] **2.3 From knapsack.** Put every device within reach of one stop, so there is no routing. Data size over rate becomes an item's size, importance its value, and the time limit less the round trip the capacity. Cite Garey and Johnson [11].
- [ ] **2.4 Both directions.** For each conversion, write the yes-to-yes and the no-to-no argument, and confirm the conversion itself takes polynomial time.
- [ ] **2.5 What each proof shows.** State the two results separately: routing alone makes the problem hard even with one setting and no data, and packing alone makes it hard even with one stop.
- [ ] **2.6 Simpler versions.** Settle each one. One stop: knapsack, solved by dynamic programming in time proportional to devices times the time limit. One stop with equal data sizes: serve the most important first, which is optimal. One setting: still hard, by 2.2. A fixed small number of stops: to work out (try every order, then a knapsack over time per stop).
- [ ] **2.7 Write T1.** One theorem, two lemmas and one figure per conversion, in the report's LaTeX.

Produces: the T1 section and the table of simpler versions.

Exit check: both conversions are written in both directions. Their machine check on small cases is step 3.7.

### Phase 3 · Exact solver and test layouts, T2 (weeks 2 to 3)

Goal: the true best plan for small cases, a repeatable set of layouts, and a harness that runs FeRRy's planner unchanged on them.

- [ ] **3.1 Layout generator.** Write `layouts.py` with three seeded families. Spread out calls `device_positions`, so layouts match recorded trials. Clustered puts devices around a few centers. Tricky is hand-built and filled in Phase 4. Parameters: 6 to 96 devices, field size 200 to 500 m, data size, loose or tight time limit. Each layout saves as JSON with its seed.
- [ ] **3.2 Exact solver.** Write `exact.py`: dynamic programming over (set of devices served, current stop), keeping the earliest time the drone can be there. Earlier is never worse, because every rule only gets harder to pass as time goes on. Run it once per radio setting and keep the best. Time grows like 2 to the power n.
- [ ] **3.3 One-stop method.** Write the knapsack method from step 2.6 and test it against the exact solver on one-stop layouts.
- [ ] **3.4 Solver tests.** Compare the solver with plain enumeration (`itertools` over every setting, device subset and stop order) on 500 random cases of at most 7 devices. Every plan returned must pass `check_plan`.
- [ ] **3.5 Harness.** Write `harness.py`: build the planner's inputs for a layout the way `ClassWorld` does, call `plan_search` unchanged, and record the setting chosen, the stops, the devices served, the search used (`exact`, `stop_subsets` or `local`), the candidates scored and the planning time.
- [ ] **3.6 Two stop lists.** Let the solver run on the planner's own offered stops and on the full list from step 1.5: those stops, the device positions, and every point where two devices' reach circles cross. The first measures what the route search loses. The second adds what stop placement loses.
- [ ] **3.7 Check the conversions.** Turn small traveling salesman cases (up to 7 cities) and knapsack cases (up to 12 items) into planning cases, and confirm the solver gives the same yes or no.
- [ ] **3.8 Match the real planner (Risk 2).** With the step 1.8 rules switched off, compare the planner's `exact` search with the solver on the same stops at 6 devices or fewer, over 500 layouts. Keep `coverage_rank = lexicographic`: with the age cap off it ranks by served importance first, which is the simplified goal. If the sending rate depends on distance (step 1.7), give the solver the planner's own stop times. List every difference with its cause before drawing any conclusion.
- [ ] **3.9 How far the solver goes (Risk 1).** Time the solver from 6 to 24 devices, 20 layouts each. Record the largest size that finishes within 60 s per layout.
- [ ] **3.10 Upper bound.** Write `bounds.py`: a linear-programming bound on the best plan for sizes the solver cannot reach (SciPy 1.17 with its HiGHS solver is installed). Confirm the bound is never below the exact best wherever both exist.

Produces: the generator, solver, harness and bound with their tests, and a first dataset of layouts with their best plans.

Exit check: the solver equals enumeration on every small case; the planner's `exact` search equals the solver with the extra rules off, or each difference is explained; the largest solvable size is known.

### Phase 4 · Shortcut limits and tipping point, T3 and T4 (weeks 3 to 4)

Goal: a proved limit, or a concrete bad case, for each of the planner's two shortcuts, and a formula for when the long-reach setting wins. This feeds Q2 and answers Q3.

- [ ] **4.1 Placement rule.** Write down exactly what `cluster_by_rf_range` does: anchor on the most urgent device, take every device within reach of the anchor, and stop at the group's center if that still reaches everyone, else at the anchor.
- [ ] **4.2 Placement limit.** Prove that it uses at most a fixed multiple of the fewest stops possible. Target: 5 times, by a packing argument (anchors are more than one reach apart, and a circle of one reach holds at most 5 such points). Compare with Hochbaum and Maass [6].
- [ ] **4.3 Placement bad case.** Build a layout family that gets as close to that multiple as possible, and add it to the tricky layouts.
- [ ] **4.4 From stop count to importance served.** Fewer stops is not the goal itself. Show by example whether extra stops can cost served importance under a tight time limit, and say what the stop-count limit does and does not imply for the gap.
- [ ] **4.5 Route-search moves.** List the local search's moves from `plan_search.py`: drop a stop, insert a stop, reverse a segment, drop one device from a stop. No move swaps one stop for another.
- [ ] **4.6 Route-search bad cases.** Build layouts where the search stops short of the best plan: one where only a swap helps, and one where the default ranking takes the first insert that adds any importance. Start from the 3 of 600 random cases where one version of the search did worse than another.
- [ ] **4.7 Route-search limit.** Try to prove a limit on the gap. If no proof holds by the end of week 4, record the worst gap found and the layouts that cause it instead.
- [ ] **4.8 Tipping-point formula.** Estimate the wide setting's flying time from the random-tour formula of Beardwood, Halton and Hammersley [10], applied to its stop count. Set it against the narrow setting's extra sending time: total data times the difference in one over rate. Solve for the tipping point in device count, field size and data size.
- [ ] **4.9 Fit the constant.** The tour formula is for many points, and the planner has few stops. Fit its constant on generated layouts and state the range it holds for.
- [ ] **4.10 Skip rule.** Write `tipping.py`: the predicted best setting, and a rule for when a setting cannot win and could be skipped. The rule is tested offline in Phase 5; the planner is not changed.
- [ ] **4.11 Write T3 and T4.** Each as a numbered result with its proof or its bad case, in the report's LaTeX.

Produces: the T3 and T4 sections, the tricky layout family and `tipping.py`.

Exit check: the placement limit is proved or a counterexample is on file; at least one family of route-search bad cases has a measured gap; the formula has a unit test.

### Phase 5 · Experiments and parallel test (weeks 4 to 5)

Goal: measure how close the unchanged planner gets to the best plan, how planning time grows, whether the formula predicts the setting chosen, and what planning the settings in parallel saves.

- [ ] **5.1 Freeze the grid.** Proposed: 6, 7, 8, 10, 12, 14, 16, 18 and 20 devices against the exact best; 24, 48 and 96 against the upper bound; fields of 200, 300 and 500 m; three layout families; loose and tight time limits; 1 MB of data per device; 30 layouts per cell. That is 4,860 exact cases and 1,620 bounded ones. Write the grid to a file before any run.
- [ ] **5.2 Define tight and loose.** Fix both time limits as a share of the time a full tour of every device takes, so they scale with the field.
- [ ] **5.3 Gap study.** For each case run the unchanged planner, and the solver on both stop lists. Gap = 1 minus the planner's served importance over the best. Record which search the planner used.
- [ ] **5.4 Bigger cases.** The same against the upper bound, for 24, 48 and 96 devices. Report these as "at most this far from the best".
- [ ] **5.5 Parts switched off.** Rerun with one change each, through the planner's own settings only. At 6 devices, `exact_max_devices = 0`: the shortcuts where the planner's own exact answer is known. At 7 to 20 devices, `exhaustive_max_stops = 0` (local search alone), `member_admission = whole`, `coverage_rank = weighted`, and each setting pinned with `fixed:<setting>`.
- [ ] **5.6 Planning time.** Median and 95th-percentile planning time by device count, for the planner and for the solver, on one idle machine.
- [ ] **5.7 Tipping point.** Over device count, field size and data size, record the setting the planner picks and the setting the formula predicts. Report how often they agree and where they differ.
- [ ] **5.8 Parallel test.** In the harness, run `search_class` for the three settings in separate processes and compare with the planner's one-after-another loop. The search is pure Python, so threads will not help. The ceiling is 3 times, set by the slowest setting. Report the speedup by device count, including where start-up cost cancels it.
- [ ] **5.9 Statistics.** Mean gap with a 95% confidence interval per cell (`bootstrap_ci` in `experiments/analysis/stats.py`), and a table of the ten worst cases.
- [ ] **5.10 Charts.** Gap by device count, planning time by device count, the map of which setting wins, and the worst cases, in the paper's figure style (`experiments/analysis/figstyle.py`).
- [ ] **5.11 Freeze the dataset.** Save layouts, best plans, the planner's plans and timings under `results/cs8050/`, with seeds and the commit hash.

Produces: the result tables, four charts and the dataset.

Exit check: one command reruns the whole grid, and every chart is rebuilt from the saved tables alone.

### Phase 6 · Report and presentation (week 6)

Goal: hand in the report, the presentation and a repo state that reproduces every number.

- [ ] **6.1 Assemble the report.** Problem statement, T1 to T4, experiments, limits, and sources [1] to [14] from the proposal.
- [ ] **6.2 Limits section.** State what the simplified problem leaves out (step 1.8) and what the step 3.8 comparison showed, so no proof is read as covering the full planner.
- [ ] **6.3 Trace every claim.** Each number in the report points to a saved table, and each "we prove" points to a numbered result.
- [ ] **6.4 Proof pass.** Reread each proof against the Phase 1 statement as fixed, and rerun steps 3.4, 3.7 and 3.8.
- [ ] **6.5 Clean rerun.** From a fresh clone, run the tests and the one-command rerun, and compare with the frozen tables.
- [ ] **6.6 Slides.** Build the final presentation from the proposal deck: one slide per question with its answer, then the four charts.
- [ ] **6.7 Hand in.** The report, the slides and the commit hash of the dataset.

Produces: the final report, the presentation and the tagged commit.

Exit check: a fresh clone reproduces the tables, and each of Q1 to Q3 has a stated answer.

## Deliverables

The proposal promises six things; each has a home in the repo and a test for being finished. File and folder names are proposals until the first commit.

| Deliverable | Built in | Where it lands | Done when |
| --- | --- | --- | --- |
| Proofs T1 to T4 | Phases 2 and 4 | The report's LaTeX, proposed `DeveloperDocs/cs8050/` | Each is a numbered result with a proof or a documented bad case |
| Layout generator | Phase 3, tricky layouts in Phase 4 | `experiments/cs8050/layouts.py` | The same seed gives the same layout, for all three families |
| Exact solver | Phase 3 | `exact.py` and `bounds.py` | It equals enumeration up to 7 devices and the planner's `exact` search with the extra rules off |
| Test harness | Phase 3, used in Phase 5 | `harness.py` | It calls `plan_search` unchanged and one command reruns the grid |
| Charts | Phase 5 | `results/cs8050/figures/` | All four are rebuilt from the saved tables alone |
| Dataset and report | Phases 5 and 6 | `results/cs8050/` and the report PDF | A fresh clone reproduces the tables |

The course's due dates for the report and the presentation are not in the proposal; they are the first open decision below.

## Risks and open decisions

The proposal names two risks; each now has a step that detects it and a fallback that keeps the project deliverable.

| Risk | Sign it is happening | Fallback | Detected at |
| --- | --- | --- | --- |
| 1. The exact solver is too slow | The largest size it finishes is under 15 devices | Compare against the upper bound instead, and keep the focus on 7 to 20 devices, where the planner stops checking every option | Step 3.9 |
| 2. The simplified problem differs from the code | The planner's `exact` search and the solver disagree with the extra rules off, and a difference has no explanation | Limit each claim to the rules that match. Where a proof does not carry over, show concrete bad cases instead | Step 3.8 |
| A proof does not work out | No limit on the route-search gap by the end of week 4 | Report the worst gap found and the layouts that cause it | Step 4.7 |

Open decisions, to settle before the step that needs them:

- [ ] **Calendar dates.** The proposal gives six weeks but no start date and no due dates. Needed: the day week 1 starts, and the course's due dates for the report and the presentation.
- [ ] **Proofs or experiments.** The proposal asks which should carry more weight. If time runs short, which gives way first: the route-search limit (step 4.7) or the bigger cases (step 5.4)?
- [x] **Where stops may be (step 1.5).** Settled 16 Nov: a finite site list in the input, and experiments on two lists (the planner's stops, and the full list). The planner's `exact` search is exact only over the stops it is offered, so this choice sets what "the best plan" means.
- [x] **How travel times are given (step 1.6).** Settled 16 Nov: rounded in the proofs, exact in the experiments. This decides whether "in NP" is claimed or only "NP-hard".
- [ ] **The run grid (step 5.1).** The device counts, 30 layouts per cell and the 60 s solver limit are proposals, not from the deck.
- [ ] **Overlap with the paper.** The NOMS submission is due 30 November 2026. Decide which project weeks give time to it.
- [ ] **Folder names.** `experiments/cs8050/` and `DeveloperDocs/cs8050/` are proposals.

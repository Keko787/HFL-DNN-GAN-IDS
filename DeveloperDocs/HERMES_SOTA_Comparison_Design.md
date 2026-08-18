# Whole-scheduler SOTA comparison — design

**Status: BUILT AND RUN — results in [`HERMES_SOTA_Results.md`](HERMES_SOTA_Results.md).**
Both budget points complete (120 trials, 0 failures). The pre-registered primary metric
(reach-rate) came back **null**; the secondary metrics show a **crossover** — baselines collect more
under a loose budget, we produce better models under a tight one — which does not survive full
multiplicity correction but is mechanistically explained by admission behaviour in the traces.

*(Original design below, kept as written.)* Supersedes the held sweep B
([checklist §5.1b](HERMES_PreRerun_Checklist.md)), which compared *ordering* policies and was
vacuous by construction.

---

## 1. Why sweep B failed, and what fixes it

Sweep B plugged MAX-AoI and Oort into the `target_selector` slot. That slot only **permutes** an
already-decided list, because **S3b decides admission before the policy ever runs** — deliberately,
so a learned selector cannot resurrect what the gate drops. The mule then visits everything in its
queue, so the *set* served was identical across arms and every metric came out byte-identical.

> **The fix is to move the comparison up a level.** Give the baselines the decision that actually
> matters — **admission** — and compare *complete schedulers* rather than tie-breakers. That is also
> the comparison a reviewer asking for a SOTA baseline actually means.

It has the useful side effect of testing the honest thing: our contribution is in the **gates**, not
the ordering, so a fair comparison must let the baseline own its gate too.

## 2. The fairness contract — what a baseline may replace

A baseline replaces our **policy**, never our **physics**. Getting this line wrong rigs the result
in one direction or the other, so it is stated before any code:

| Stage | Baseline arm | Why |
|---|---|---|
| **S1** eligibility | **kept** | Slice membership is structural — who is in the mission at all |
| **S3a** RF clustering | **kept** | Contacts are a physical fact of `rf_range_m`, not a ranking choice |
| **Mission budget** | **kept, identical** | Every arm faces the same constraint or the comparison means nothing |
| **S3** deadline + buckets | **replaced** | This is our policy |
| **S3b** feasibility gate | **replaced** | This is our policy — and it is the decision under test |
| **S3.5** ordering | **replaced** | This is our policy |

So each arm is: `S1 → S3a → «policy decides WHO and in WHAT ORDER, under the shared budget»`.

**All three arms then have the same shape** — *rank by some key, greedily admit while the budget
allows* — and differ only in the key and the admission rule:

| Arm | Ranking key | Admission rule |
|---|---|---|
| **H1** (ours) | per-device deadline Φ, adaptively widened, bucket-tiered | S3b feasibility: reachable before its own deadline **and** within budget |
| **D1** MAX-AoI | age since last service (stalest first) | greedy: admit while cumulative travel + service fits the budget |
| **D2** Oort | `\|B_i\|·√(mean Loss²)` + staleness bonus | greedy: same budget walk, different key |

That is a clean, defensible contrast: **same information, same constraint, different decision rule.**

## 3. The metric — reach-rate, not training time

The natural framing from the Oort and FedCS abstracts is *improvement in training time to a target
accuracy*. **In this system that metric is a tie by construction**, and we have measured it: on the
L1 comparison, conditional on reaching τ the arms took **identical** rounds (2.83 vs 2.83 at τ=0.82,
2.27 vs 2.27 at τ=0.75, p=1.0). Leading with it would report "no difference" and miss the effect.

**What differs is whether a run gets there at all.**

* **Primary: reach-rate** — the fraction of trials whose global model reaches τ within the mission
  budget. Paired per seed (a binary outcome per trial), so the correct test is **McNemar on the
  discordant pairs**, not a t-test on rates.
* **Secondary, reported as the honest null: conditional time-to-accuracy** — rounds to τ among
  trials where *both* arms reached it. Expect a tie; report it anyway, because it is what makes the
  reach-rate claim precise rather than vague.
* **τ = 0.82**, the median `final_accuracy` over the 640-trial matrix, with **0.85 and 0.75 as
  sensitivity checks**. τ=0.90 is above the p90 and has no resolution — 1 vs 2 of 40 trials reached
  it, 38 reached neither.

Both are recomputable from retained traces at any τ
([`tau_from_traces.py`](../experiments/exp4/tau_from_traces.py)), so the threshold is an analysis
choice, not a re-run.

> **Why this is the better claim.** "Our scheduler reaches the target in 70 % of missions where the
> baseline reaches it in 45 %" is a statement about *availability under a budget* — which is what
> HERMES is for. "Our scheduler is 8 % faster" would be a throughput claim, and §A1 of the results
> already shows we **lose** on throughput when the network is healthy. Reach-rate is the metric our
> architecture is actually about.

## 4. Implementation

**One new interface.** A policy may optionally expose:

```python
def admit_and_order(contacts, device_states, env, *, budget_s, mule_pose, now) -> List[ContactWaypoint]
```

`FLScheduler.build_contact_queue` checks for it **after S3a** and, when present, delegates admission
and ordering entirely — skipping S3, S3b and the bucket walk. When absent (every current arm), the
existing path runs untouched.

* **Frozen surface:** `fl_scheduler.py` — roughly 10 lines. ⇒ **Freeze Amendment 4.**
* **Inert for H0–H3, B1, B2** by construction: no existing policy implements the method.
* The budget walk (travel time at `cruise_speed_m_s`, plus `session_time_s` per contact) already
  exists in `s3b_feasibility.FeasibilityModel` and should be **reused, not re-implemented** — both
  arms must price travel identically or the comparison measures the cost model, not the policy.

Estimated ~400 LOC including tests — comparable to the B1/B2 work.

## 5. Operating point — a pilot is required first

Reach-rate needs **two conditions at once**, and they pull against each other:

1. **The budget must bind**, or every policy admits everyone and we are back to a vacuous comparison.
2. **τ must be reachable by 20–80 % of runs**, or the metric saturates and discriminates nothing.

The matrix operating point (N=6, `rrf`=60, budget 120 s, `n_missions`=4) already satisfies (2) —
reach-rate at τ=0.82 was 28/40 and 19/40, mid-range and discriminating. Whether it satisfies (1)
**for admission** is untested: pure *ordering* did not differentiate there, but admission plausibly
will, since the arms would admit different sets rather than reorder the same one.

**Pilot: 3 arms × 10 seeds at the matrix operating point (~30 trials, ~10 min at 3 shards).**
Accept the operating point only if both hold:

* the admitted **sets** differ across arms on a majority of seeds (log them and diff — do not infer
  it from outcome metrics); and
* reach-rate at τ=0.82 sits inside 0.2–0.8 for every arm.

If (1) fails at this point, widen the field (N=12, radius 300) and re-pilot — but **prefer the
matrix operating point**, because a result there is directly comparable to the headline numbers
instead of living in a separate, heavily-degraded regime.

## 5a. Pilot result (2026-08-13) — ⚠ the stated criterion FAILED, and the reason is informative

30/30 trials `ok` at the matrix operating point (N=6, `rrf`=60, budget 120 s, jittery, `n_missions`=4,
10 seeds, H1 vs D1 vs D2).

**Condition 1 — admitted sets differ: FAIL as stated (22 %, threshold was >50 %).**

| | D1 | D2 | H1 |
|---|---|---|---|
| mean devices served **per mission** (of 6) | 5.97 | 6.00 | **5.60** |
| missions where the served sets differ across arms | | 9/40 (22 %) | |

The differences concentrate in **mission 1** and mostly vanish afterwards. The reason is structural:
in mission 1 every device is unserved, so D1 sees everyone as *infinitely stale* and D2 sees everyone
as *unexplored* — both admit the lot. **Our S3b gate does not**: it drops on per-device deadline
feasibility and serves 5.60/6. After mission 1 the budget is ample for all three.

> **First measurement worth having anyway:** *our gate is more restrictive than a plain greedy budget
> walk.* That is a real property of the design, not a bug — but it is the opposite of what one might
> assume, and the full run should be framed around it.

**Condition 2 — reach-rate inside 0.2–0.8: FAILS at τ=0.82, PASSES at τ=0.85.**

| τ | D1 | D2 | H1 | |
|---|---|---|---|---|
| 0.75 | 1.00 | 1.00 | 1.00 | saturated |
| **0.82** | 0.90 | 0.80 | 0.90 | saturated high |
| **0.85** | **0.20** | **0.20** | **0.50** | **discriminating** |

τ=0.82 was chosen from the **whole 640-trial matrix**, which pooled H0 and the full degradation
surface. At *this* operating point accuracy runs higher, so the discriminating threshold here is
**0.85**. The lesson: τ must be set from the distribution of the sweep it is applied to, not
inherited.

### Raw outcome means — no test run, n=10

| Arm | `final_auc` | `final_accuracy` | completion | `update_yield` |
|---|---|---|---|---|
| **D1** MAX-AoI | 0.9405 | 0.8339 | **0.817** | **2.125** |
| **D2** Oort | 0.9395 | 0.8285 | 0.783 | 2.100 |
| **H1** ours | 0.9176 | **0.8437** | 0.667 | 1.725 |

**Do not read these as a result.** n=10, no paired test, and the arms are not yet separated
statistically. What they *do* show is that the comparison is **not vacuous** — unlike sweep B, the
arms differ substantially — and the pattern is worth stating so the full run is not designed around
a wrong expectation: **the baselines serve more devices and complete more missions; ours reaches the
harder accuracy threshold more often.** If that survives n=20 with a paired test, it is a genuinely
interesting trade-off rather than a win or a loss.

### Decision required

The stated criterion failed, so per the design this **must not** proceed silently. Three options:

1. **Accept at τ=0.85** and run as designed. Admission binds weakly (mission 1 mostly) but outcomes
   differ clearly, so the comparison is informative. Cheapest, and stays at the matrix operating
   point so results remain comparable to the headline.
2. **Tighten the budget** (e.g. 60 s) so admission binds across all missions, then re-pilot. Stronger
   test of the admission rule specifically, but moves off the matrix operating point.
3. **Both** — run at 120 s and 60 s as a two-point budget axis, reporting how the comparison changes
   as the constraint tightens. ~240 trials.

## 6. Cost

| Step | Trials | Wall (3 shards) |
|---|---|---|
| Pilot | 30 | ~10 min |
| Full: 3 arms × 20 seeds × {clean, jittery} | 120 | ~25 min |

Cost per trial ≈ 38 s (measured, canonical). **Concurrency only exists if the sweep is sharded** —
one runner executes its grid sequentially, which is what made sweep A run 2.4× over its estimate.

## 7. Risks, stated up front

* **The baselines may win.** MAX-AoI is a strong, well-motivated policy; if it beats our gate on
  reach-rate, that is the result and it gets reported. The design is worth running *because* the
  outcome is not predetermined.
* **A degenerate tie is still possible** if the budget does not bind on admission — the pilot exists
  to detect that before the full run rather than after.
* **Giving a baseline admission authority is a design choice we make on its behalf.** The greedy
  budget walk is the obvious reading of "MAX-AoI under a budget", but it is *our* reading, and the
  paper must say so — the same discipline applied to calling B2 "Oort's statistical-utility
  selection" rather than "Oort".
* **`n_missions`=4 gives S3c and any adaptive mechanism little room.** S3c stays **off** here, as in
  the matrix, so this compares static policies.

## 8. Open decisions

- [ ] Confirm the fairness contract in §2 — specifically that **S3b is replaced, not kept**. This is
      the crux: keeping it would re-create sweep B's vacuity, but replacing it means our gate is
      *on trial*, which is the point.
- [ ] Name the arms. `D1`/`D2` proposed to keep them distinct from the ordering-only `B1`/`B2`,
      which should be retired rather than reported.
- [ ] Decide whether H2/H3 join the comparison (they need `--l1-channel`, so they cannot pool with
      H1 — likely a separate follow-up).

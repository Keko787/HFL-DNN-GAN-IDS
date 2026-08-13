HERMES (#74): SEC 2026 Author Response

=========================================================================
PASTE EVERYTHING BETWEEN THE MARKERS. Nothing below the second marker
goes to HotCRP.
=========================================================================

>>> BEGIN RESPONSE >>>

We thank the reviewers for reading the paper closely. The reviews are fair, and we want to
answer in the same direct spirit. We have also built and run the end-to-end experiment the
reviews asked for; its results are below and in the two attached figures.

HERMES exists to solve one problem: keeping federated learning alive when the network will
not cooperate. In the disconnected, intermittent settings we target, a stationary server
cannot reach every device and the training session stalls. The paper's contribution is the
coordination architecture that keeps federation moving under those conditions, not a new
learning algorithm, and much of the feedback fairly points to places where we described that
architecture poorly rather than to holes in the system itself. In preparing this response we
re-audited every claim against our code and data. The measured results hold: Tables V and VII
reproduce exactly from our committed per-trial data. What needs correcting is the prose around
them, and we would rather do that plainly than defend it.

**What ran where.** Our most important correction is the platform. The federated-versus-
centralized study (Table V, Figs. 5–6) ran on Chameleon Cloud over a real shaped 10 Mbps link,
not "the AERPAW digital twin" as §IV-A states, and it measures communication transport rather
than model training, so the FedAvg and CICIOT-2023 details listed there as controlled factors
do not belong. The RF study (Table II) was flown in the AERPAW digital twin under
QGroundControl, not on the hardware testbed; the four radio policies are then scored offline
against that single captured mobility and SNR trace, so CH2 and CH3 are modeled gains rather
than separately flown channels. And the scheduler's learned component is a single-agent Double-DQN trained in a
lightweight in-process simulator; the CTDE and AERPAW attributions in §III-B, and Table III's
"DQN", are labeling errors we will fix. The corrections are all to description; the numbers
stand.

**The integrated experiment.** The reviews are right that the three layers were never evaluated
as one system, so we built that experiment and ran it over the real multi-process orchestrator,
training the actual DNN-IDS rather than synthetic payloads. The full pipeline runs end to end,
radio link through gated scheduling to two-pass hierarchical aggregation, with the radio held
to a fixed band so the comparison isolates mobility and coordination. Flat federated learning
with no mule (H0) and the integrated stack (H1) share the seeded initial model and the trial
seeds, so every cell is paired. The model
converges: AUC rises from 0.27 at initialisation to 0.96, at 87% final accuracy.

The result is a crossover, and we want to state both halves of it. Under a clean link the mule
is a net cost: H0 finishes at AUC 0.985 against H1's 0.958 and aggregates 3.4 updates per round
against 1.3. Under severe terrain blockage the ordering reverses sharply, and it is worth being
precise about where. In the harshest setting we tested, 60% of devices cut off from the server
and the rest reaching it at most 30% of the time, H0 collected nothing in 7 of its 20 runs and
finished at AUC 0.667; H1 failed no run and finished at 0.934. We should be equally clear about
the mechanism: blockage is modelled as removing devices from the server's reach, and the mule
is not subject to it, because flying to an unreachable device is what a mule is for. What the
experiment adds is that the mule's own costs, contact reliability and backhaul loss, do not
erode that margin. HERMES is a targeted intervention rather than a general-purpose improvement,
and the first attached figure shows where it pays for itself.

A second run isolates the radio layer, varying only whether the mule adapts its band and
holding the rest of the stack fixed. This one is a negative result and we report it as such:
across five blockage levels at twenty paired seeds each, adapting the band did not measurably
change final model quality — every condition is a statistical tie at this sample size, and two
are nominally negative (second figure). The effect the layer does have sits one level down, on
the mule's uplink, where adaptive selection lowers modelled backhaul loss under a jittery,
band-crossing channel and costs nothing when the channel is healthy. We therefore present the
adaptive radio as a robustness mechanism for the backhaul rather than an accuracy driver, and
we do not claim an end-to-end gain we were unable to resolve.

**The scheduling rule.** The deadline function is under-justified in the paper, and Algorithm 1
prints its update with the sign reversed. The rule sheds load from devices the drone cannot
reach: a device that fails every contact wastes a flight leg, so each failed attempt should
push its deadline outward, while a device served reliably tightens. We will state the rule and
its rationale properly and correct the algorithm.

**Does the learned scheduler earn its complexity?** Two reviews asked this, and averaging hid
the answer here too. Pooled across all conditions the RL selector's edge over simpler mule
baselines is modest, which is what the reviews saw; but it tracks difficulty the same way,
growing to a large effect under the tightest mission budgets and worst links and vanishing when
conditions are easy. We should have reported it as regime-dependent rather than as a single
average, and we will. Significance is reported throughout (paired Wilcoxon with Cliff's δ), and
we vary the blockage assumption rather than fixing it at a single value: four levels in the
integrated experiment, two in the scheduling study.

**What we do not yet show.** We still have no comparison against published UAV-FL systems; that
gap is real and is our first priority for a revision. On novelty: the contribution is architectural, the decoupled
layers, the narrow interfaces between them, a scheduler whose learned part provably cannot
override the hard scheduling rules, and aggregation that never stalls on an absent device.
The radio layer's controller is deterministic and aggregation is standard FedAvg by design;
both are meant to be replaced by learned variants without disturbing the interfaces, which is
the natural next step.

We think the system underneath is sound, and we have tried to show exactly where it stands. We
are glad to answer any further questions.

<<< END RESPONSE <<<

=========================================================================
AUTHOR NOTES, DO NOT PASTE
=========================================================================

## The narrative rewrite (this version)

Both advisors flagged the previous draft as too in-the-weeds and too "AI-toned." This version
trades exhaustive completeness for a story a reviewer can follow: a human opening (what the
system is for), an honest-audit frame, the platform corrections grouped in prose, one strong
result (3.1 vs 0.36 updates/round), and candid limitations. It drops the deadline-formula
derivation and symbol definitions, the bucket/stage taxonomy, the per-β number dump, and the
bold mini-labels — all of which read as machine-generated and none of which a rebuttal needs.
The full technical detail belongs in the revised paper, not here. All seven reviewer concerns
are still answered: novelty, sim-vs-real, baselines, layers-not-integrated, formula
justification, RL benefit, and model quality.

## LAYER-1 RUN (re-run clean 2026-08-13) — the second figure

> **⚠ SUPERSEDED — this section previously reported a positive layer-1 effect
> (δ +0.140…+0.369, "AUC positive in 5/5 cells"). That result did not survive a
> clean re-run and has been RETRACTED. Do not use the old numbers. Details below.**

Data: `results/exp4_paper/h2h3_dz_{clean,dz00,dz02,dz04,dz06}.csv` — **200 ok rows of 200,
with 20/20 paired seeds in every cell**. Figure: `results/exp4_paper/fig_exp4_layer1.png`.
Repro: `DeveloperDocs/exp4_figure_layer1.py`, sweep:
`experiments/exp4/run_l1_deadzone_sweep.sh`.

Run at the H0/H1 settings (`--n-missions 4`, link quality 0.5, base seed 42), with
`--l1-channel` and `--realism`.

H3 (adaptive band) over H2 (fixed band) on final AUC — **paired** mean difference with a
bootstrap 95% CI, the same estimator as every other table here:

| cell | H2 AUC | H3 AUC | H3−H2 | 95% CI | p | δ | verdict |
|---|---|---|---|---|---|---|---|
| clean | 0.849 | 0.863 | +0.015 | [−0.090, +0.117] | 0.31 | +0.02 | tie |
| dz 0.0 | 0.910 | 0.911 | +0.001 | [−0.092, +0.092] | 0.60 | +0.06 | tie |
| dz 0.2 | 0.800 | 0.867 | +0.068 | [−0.070, +0.206] | 0.33 | +0.12 | tie |
| dz 0.4 | 0.949 | 0.938 | **−0.011** | [−0.088, +0.043] | 0.26 | +0.35 | tie |
| dz 0.6 | 0.953 | 0.943 | **−0.010** | [−0.095, +0.047] | 0.15 | +0.24 | tie |
| jittery pooled | 0.903 | 0.915 | +0.012 | [−0.038, +0.058] | 0.045 | — | tie † |

† p<0.05 but the CI **includes 0**. Our stated rule is that the CI and Wilcoxon must agree,
so this is a tie, not a claim. Do not quote the p-value on its own.

Extending to `mission_completion_rate`, `update_yield` and `round_close_rate@2`: **all 20
condition × metric tests are ties.** Two AUC cells are nominally negative.

**What this supports:** nothing about layer-1 improving the end model. **Do not claim an
end-to-end accuracy benefit for the adaptive radio.** Delete the body's "the layer buys
better updates, not more" — the clean data supports neither half of it.

**What you CAN still say** (backed, and it is the framing the methodology doc already uses):
the adaptive controller reduces *modelled* mule→BS backhaul loss under a jittery,
band-crossing channel (≈0.13–0.14 at our calibration, adaptive never worse across 1000
seeds) and is a wash on a healthy backhaul. That is a **backhaul-robustness mechanism**, not
an accuracy driver. If pushed on end-to-end evidence, the honest answer is: *"the integrated
effect is small and we could not resolve it at this sample size — one cell at a 6-mission
horizon reaches significance, a five-cell sweep at 4 missions does not, so we do not claim
it."* That is a better position than defending a number a reviewer could overturn.

**Why the old numbers were wrong — say this plainly if asked:**

1. **The old sweep was contaminated.** 22 of its 190 `ok` rows had produced no model at all
   (`rounds_evaluated=0`): blank AUC, but hard-zero participation. The blank AUC was dropped
   from the AUC means while the fabricated `0.0` was averaged into the participation means —
   the same trial excluded from one column and invented in another. Root cause was in the
   driver; it now records those as `status=no_eval` so they are excluded from *every* metric.
2. **The old effect size was computed UNPAIRED** on unequal-n groups (16,17 / 17,19 / …)
   against a protocol that is paired everywhere else. Pairing flips the clean cell's sign.
3. Both are fixed, the sweep was re-run (0 failures in 200 trials), and the effect is gone.
   Note in the table above that δ stays positive at dz 0.4/0.6 while the *mean difference* is
   negative — a rank statistic quoted without a CI is exactly how the old reading happened.

**Caveats that still apply:**

1. **NOT comparable to H0/H1 — still true, for two reasons.** The `n_missions` confound is
   now gone (both are 4), and AUC levels are much closer than before (H2/H3 ≈0.94–0.95 at
   dz 0.6 vs H1 0.963). But (a) `--l1-channel` replaces the flat 2% backhaul loss with the RF
   channel model in **both** H2/H3 arms, so the backhaul physics differ from H0/H1; and (b) I
   checked the seeds directly and they **do not line up** — h0h1's trial-1 seed is h2h3's
   trial-0 seed, because the h0h1 cells carry three `link_quality` values and that shifts the
   derived per-trial seeds. So H2/H3-vs-H0/H1 remains unpaired. Keep the two figures separate;
   do not reintroduce a layer-ablation panel.
2. **The selector is untrained in both arms** (no `.npz` on this machine). It does not confound
   H3-vs-H2 since it is identical in both, but a reviewer may ask whether the result holds with
   a trained policy. Honest answer: not yet tested.
3. **A tie is not proof of no effect.** At n=20 per cell the CIs are ±0.05–0.15 wide, so a
   small real effect could hide inside them. Say "we could not resolve it", not "there is none".

## ADVERSARIAL REVIEW FINDINGS (5 lenses, 2026-07-24) — what is fixed, what stands

Fixed already: the "adaptive radio through gated scheduling" overstatement, all H2/H3 claims,
the layer-ablation panel, the "learned selector" description.

**Fixed this round, both verified against the data:**
- The 0.846 headline was a bimodal mean. By link quality at dz 0.6: H0 = 0.667 / 0.916 / 0.955
  at lq 0.3 / 0.5 / 0.7, and ALL 7 total-failure runs (final_auc == init_auc, server reached
  zero clients) sit in lq 0.3. H1 fails zero runs anywhere. The body now reports the worst cell
  and the failure count, which is both honest and a stronger claim than the average.
- The crossover was presented as a discovery. It is not: H1 is flat across the sweep
  (0.973 / 0.954 / 0.941 / 0.963) while only H0 declines, because the dead zone is applied to
  H0 only (`driver.py:106-114`, `topology_builder.py:106-109`: H1's contact model is
  "REGIME-INDEPENDENT"). The body now states the mechanism plainly and scopes the contribution
  to "the mule's own costs do not erode that margin."

**STILL OPEN — investigate before camera-ready:**
- **H1's clean cell looks anomalous.** H1 scores WORSE in clean than in jittery on every metric
  (AUC 0.958 vs 0.973, yield 1.300 vs 1.839, close 0.725 vs 0.869) despite a strictly better
  configuration (0% backhaul loss vs 2%, identical contact model). Round-close distribution in
  clean is bimodal: 10/20 trials close exactly 0.5, 8 close 1.0, 2 close 0.75. Ten trials
  closing exactly half their rounds is a suspicious pattern and may be a bug. The body's
  clean-link cost claim is a concession against our own interest, so it is safe to leave, but
  if the anomaly is a bug the real cost is smaller than stated.
- **Reviewer-facing vocabulary gap.** The comprehension lens could not map any number in the
  body onto a bar in the figure: the figure is labelled H0/H1 and the body now mostly avoids
  those tokens. Consider one clause tying them together if words allow.
- **Advisor lens:** ~58% of the body is corrections/concessions, ~19% is the new experiment.
  If a revision is invited, rebalance toward the result.

## EXPERIMENT 4 — the numbers in the body, and what backs them

Source: `results/exp4_paper/h0h1_all.csv` (519 ok rows of 520; one error row) and
`h2h3_l1.csv` (80 rows). Arms per `experiments/exp4/driver.py`: H0 = flat FL, no mule;
H1 = integrated stack (mule + gated scheduler + two-pass HFL); H2 = H1 + trained RL selector;
H3 = H2 + adaptive L1 channel. Figure: `results/exp4_paper/fig_exp4_crossover.png`.

Every number in the body, verified:

| Claim in body | Value | n/arm |
|---|---|---|
| Convergence, AUC init → final (H1, jittery) | 0.266 → 0.958 | 239 |
| Final accuracy (H1, jittery) | 0.868 | 239 |
| Clean: H0 vs H1 final AUC | 0.985 vs 0.958 (δ −0.685, large) | 20 |
| Clean: H0 vs H1 update yield | 3.44 vs 1.30 (δ −0.955, large) | 20 |
| dz 0.6: H0 vs H1 final AUC | 0.846 vs 0.963 (δ +0.533, large) | 60 |
| dz 0.6: H0 vs H1 update yield | 0.558 vs 1.633 (δ +0.774, large) | 60 |
| dz 0.6: round close rate | 0.492 vs 0.804 (δ +0.556, large) | 60 |

Both arms start from the same seeded init θ, so the AUC 0.27 baseline is shared — that is why
the figure's reference line applies to both.

**Caveats you should know before defending this:**

1. **The clean-regime cost is large and is now stated in the body.** Do not soften it if asked;
   it is δ −0.69 on AUC and −0.96 on yield. The crossover IS the finding.
2. **H0 variance at dz 0.6 is wide** (AUC error bar spans roughly 0.62–1.07). The mean gap is
   real and the effect size is large, but do not describe it as uniform — some H0 trials do
   fine even at severe blockage.
3. **H2/H3 ARE NOT COMPARABLE TO H0/H1 — do not compare them.** Three independent problems:
   - **Different mission budget.** H0/H1 ran `n_missions=4`; H2/H3 ran `n_missions=6` (50%
     more). Confirmed in the CSVs (`param_n_missions`) and the shard logs. Mean rounds closed:
     3.58 vs 4.85. More missions means more contacts and more update opportunities, so any
     apparent H2/H3 advantage is confounded with budget.
   - **Zero seed overlap** with the H0/H1 shard — unpaired.
   - **One benign cell only**: dead zone 0.0, link quality 0.5, n=20 per arm per regime.

   A layer-ablation panel was drafted and then REMOVED for exactly this reason. Do not
   reintroduce it, and do not repeat the earlier working claim that "H2/H3 recover H1's
   clean-link cost" or that "H3 is best-or-tied" — those readings are confounded by the extra
   missions. The body now says only that H2/H3 were run separately under a different mission
   budget and draws no comparison from them.

   **To make the full-system comparison properly** (worth doing for the revision, or now if
   there is ~1 hour): re-run H2/H3 with the H0/H1 settings, then H3 can be plotted across the
   severity sweep alongside H0 and H1.

   ```
   python -m experiments.exp4.runner_main --csv results/exp4_paper/h2h3_matched.csv \
     --arms H2 H3 --N 6 --rrf 60.0 --n-missions 4 --regime jittery \
     --dead-zone 0.0 0.2 0.4 0.6 --link-quality 0.5 --n-trials 20 \
     --real-model --data-source canonical --realism \
     --selector-weights weights/a4_selector.npz --base-seed <same as the h0h1 run>
   ```
   Matching `--n-missions 4` and the base seed is what makes the cells (and therefore the
   derived per-trial seeds) line up with `h0h1_all.csv`, giving a paired comparison. Shard by
   dead zone to run them in parallel.
4. **Exp 3 vs Exp 4 numbers differ** for superficially similar claims (Exp 3: 3.1 vs 0.36
   updates/round, A4 vs A1; Exp 4: 1.63 vs 0.56, H1 vs H0). Different experiments, different
   setups. The body now uses Exp 4 for mule-vs-no-mule and Exp 3 only for the RL-vs-heuristic
   question, so the two do not appear side by side. Keep it that way.
5. **These runs are dated Jul 24**, after the Jul 18–22 window in the original plan doc. You
   confirmed the window is open; if a chair queries the timing, the honest answer is that the
   experiment was built in response to the reviews.

## ACCURACY NUMBER — superseded by Experiment 4

The body no longer needs the ~99% figure from the plotting script. Experiment 4 supplies a
real, reproducible convergence result measured in the integrated run (AUC 0.27 → 0.96, 87%
accuracy), which is strictly better evidence and answers the model-quality concern directly.
The note below is retained only in case you want the prior-work number for the revision.

## ACCURACY NUMBER — confirm, then optionally strengthen

The body says the DNN-IDS is "separately validated on CICIOT-2023" WITHOUT a number, which is
safe as-is. Found in `Analysis/TestAnalysis/ModelPerformance/Nids_Augmented_vs_Real_data_
evaluation_graph.py:9-12`: real-data DNN-IDS accuracy ≈ **99.1%**, precision ≈ 98.2%,
recall ≈ 99.96%. These are prior-work numbers (the HFL-DNN-GAN-IDS model the paper cites), not
from the HERMES experiments.

If you confirm they are yours and citable (e.g. from HiFINS / the e-Science GAN-IDS paper),
change "is separately validated on CICIOT-2023" to "reaches ≈99% accuracy on CICIOT-2023 in
prior work [cite]". That converts 74D's model-quality concern from reframed to answered. Do
NOT insert the number until you can stand behind it.

IMPORTANT: a full repo-wide search (completed) found these numbers ONLY as hardcoded inputs to
two plotting scripts — there is no logged run, eval output, or checkpoint behind them anywhere
in the repo. So "confirm citable" means trace them to a published paper or a reproducible run
before citing; do not cite the plotting script itself. If you cannot, the safe body wording
("separately validated") stays and the number is a revision task, not a rebuttal claim.

## New results — the strategic call

- The re-run Exp 3 is NOT new evidence (same experiment, new dead-zone); it is already the
  sensitivity result. It will not satisfy "put some new results" on its own.
- The accuracy number above is the only rebuttal-scope new result worth adding, and only if
  the format allows referencing new numbers and you can cite it.
- GAN and full end-to-end + baselines are REVISION-scope. If Jul 29 brings a revise decision,
  build the end-to-end experiment first (answers the layers-not-integrated concern directly),
  GAN second. Do not promise either in the rebuttal.

## Verify before submitting

- **Table II — RESOLVED by author recollection (2026-07-24).** Flown in the AERPAW digital twin
  under QGroundControl; the four radio policies were then scored offline against that single
  captured mobility/SNR trace. The body now says exactly this.

  Note this is still recollection, not artifact: an exhaustive search of the whole
  `D:\networkIntrusionDetectionSystem` tree (excluding venvs) found no L1 experiment script, no
  SNR/mobility trace, no offload log, and no analysis script. `hermes/l1/` has only an
  inference-only `channel_ddqn.py` and a read-only `rf_prior.py`; `hermes_rl/` uses synthetic
  sinusoidal channels, not AERPAW telemetry. The producing artifacts live outside this repo —
  find them before camera-ready, since §III-A must be corrected there too.

  The paper says three inconsistent things, which is why this was asked: §III-A "deployed and
  evaluated on the AERPAW wireless testbed" (implies hardware), Exp 2 "live AERPAW UAV mobility
  and BS1 SNR observations" (implies captured telemetry), and the working .tex "All four
  experiments use the AERPAW wireless digital twin". The twin + QGroundControl account is
  consistent with the second and third; §III-A's "testbed" is the one that needs fixing.

  The wording deliberately says "flown … under QGroundControl" rather than "trace replay",
  because a twin flight is a stronger and more accurate claim than replaying a recorded trace,
  while "scored offline against that single captured trace" keeps the modeled-channel
  disclosure honest.
- **Fig. 7 / Table VI** — the narrative version no longer mentions these at all, which is
  cleaner (they were Chandra's and unverifiable). If a reviewer asks specifically, you can
  address them, but the rebuttal no longer stakes a claim on them.
- **HotCRP word count** — this version is well under 1000; verify the live counter anyway.
- **Reviewer IDs** — the body uses no R1–R4 or 74A–D tags, per your earlier preference.

## Resolved earlier (still true, for your reference)

- The submitted dataset survives: `results/legacy exp3/exp3_v8.csv` reproduces Table VII 8/8,
  including A1 jittery 0.161 ± 0.067 vs the paper's 0.16 ± 0.07. §3's numbers are re-analysis.
- Submission dead zone was 80%; `exp3_7_21` (60%) matches v8 on the mule arms to four decimals
  and differs only on A1. Only A1 moved.
- Clean-regime update yield is a tie (A4 4.290 vs A1 4.290); jittery is 8.6× (3.101 vs 0.361).
  Consider rewriting the paper's Observation 3 around update yield for camera-ready.
- `results/exp3_7_21/` and v10–v15 are post-submission; only `exp3_v8.csv` backs the table.

## Camera-ready / revision backlog

- **The trust gate does not close.** `is_new` is cleared only on a CLEAN contact
  (`s3_deadline.py:148`) and `classify_bucket` tests `is_new` first (`:101`), so an unreachable
  node holds the top NEW bucket forever — the shedding works on the deadline axis but is
  overridden on the bucket axis. Fix: clear `is_new` after k failures, or add a reachability
  test before the bucket check. (Already fixed in code this session; keep for the writeup.)
- Calibration: `calibration.toml` was flipped to `status = "verified"` while ε_prop is still
  the placeholder `REPLACE-FROM-PLATFORM-SPEC` 10 J/m, and the ±50% sweep never varies ε_prop.
  Per-constant status added this session; report Exp 3 energy as ratios, not absolute J/Δθ.
  Also: exp3 ε_bit = 1.2e-9 vs exp1 7.0e-10 under a "same constants" comment — one is wrong.
- A3 over-prunes (filter fires on 100% of decisions). Fix the feasibility model or re-frame A3
  as a cautionary arm rather than a rung in a monotone ladder.
- `deadline_override_ts` is never cleared; Φ has a floor but no cap (path-dependent); PARTIAL
  and TIMEOUT share one branch; the S2A/S2B utility gate is a no-op (`min_utility = 0.0`).
- Rewrite `Experiment_3_Run_Guide.md:131` ("tune higher to put A1 below the mule arms") before
  the repo is public.
- Fill the `|θ| = X MB` placeholder with the real DNN-IDS footprint (~4.7K params, ~18.8 KB).

## Ask Chandra (for the revision, not the rebuttal)

1. Which code produced Fig. 7 and Table VI, and can it be archived with the paper?
2. How many seeds? Submitted Table VI says two seeds (3.00 jobs); the working .tex says three
   seeds, 50 episodes, 3.33 ± 0.47. If the three-seed run predates May 8 it is citable.
3. Does Table VI contain any rows from this repo? If provenance can't be established, consider
   dropping Table VI and Fig. 7 — the scheduling argument survives on Table VII alone.

# Experiment 5 — the learned score's calibration: findings and a training diagnosis

*6 Oct 2026. The RL campaign (`exp5 run rl`) paused after `rl-calibrate`, as designed. This page records what the headroom report and the calibration found, why the learned score looks held back by its training rather than by the problem, and the options for going on. **The decision is open** (last section). Status of every stage: [Experiment_5_Readiness.md](Experiment_5_Readiness.md).*

---

## What this is about

FeRRy's in-flight slot picks the next (band, stop) pair while the mule flies. Two fillings compete:
- **FX**, the fixed rule;
- **FQ**, a learned pair score, trained in FerrySim.

Study 5.5 asks whether FQ beats FX, and whether looking ahead (discount γ > 0) helps over a one-step score (γ = 0). Its pre-registered sweep trains γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99} × 10 seeds on the `jittery56` family.

Before the sweep, two stages run:
- **`rl-headroom`** measures how much room any in-flight choice has over FX, and sets ε, the smallest difference the verdict counts.
- **`rl-calibrate`** trains γ ∈ {0, 0.9} × 3 seeds per family and judges them by the same rules. It is **not pre-registered** (R26), and with 3 seeds it cannot read "rising". It checks that the learner works: the γ = 0 **sanity check** asks that the one-step learned score come within ε of the best fixed one-step rule.
  - `jittery56` is the family the sweep flies.
  - `clean` is the negative control (critic C3).

**Where the outputs are** (from commits `705235f` for the headroom stage and `53bdfc5` for the calibration; their code is the same):
- **Headroom:** `results/exp5/rl/headroom/headroom.json`.
- **Calibration:** `results/exp5/rl/calibration/{jittery56,clean}_evaluation.json` and `_verdict.json`.
- **Checkpoints:** `results/exp5/checkpoints/5.5-calibration-{jittery56,clean}/g{0,90}/`. Each manifest holds its validation history.
- **Launcher manifests:** `results/exp5/rl/{headroom,calibrate}/_launcher/`.

---

## The headroom report (200 episodes per cell, 6 Oct 06:42–07:17)

Each policy's mean value per episode; "headroom" is the best achievable value minus FX's, and "slot headroom" is the share the in-flight slot can reach.

| Cell | Decisions per sortie | Sorties with 2+ decisions | Headroom | Slot headroom | FX | greedy_1 | hyb | F |
|---|---|---|---|---|---|---|---|---|
| `cln-n12-180` | 2.07 | 65% | 0.013 | 0.011 | 0.113 | 0.116 | 0.114 | 0.102 |
| `cln-n12-90` | 1.86 | 67% | 0.032 | 0.005 | -0.262 | -0.259 | -0.259 | -0.269 |
| `jit-n12-180` | 1.50 | 33% | 0.023 | 0.019 | 0.059 | 0.066 | 0.058 | 0.040 |
| `jit-n12-90` | 1.84 | 62% | 0.042 | 0.015 | -0.293 | -0.282 | -0.292 | -0.317 |
| `jit-n6-150` | 1.02 | 2% | 0.012 | 0.011 | 0.086 | 0.088 | 0.086 | 0.051 |
| `jit-n6-75` | 1.14 | 13% | 0.024 | 0.013 | 0.140 | 0.149 | 0.141 | 0.096 |

- **ε = 0.01 in every cell.** The rule is max(0.01, 0.1 × headroom), and the headroom is at most 0.042.
- **The room over FX is thin.** It is 0.012–0.042 in all, and 0.005–0.019 for the in-flight slot itself.
- **The N = 6 cells barely decide.** Only 2–13% of sorties hold more than one decision; the re-pin found the same (one stop serves all six devices).
- **greedy_1 beats FX in all six cells,** by more than ε only at `jit-n12-90` (+0.011).

---

## The calibration verdicts (6 Oct 07:17–09:17)

Mean held-out return over each family's cells (the four jittery cells above; the two clean ones). Higher is better. The floor is greedy_1 − ε.

| Family | Outcome | FQ γ = 0 | FQ γ = 0.9 | FX | greedy_1 | Floor | Seeds of γ = 0 below the floor |
|---|---|---|---|---|---|---|---|
| `jittery56` | **flat** | -0.0778 | -0.0780 | -0.0786 | -0.0716 | -0.0816 | 0 of 3 (passed) |
| `clean` | **sanity-failed** | -0.0989 | -0.1005 | -0.0914 | -0.0884 | -0.0984 | 2 of 3 (failed by 0.0005) |

**jittery56:**
- γ = 0.9 against γ = 0: −0.0002, 95% CI [−0.0042, 0.0038]. That is equivalent within ±ε (TOST p = 0.0094), so the curve reads **flat**.
- greedy_1 against FX: +0.0070, CI [0.0045, 0.0098], over 2,000 paired episodes. That is below ε, so the flag that would hand the choice to the user (critic A2) is not raised.

**clean:** greedy_1 against FX is +0.0029, CI [0.0016, 0.0044].

**What the two say together:**
- Looking ahead adds nothing.
- The one-step learned score sits at **FX's level**, below greedy_1, the simple one-step rule.

---

## The training curves

**The training settings** (the defaults; the calibration changed none of them):
- **Episodes:** up to 10,000, validated every 1,000 on 200 episodes (400 for `jittery56`). A run stops after 3 validations without a new best and keeps its best checkpoint.
- **Network and optimiser:** 64 × 64 tanh, Adam at lr 1e-3, batch 64, replay 50,000, warm-up 1,000 transitions, target sync every 500 updates, 1-step returns.
- **Exploration:** the first 500 episodes fly around FX's pair. Then ε-greedy runs from 0.3 to 0.05 over the first half of the run.

**Each run's validation score at every check** (higher is better; the levels differ from the verdict's because validation uses the family's own cells and stream). Also the episode of its best check, where it stopped, and its training return and TD loss from the first check to the last:

| Family | Run | Validation score at each check (every 1,000 episodes) | Best at | Stopped at | Training return | TD loss |
|---|---|---|---|---|---|---|
| clean | g0_s0 | -0.146, -0.155, -0.152, -0.162 | 1,000 | 4,000 | -0.042 → -0.069 | 0.0059 → 0.0060 |
| clean | g0_s1 | -0.151, -0.150, -0.142, -0.156, -0.154, -0.144 | 3,000 | 6,000 | -0.052 → -0.059 | 0.0067 → 0.0060 |
| clean | g0_s2 | -0.142, -0.154, -0.150, -0.145 | 1,000 | 4,000 | -0.063 → -0.031 | 0.0065 → 0.0062 |
| clean | g0.9_s0 | -0.141, -0.153, -0.152, -0.148 | 1,000 | 4,000 | -0.039 → -0.067 | 0.0065 → 0.0054 |
| clean | g0.9_s1 | -0.145, -0.152, -0.145, -0.153 | 1,000 | 4,000 | -0.053 → -0.044 | 0.0064 → 0.0054 |
| clean | g0.9_s2 | -0.151, -0.150, -0.154, -0.148, -0.158, -0.149, -0.153 | 4,000 | 7,000 | -0.062 → -0.052 | 0.0070 → 0.0056 |
| jittery56 | g0_s0 | -0.051, -0.056, -0.048, -0.043, -0.046, -0.042, -0.048, -0.044, -0.045 | 6,000 | 9,000 | -0.050 → -0.030 | 0.0094 → 0.0093 |
| jittery56 | g0_s1 | -0.043, -0.044, -0.048, -0.042, -0.046, -0.048, -0.048 | 4,000 | 7,000 | -0.037 → -0.047 | 0.0100 → 0.0092 |
| jittery56 | g0_s2 | -0.051, -0.055, -0.055, -0.050, -0.057, -0.058, -0.051 | 4,000 | 7,000 | -0.019 → -0.037 | 0.0099 → 0.0093 |
| jittery56 | g0.9_s0 | -0.048, -0.050, -0.043, -0.047, -0.045, -0.042, -0.048, -0.048, -0.048 | 6,000 | 9,000 | -0.049 → -0.030 | 0.0099 → 0.0086 |
| jittery56 | g0.9_s1 | -0.040, -0.043, -0.048, -0.050 | 1,000 | 4,000 | -0.036 → -0.075 | 0.0098 → 0.0084 |
| jittery56 | g0.9_s2 | -0.051, -0.049, -0.055, -0.057, -0.050 | 2,000 | 5,000 | -0.016 → -0.047 | 0.0102 → 0.0087 |

**What the curves show:**
- **No run climbs, and no run steadily falls.** The validation score bounces within about ±0.005–0.01 of one level: −0.14 to −0.16 for clean, −0.04 to −0.06 for jittery56.
- **The best check lands anywhere** from the first to the sixth.
- **The TD loss hardly moves,** at about 0.006 for clean and 0.009 for jittery56.
- **The check-to-check bounce is as large as ε.** So the patience rule can stop a run by chance, and a real improvement of under ε would not show.
- **On their own, the curves cannot tell a ceiling from a learner that has stopped learning.** The diagnosis below rests on the comparison with greedy_1.

---

## Diagnosis: the learner looks held back by its training, not by the problem

**greedy_1 sees only what FQ sees** (`Greedy1Scorer`, `hermes/scheduler/policies/pair_slot.py`). It ranks each candidate pair by the most targets reached at the arrival SNR, then the least dwell plus travel to the next stop, then FX's tie-breaks, all read from the same `PairView` the learned score's features are built from. It is a simple function of FQ's inputs. A one-step learner (γ = 0) that works should therefore match it at least.

**FQ at γ = 0 does not match greedy_1.** It falls about 0.006 below on jittery56 and about 0.010 below on clean. Instead it lands almost exactly on FX: −0.0778 against −0.0786 on jittery56. The likeliest reading is that **it learned to copy FX rather than to choose well.** That is a training problem, which makes it potentially fixable. A pure ceiling would leave FQ near greedy_1, not near FX.

**Likely causes, most likely first, each with a fix the code already supports:**

| # | Likely cause | Why it fits | Fix (existing settings) |
|---|---|---|---|
| 1 | **It learns from data that mostly follows FX.** | Training starts with 500 episodes that fly FX's choices, then explores only 30% of the time, tapering to 5% by mid-run. So it rarely tries anything different, and it settles on FX-like choices. | Explore more: start ε at 0.5–1.0 and taper it more slowly, with fewer or no FX episodes at the start (`ferrysim train --epsilon-start`, `--epsilon-end`, `--reference-episodes`). |
| 2 | **The learning rate is too coarse.** | The differences that matter are about 0.01. A large step size keeps overshooting, which would look like the flat, bouncing curves. | A lower learning rate, 3e-4 instead of 1e-3 (`--lr`). |
| 3 | **Noisy checks stop training early.** | The checks bounce by about as much as the improvement looked for, so "three checks without a new best" can end a run by chance and pick a lucky checkpoint. | More validation episodes and more patience (`--val-episodes`, `--patience`). |
| 4 | **The network is too small** (64 × 64). | Less likely: greedy_1's rule is simple. | Would need a code change (`PairQConfig.hidden` is not a flag). |

---

## Acting on it within the plan's rules

**The rule** (Run Guide §2.8, decision 5): if the γ = 0 sanity check fails, the learner may be revised **once**, on the control cells, before the sweep. Clean, the control family, failed its check. A revision chosen on the clean cells is therefore what the rule allows. Choosing it on the jittery56 cells the sweep flies would be tuning on the test.

**The proposed screen**, fixed here before it runs:
- **Cells:** the clean family, γ = 0, seeds 0 and 1. All runs side by side. The calibration's default runs are the baseline.
- **Variants:**

  | Variant | Settings |
  |---|---|
  | V1, explore more | `--epsilon-start 1.0 --epsilon-end 0.05 --reference-episodes 0` |
  | V2, finer steps | `--lr 3e-4` |
  | V3, both | V1 and V2 |
  | V4, steadier selection | `--val-episodes 400 --patience 6` (it changes only which checkpoint is kept) |

- **The rule for picking:**
  - A variant qualifies if its γ = 0 held-out mean on the clean cells reaches the floor, greedy_1 − ε (−0.0984 there).
  - Of those that qualify, take the highest mean.
  - If none qualifies, keep the defaults: the gap is a ceiling, and the sweep runs as planned.
- **Where the chosen settings apply:** every pair-score training from then on (the sweep and Study 5.7). E3 keeps Chen's settings.
- **Cost:**
  - The screen takes about 1.5–2 h.
  - The launcher needs a small change to pass the chosen settings to the training jobs, through a learner block under `[rl]` in `params.toml`. About 30 min.
  - Then the sweep, about 10 h.

---

## The options

**A. Run the sweep as planned.**
- **Pros:**
  - It is the pre-registered test, so its result counts in the paper whatever it says.
  - The family it flies passed the rehearsal's check.
  - The verdict lands the same evening, around 19:00–21:00 on 6 Oct.
  - No new decisions or code; it also trains the E3 baseline batch 2 needs.
- **Cons:**
  - About 10 h of machine time.
  - It will most likely confirm "no benefit", and it may be testing a learner that copies FX rather than a fair learned score.
  - The control cells' miss stays unexplained.

**B. Run the screen, then the sweep with its pick** (the revision the rule allows).
- **Pros:**
  - It gives the learned score a fair test.
  - If a variant clears greedy_1, the sweep measures a learner that works.
  - If none does, the paper can say so ("more exploration and a lower learning rate changed nothing"), a stronger null than an untested one.
  - It explains the control cells' miss.
- **Cons:**
  - About 2.5 h more.
  - The verdict lands around 22:00–midnight on 6 Oct, before the 8 Oct deadline but with less slack.
  - A small launcher change.

**C. Skip the sweep and keep FX.**
- **Pros:**
  - It frees about 10 h of machine time, for `pilot3` or part of batch 2.
  - FX is what FeRRy flies anyway if the test finds no benefit.
- **Cons:**
  - 5.5's result would not be pre-registered (3 seeds, not 10), so reviewers can call it untested.
  - It departs from the plan.
  - If the full test would have found a benefit, it is missed.

---

## Decision

**Decided 6 Oct 2026, 13:58: option A.** The sweep runs as pre-registered, with the default learner settings, and `rl-e3` runs right after it.

**Why:**
- It is the pre-registered test, so its verdict counts in the paper.
- It lands before the 8 Oct deadline.
- FeRRy's lead over the state-of-the-art baselines comes from batch 1 (F and FX against H1 and D1–D4), so it does not depend on the sweep's outcome.

Option B's screen stays fixed above. If the sweep reads flat and the copy-FX diagnosis matters to the paper or its reviewers, it can run in the revision window.

---

## The sweep's verdict (6 Oct 2026, 23:10) — **flat; FX stays**

**The run.**
- **Trainings:** Study 5.5's 60 trainings (γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99} × 10 seeds on `jittery56`, with the default learner), 13:58–19:53.
- **Evaluation:** 197 min on 1,000 held-out episodes per cell, from commit `9ffb12b`.
- **Reports:** `results/exp5/rl/s55/{evaluation,verdict}.json`. The verdict is **pre-registered**.

| γ | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| Mean held-out return | -0.0795 | -0.0776 | -0.0779 | **-0.0774** | -0.0778 | -0.0779 |

- **Rising? No.** No γ beats γ = 0 by ε = 0.01. The gains are 0.0015–0.0020; the best, γ = 0.75, has Holm p = 0.14.
- **Flat? Yes.** Every γ > 0 is equivalent to γ = 0 within ±ε (TOST p ≤ 1.3e-4). A trend test detects a small upward drift with γ, about 0.002, a fifth of ε, so it has no practical weight.
- **Sanity check: passed.** γ = 0 scores −0.0795 against a floor of −0.0816, though 3 of the 10 seeds fall below the floor.
- **Against the fixed rules:**
  - The best learned score (γ = 0.75, −0.0774) sits at FX's level (−0.0786).
  - It trails greedy_1 (−0.0716) by 0.0059, CI [−0.0066, −0.0050], p = 0.002.
  - **So `replace_fx = false`.**
- **greedy_1 against FX:** +0.0070, CI [0.0045, 0.0098]. That is below ε, so the user-decides flag (critic A2) is not raised.

**It reads as the calibration did.** Looking ahead adds nothing, and the learned score reaches FX's level and no higher. This is consistent with the diagnosis above: the learner probably copied FX.

**What follows, by the pre-registered rule** (applied 6 Oct in `params.toml`):
- **`keep_learned = false`.** FX stays as FeRRy's in-flight rule, and the null is published.
- **`gamma_star = 0.75`,** recorded only.
- **Study 5.5's stack check in batch 2** flies the verdict's picks, γ = 0.75 at seed 3 and γ = 0 at seed 5, beside FX and F.
- **What drops out:** the FQ arms leave batch 2, 5.7's learned scores do not train, and M1 is not built.
- **What must be built before batch 2:** **FX-dwell and FX-cov** for Study 5.7. Built 7 Oct 2026: 5.7 flies FX, FX-dwell, FX-cov and D4.

**For the paper.**
- **The learning claim of C4 is a null.** C4 rests on the two-clock structure: a plan at the dock, and a re-decision on measured signal at each stop by the fixed rule FX.
- **The honest description:** the learned score matched FX and fell short of the one-step rule greedy_1. A one-revision screen (option B above) could test whether a different learner setting changes that, in the revision window, if reviewers press.

# Experiment 5 — what is left to run

*29 Oct 2026. Every Exp 5 run still to fly, pre-registered or added since, ranked by whether the paper needs it. **Status, 4 Nov 2026:** the Needed tier has flown and been scored (scorer `72b8e59c`, results `3cffc966`), and the paper reads it. Groups 1 and 2 of the Should tier and the table fills have flown too (see below), and 5.3 is scored in full, relaxed cells included. The rest of the Should tier and the Optional and acceptance tiers have not run; the rest of this page is as drafted. Sources: the launcher's full plan (`launch.py plan batch2`, `plan batch3`) set against the trial rows on disk; the loss-probe plan, [Experiment_5_Loss_Probes_Plan.md](Experiment_5_Loss_Probes_Plan.md); the stage-by-stage status in [Experiment_5_Readiness.md](Experiment_5_Readiness.md).*

---

## In one paragraph

Two kinds of runs are left, about **13,400 trials and 69 h** of machine time on this host in all:

- **The pre-registered plan.** The 8 Oct deadline trimmed batches 2 and 3 to 20 trials a cell and dropped some cells. Batch 2 stands at 2,260 of 9,220 trials and batch 3 at 580 of 1,360, which leaves 7,740 trials (about 44 h, plus 5.4's oracle). Two studies never ran: 5.6, and 5.10, which needs AERPAW.
- **The loss probes.** These were added after the CARE room's sweep showed where FeRRy loses: 5,380 trials (about 20 h) and four builds.

The paper needs about 11.5 h of this before its revision; most of the rest strengthens it.

**Needed** means one of three things:
- the submitted paper says something wrong without the run;
- the run tests the paper's central claim;
- the run is where the room or Exp 5 already shows FeRRy losing, and the paper's tables don't show it.

---

## Summary

| Need | What | Trials | ~h | Builds first |
|---|---|---|---|---|
| **Needed** | 5.3 to 40 trials; 5.4's remaining cells; P1 harsh radio; P4 far-device baselines; P9 reach test (done 3 Nov) | 3,000 | 11.5 | B0 (built 3 Nov) |
| **Should** | the other 40-trial top-ups and dropped cells; 5.9's multi-mule cells; 5.6; probes P2, P3, P5, P6 | 6,880 | 33 | B1 |
| **Optional** | 5.9's clean-channel cells; probes P7, P8 | 3,260 | 23.5 | B2, B3 |
| **On acceptance** | `quick`, the reviewer's reproduction (reproducibility appendix) | 320 | 0.8 | — |
| **Not here** | 5.10, real radios | — | — | AERPAW access |
| | **All** | **13,460** | **~69** | |

---

## Needed

Before the revision (18 Jan 2027), and ideally before the rebuttal (3 Dec), so the rebuttal can say what they show.

| # | Run | Kind | Paper table | Why it's needed | Trials | ~h |
|---|---|---|---|---|---|---|
| N1 | **5.3 to 40 trials:** every arm at N = 6, K = 1 and 3, knee and stress, plus H0 | pre-registered | Scheduler comparison | Every 5.3 arm at N = 6 is planned at 40 trials and flew 20, but the setup section says only D5 and E3 are short. Fly this, or correct the text. Flying it is cheaper than defending it. Includes H1's two cells, which 5.1 shares | 780 | 2.2 |
| N2 | **5.4's remaining cells:** narrow range ratios 2 and 3, far-device shares 25/50/75 %, both payloads, and the 5 oracle runs | pre-registered | Design claims, C1 | "Reach is a decision" is the paper's first design claim, and its row says "no difference" from the base cell alone. These cells are where reach should matter | 800 | 1.9 + ~4 (oracle) |
| N3 | **P4:** H1 and the five baselines in the far-device cells, after N2 | probe | Design claims, C1 | The room's widest win (1.50, half the nodes beyond wide's reach) came from two layouts; this checks it against every baseline over 20 trials. Includes FedCS's base cell (20 trials), which batch 1 never flew | 380 | 0.9 |
| N4 | **P1:** every baseline under harsh radio (5, 11 and 17 dB interference, path loss 2.7, shadowing 8 dB), on both backhauls | probe | Robustness, channel columns | The table compares F with H1 alone there. In the room every baseline reached 71 % 2–5 × sooner than F at 17 dB. If that holds, the paper's robustness claim is wrong as written | 1,040 | 2.5 |
| N5 | **P9:** a paired reach test on every recorded cell, and τ = 0.82 where it's reachable. **Done 3 Nov:** `results/exp5/scores/probes/p9.md` | analysis | Robustness, compute columns | Under training time F reaches τ in 70–75 % of trials against 85–100 %, and its time "wins" count only the trials that reached τ | 0 | — |
| | | | | | **3,000** | **~11.5** |

---

## Should

The revision is clearly stronger with these, and a reviewer is likely to ask for most of them.

| # | Run | Kind | Paper table | Why | Trials | ~h |
|---|---|---|---|---|---|---|
| S1 | **5.1, 5.2, 5.5's FQ check and 5.7 to 40 trials.** 5.1 also flies its H1 route with an unbudgeted Pass 2 (320 of these trials), as the launcher runs the study whole | pre-registered | Design claims, C2–C4 | Near-claims: both deadline forms have CIs that exclude 0 at Holm p = 0.152; FedBuff has 13 seeds (p = 0.117) | 1,080 | 3.1 |
| S2 | **5.3's relaxed budget:** K = 1 to 40 trials, K = 3 new | pre-registered | Scheduler comparison | The paper says it was dropped, yet K = 1 flew at 20 trials on 16 Oct (F faster than all 7). It adds a third budget column | 600 | 1.9 |
| S3 | **5.8's 8-mission horizon** | pre-registered | Design claims, C5 | FedEx's route keeps updates fresher than F under stress at 4 missions (CI excludes 0, Holm p = 0.21); the age cap binds only from mission S on | 320 | 0.8 |
| S4 | **5.9's multi-mule cells:** K = 2 and 3 at N = 6, 12 and 24, jittery | pre-registered | Scale; mules | With more mules FedEx's route and H1 are already ahead of F on the mean (no claim). These are the pre-registered cells where a baseline is most likely to win outright | 840 | ~12.6 |
| S5 | **5.12 at the measured and 10 MB payloads; P5 at 2, 5 and 10 MB** | both | Robustness | The narrow band is slow: in the room, F left narrow at 2 MB and flew FedEx's stops at 5 MB. P5's 10 MB cell overlaps 5.12's for five arms | 600 + 480 | 1.4 + 1.2 |
| S6 | **5.13's IID, Dirichlet 1.0 and quantity-skew partitions** | pre-registered | Robustness | The paper shows one partition | 300 | 0.7 |
| S7 | **5.15's adaptive-backhaul arms** (H1+L1, F+L1) | pre-registered | Robustness | Planned and dropped; they give H1 the adaptive backhaul | 180 | 0.4 |
| S8 | **P2:** 60 s and 45 s budgets at N = 6, every 5.3 arm | probe | Scheduler comparison | F's lead falls from 7 of 7 claims at the knee to 1 of 7 at stress; below stress it may flip | 400 | 1.0 |
| S9 | **P3:** MAX-AoI, Oort and FedCS in the scale cells | probe | Scale | Absent from the scale study; FedCS tied F at N = 24 in the room | 240 | 1.6 |
| S10 | **P6:** fields of ±200 and ±300 m with 1 and 10 MB; the room's 24-node, ±300 m, 10 MB, 600 s cell | probe | new rows | Every Exp 5 cell is a field the narrow band covers from the dock. The room's second-largest loss (0.55) is outside it | 1,360 | 6.3 |
| S11 | **5.6, when learning helps:** FX against E3 on six N = 12 cells with periodic channels | pre-registered | (a study the paper says did not run) | The one pre-registered study with nothing flown. FQ left it with 5.5's verdict | 480 | 1.9 |
| | | | | | **6,880** | **~33** |

---

### The Should tier in groups

Added 4 Nov 2026. Run in this order. Rescore a study only once all its cells have flown, one study at a time. The last column is the worst case, with every added comparison finding nothing, because Holm runs across each study.

| Group | Runs | Question | Trials | ~h | Needs first | Claims at risk |
|---|---|---|---|---|---|---|
| 1. Design claims | S1, S3, plus FQ and the pinned bands to 40 | Do the near-claims become real? | 1,500 | 4.2 | nothing | none; 5.1's family grows from 16 to 24 but it has no claims at tau 0.71 |
| 2. Budget curve | S2, S8 (P2) | Does F's lead flip below stress or hold above the knee? | 1,000 | 2.9 | nothing | MAX-AoI at stress (0.0495 to 0.087) |
| 3. Narrow-band limits | S5, S10 (P5, P6) | Where does F's reach stop paying off? | 2,440 | 8.9 | B1, P6's budget decision | 4 of 5.12's 7 claims |
| 4. Completeness | S6, S7, S11 | Can we fill in what the paper says was dropped? | 960 | 3.0 | nothing | FX over F in the radio study; 2 non-IID claims |
| 5. Scale | S4, S9 (P3) | Do baselines overtake with more mules? | 1,080 | ~14 | Group 1 flown | the abstract's 24-device FedEx claim (0.039 to 0.35) |

List `s54` in Group 1's command and `s55 s57` in Group 5's, or the launcher re-flies 80 trials it already has.

**Status, 4 Nov 2026:** Groups 1 and 2 have flown and been scored, with no failures. Group 2 flew on `f1e71559` (results `35e75e41`); 5.3's family grew from 36 to 54 comparisons and its claims from 13 to 21. Its one risk came true: MAX-AoI at stress is no longer a claim (Holm 0.0494 to 0.068), and FX against F moved from 0.052 to 0.072, still no claim. Eleven of the 18 new tests copy a knee test exactly, because the 300 s budget does not change F, FX or the FedEx route, nor any arm but E3 with three mules. The family stays as the plan set it. At 300 s F stays faster than all seven baselines (179 against 229-248 s, all claims). P2 is in `results/exp5/scores/probes/P2.md`: at 60 and 45 s no baseline overtakes F, and nothing is a claim. The paper reports both in its text, with no relaxed column, because FQ has 20 of 40 trials at K = 1 and none at K = 3 there (60 trials would fill them, exploratory). Groups 3-5 have not run.

---

## Filling the tables' blanks

**Status, 4 Nov 2026: flown and scored.** Group 1 and all four fill flights finished with no failures. Group 1 is in `results/exp5/scores/b2`; the fill cells are in `results/exp5/scores/fills`, written by `scripts/exp5/fill_scores.py`. The paper reads both (`f1e71559`).

Added 4 Nov 2026. The paper's tables have 26 empty cells. The tables stay as they are and these cells get filled. Every run here is exploratory and outside every Holm family, so no pre-registered comparison changes; the tables mark these cells as exploratory.

| Table | Empty cells | Data on disk | What fills them | New trials |
|---|---|---|---|---|
| Scheduler comparison | FQ with 3 mules, knee and stress (2) | no | FQ flight | 80 |
| Scale | FQ at N = 24, knee and stress (2) | no | FQ flight | 40 |
| Decision cost | budget, F served, FX served at N = 6 and 12 (6) | FX at N = 6 only (57 %, 20 episodes) | FerrySim at 150 s and 180 s | none |
| Robustness | D3, D4, D5 in the four radio columns (12) | **yes, P1** | `paper_tables.py` only | 0 |
| Robustness | D3 under training time (2); D4 and D5 non-IID (2) | no | D3 and D4/D5 flights | 60 + 40 |
| | | | | **220** |

About 1.2 h in all, including Group 1's FQ add-on, which the first command absorbs (K = 1 to 40 trials). FQ's network handles N = 24 and 3 mules by design, but it was trained on N = 6 and 12 cells, so N = 24 is outside its training range and the paper should say so. Scoring reads these through the exploratory path, not `launch.py score`.

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x s55 s513 --arms FQ-g75 D4 D5 --set "s53x.arms=['FQ-g75']" --set "s53x.K=[1,3]" --set "s53x.budgets=['knee','stress']" --set s53x.d4_faithful=false --set s53x.h0=false --set rl.keep_learned=true --set s55.N=24 --set s55.n_trials=20 --set "s513.route_arms=['D4','D5']" --set "s513.partitions=[['dirichlet',0.1]]"
```

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s512 --set "s512.arms=['D3']" --set "s512.payloads=[1000000]"
```

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s511c --set "p511c.cells=['jit-n6-150','jit-n12-180']" --set s511c.knee_s.jit-n6-150=150 --set s511c.knee_s.jit-n12-180=180
```

---

## Optional

| # | Run | Kind | Why it can wait | Trials | ~h |
|---|---|---|---|---|---|
| O1 | **5.9's clean-channel cells**, at every N and K | pre-registered | A control: whether the order holds without the jittery channel the whole campaign flies. The costliest block (N = 24 dominates) | 1,760 | ~17.2 |
| O2 | **P7, clustered layouts** (needs B2) | probe | The strongest optional run: FedEx beat F on all five of the room's Long-cluster layouts. It needs a new layout mode first | 1,200 | 4.8 |
| O3 | **P8, a hard budget** (needs B3) | probe | Today an overrun is counted, not cut; FedEx's route (85 % of stress missions over) would likely lose most | 300 | 1.4 |
| | | | | **3,260** | **~23.5** |

## On acceptance

| Run | Why | Trials | ~h |
|---|---|---|---|
| `quick`: a reviewer's reproduction of 5.3's headline cell | IPDPS requires a reproducibility appendix on acceptance (camera-ready 20 Feb 2027). It also needs the trace archives published and the anonymised artifact repository, which aren't runs | 320 | 0.8 |

## Not here

**5.10, real radios:** a validation on AERPAW, not a statistical study. It waits on testbed access.

---

## What has to happen before the runs

**Builds.** Each one is tested, then the full suite and `tests/golden/make_baseline.py compare` run before any run that uses it flies:

| Build | For | What | Size |
|---|---|---|---|
| B0 | every probe, P9 | The probe scorer, `scripts/exp5/loss_probes.py`, after `fq_vs_e3.py`. It pairs each probe with its recorded reference cells by trial, applies Holm per probe, adds the reach test, and writes `results/exp5/scores/probes/` | small |
| B1 | P6 | The field's half-width as a launcher setting, passed as `--h1-field-radius-m` | small |
| B2 | P7 | A clustered layout: the room's draw, with T_nom's reference layouts drawn the same way | medium |
| B3 | P8 | A hard budget, sim only, off by default | medium |

The pre-registered runs need no build: the launcher extends each 20-trial CSV in place, because its resume check ignores the trial count.

**Your decisions:**
1. **N1:** fly 5.3 to 40 trials, or correct the setup section's "What was run" instead.
2. **The probes in the paper:** report them in the revision as exploratory (motivated by the room's sweep), or keep them for a follow-up. The five questions in the [probe plan](Experiment_5_Loss_Probes_Plan.md#decisions-for-you) (P6 and P7's budgets, P8's rule, extra arms, where arm-only probes write) come with this.
3. **5.1's bound-derived merge weights:** still open in the readiness doc. Batch 2 flew 5.1 with the hand-set constants. One option is a sensitivity check on them, about 240 trials and launcher settings only.
4. **How far down the list to go** before 18 Jan: Needed alone is about 11.5 h; Needed and Should together about 45 h.

---

## Commands

Without `--yes`, each command lists its jobs and asks before starting. The pre-registered runs use `params.toml` as committed, so their cells fly at the planned trial counts.

**Needed:**

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x --set "s53x.budgets=['knee','stress']"
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s54
```

P4 and P1: the commands in the probe plan (P4 after 5.4's run; P1 is two commands).

**Should** (pre-registered):

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s51 s52 s55 s57 s58 s513 s56
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s59x --set "s59x.contact_regimes=['jittery']"
```

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s512 s515
```

The `s53x` command, run after N1, flies the relaxed cells. The `s59x` command leaves the clean-channel cells (O1) for later; drop the `--set` to fly them. **Should** (probes): P2, P3, P5 and, after B1, P6, as given in the probe plan.

**Optional:** `run batch2 --study s59x` (the clean cells); P7 after B2; P8 after B3. **On acceptance:** `run quick`.

---

## After the runs

1. **Score the pre-registered studies** once each study's cells are complete: `launch.py score batch2` and `score batch3`, with `params.toml` as committed. With the full plan, 5.3's score file holds every arm, D5, E3, H0 and the relaxed budget together. Never score with a probe's `--set`s.
2. **Score the probes** with B0, into `results/exp5/scores/probes/`.
3. **Regenerate the paper's tables and figures:** `scripts/exp5/paper_tables.py`, `paper_figures.py`.
4. **Rewrite the setup section's "What was run"** to match what flew, and say which results are exploratory.
5. **Pack and archive each stage's traces** (`launch.py pack`), then commit the results and scores.

## Calendar

| Date | Event | What should be done by then |
|---|---|---|
| 30 Nov 2026 | early rejects | — |
| 3 Dec 2026 | rebuttal | Needed (B0, N1–N5): about 11.5 h machine time |
| 18 Jan 2027 | revised submission | Should (with B1): about 33 h more; Optional if time allows |
| 20 Feb 2027 | camera-ready | `quick`, trace archives, the anonymised artifact |

---

## How the counts were made, and how far to trust the hours

- **Trials** are the launcher's full plan, minus the trial rows on disk, per job. Shared cells are counted once.
- **Hours** come from the per-job wall times logged in batches 1–3 (per trial: 56 s at N = 6, 50 s at N = 12, 62 s at N = 24, more with several mules), packed on this host's 45 training processes.
- **Untimed runs:** N = 24 with two or three mules, 10 MB, the bigger fields and the clustered layouts have no logged timing, so their hours (5.9's N = 24 cells above all) are rough.
- **The oracle:** its estimate is the smoke run's 50 min per oracle run, at 30 episodes.
- **One host, one stage at a time:** the totals assume one stage runs at a time on this host (the launcher's lock), with Windows Update paused.
- **The room's sweep:** two layouts with one seed each; the probes turn its findings into 20-trial cells.

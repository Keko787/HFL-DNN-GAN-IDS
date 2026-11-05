# Experiment 5 — what has run and what is left

*Drawn up 29 Oct 2026. Tracker last checked 5 Nov 2026, through results `cf07705b` and paper `dcbc96dc`. Every Exp 5 run, pre-registered or added since, ranked by whether the paper needs it, with where each one stands. Sources: `launch.py status` and the job lists the launcher prints before a run, set against the trial rows on disk; the reports in `results/exp5/scores/probes/`; the loss-probe plan, [Experiment_5_Loss_Probes_Plan.md](Experiment_5_Loss_Probes_Plan.md); the stage-by-stage status in [Experiment_5_Readiness.md](Experiment_5_Readiness.md).*

---

## Where it stands

**Flown: 8,520 of the list's 13,460 trials. Left: 4,940 trials, about 31 h.** The Needed tier and the table fills are done. One Should run is left, and the Optional tier and `quick` have not started. The launcher shows no failed rows in any stage.

| Need | Runs | Done | Left | Trials planned | Trials left | ~h left | Builds first |
|---|---|---|---|---|---|---|---|
| **Needed** | N1–N5 | 5 of 5 | none | 3,000 | 0 | 0 | none left |
| **Should** | S1–S11 | 10 of 11 | S10 | 6,880 | 1,360 | 6.3 | B1, for S10 |
| **Optional** | O1–O3 | 0 of 3 | O1, O2, O3 | 3,260 | 3,260 | 23.5 | B2, B3 |
| **On acceptance** | `quick` | 0 of 1 | `quick` | 320 | 320 | 0.8 | — |
| **Not here** | 5.10, real radios | — | 5.10 | — | — | — | AERPAW access |
| | **All** | **15 of 20** | | **13,460** | **4,940** | **~31** | |

Batch 3 is complete. Batch 2 stands at 6,140 of 7,900 trials by `launch.py status`, and all that remains of it is 5.9's clean-channel cells.

This list was drawn up for IPDPS's revision. On 4 Nov 2026 the working target moved to NOMS 2027, with the paper due 9 Nov 2026. No run that is left is required to submit. S4 has already changed one thing: the abstract's 24-device FedEx claim no longer holds.

### Not done

| Run | Trials | ~h | Ready? | What it can change |
|---|---|---|---|---|
| S10: P6, fields beyond the narrow band's reach | 1,360 | 6.3 | Needs B1 and P6's budget decision | Exploratory. The one place F may lose outright; Group 3 stays open until it flies |
| O1: 5.9's clean-channel cells | 1,760 | ~17.2 | Ready | Pre-registered. A control: whether the order holds off the jittery channel. It doubles 5.9's family to 144, which at worst costs three of its eight claims (Holm 0.055 and 0.069) |
| O2: P7, clustered layouts | 1,200 | 4.8 | Needs B2 and P7's budget decision | Exploratory. FedEx beat F on all five of the room's Long-cluster layouts |
| O3: P8, a hard budget | 300 | 1.4 | Needs B3 and P8's rule | Exploratory. Cuts overruns off, which should cost FedEx's route most |
| `quick`: the reviewer's reproduction | 320 | 0.8 | On acceptance | Nothing in the paper |
| | **4,940** | **~31** | | |

Also open, and not in these counts:

- **5.10, real radios:** waits on AERPAW access.
- **5.1's bound-derived merge weights:** a sensitivity check, about 240 trials, if it is wanted (decision 3 below).
- **E3 on 5.6's own periods:** E3 flew the checkpoint trained at P_c 60 s, so retraining it for the study's periods is still open. Not costed.
- **Trace archives:** `results/exp5/archives` holds batch 1's and `sens`'s. Batches 2 and 3 and the probes are not packed, and nothing is published without a go-ahead.

### Done

| Flight | Runs | Trials | Results | Paper | What it changed |
|---|---|---|---|---|---|
| Needed tier, 3–4 Nov | N1–N5 | 3,000 | `3cffc966` | `2ae0f08f` | 5.3 moved to 40 trials, F's knee time to 179 s and the headline to 23–32 % |
| Group 1, 4 Nov | S1, S3, plus FQ and the pinned bands to 40 | 1,500 | `66410bd1` | `f1e71559` | 5.1, 5.2, 5.5 and 5.8 gained claims, and none was lost |
| Table fills, 4 Nov | all 26 blank cells | 220 | `66410bd1` | `f1e71559` | Every blank filled and marked exploratory |
| Group 2, 4 Nov | S2, S8 (P2) | 1,000 | `35e75e41` | `c10487f7` | MAX-AoI at stress lost its claim |
| FQ's relaxed cells, 4 Nov | table fill | 60 | `72a75334` | `19519e46` | A 300 s column in the scheduler table |
| Group 4, 4 Nov | S6, S7, S11 | 960 | `e8d6f72b` | `96a4406e` | FX over F lost its two radio claims, and F+L1 gained three |
| Group 3, first part, 4–5 Nov | S5 (5.12's payloads and P5) | 1,080 | `654e095c` | `3bf9ab7f` | 5.12 lost three claims and gained six; F's lead is gone by 10 MB |
| Group 5, 5 Nov | S4, S9 (P3) | 1,080 | `cf07705b` | `dcbc96dc` | 5.9 lost the 24-device FedEx claim and gained eight; FX beats F at N = 12 with three mules |
| | | **8,900** | | | |

The 8,900 includes 380 trials that were never on the list: Group 1's add-ons (100), the fills (220) and FQ's relaxed cells (60). P4 and P9 are not in the paper; their reports are in `results/exp5/scores/probes/`. P1, P2, P3 and P5 are, as exploratory results.

**Keeping this page current:** after each flight, add its row to Done, take it out of Not done, update the totals, set the Status columns below, and cite the results commit.

---

## The list as drawn up, 29 Oct 2026

Two kinds of runs were left, about **13,400 trials and 69 h** of machine time on this host in all:

- **The pre-registered plan.** The 8 Oct deadline trimmed batches 2 and 3 to 20 trials a cell and dropped some cells. Batch 2 stood at 2,260 of 9,220 trials and batch 3 at 580 of 1,360, which left 7,740 trials (about 44 h, plus 5.4's oracle). Two studies had never run: 5.6, and 5.10, which needs AERPAW.
- **The loss probes.** These were added after the CARE room's sweep showed where FeRRy loses: 5,380 trials (about 20 h) and four builds.

The paper needed about 11.5 h of this before its revision; most of the rest strengthens it.

**Needed** means one of three things:
- the submitted paper says something wrong without the run;
- the run tests the paper's central claim;
- the run is where the room or Exp 5 already shows FeRRy losing, and the paper's tables don't show it.

---

## Needed

All five are done. They were set for before the revision (18 Jan 2027), and ideally before the rebuttal (3 Dec).

| # | Run | Kind | Paper table | Why it's needed | Trials | ~h | Status |
|---|---|---|---|---|---|---|---|
| N1 | **5.3 to 40 trials:** every arm at N = 6, K = 1 and 3, knee and stress, plus H0 | pre-registered | Scheduler comparison | Every 5.3 arm at N = 6 is planned at 40 trials and flew 20, but the setup section says only D5 and E3 are short. Fly this, or correct the text. Flying it is cheaper than defending it. Includes H1's two cells, which 5.1 shares | 780 | 2.2 | Done, `3cffc966` |
| N2 | **5.4's remaining cells:** narrow range ratios 2 and 3, far-device shares 25/50/75 %, both payloads, and the 5 oracle runs | pre-registered | Design claims, C1 | "Reach is a decision" is the paper's first design claim, and its row says "no difference" from the base cell alone. These cells are where reach should matter | 800 | 1.9 + ~4 (oracle) | Done, `3cffc966` |
| N3 | **P4:** H1 and the five baselines in the far-device cells, after N2 | probe | Design claims, C1 | The room's widest win (1.50, half the nodes beyond wide's reach) came from two layouts; this checks it against every baseline over 20 trials. Includes FedCS's base cell (20 trials), which batch 1 never flew | 380 | 0.9 | Done, `3cffc966`; not in the paper |
| N4 | **P1:** every baseline under harsh radio (5, 11 and 17 dB interference, path loss 2.7, shadowing 8 dB), on both backhauls | probe | Robustness, channel columns | The table compares F with H1 alone there. In the room every baseline reached 71 % 2–5 × sooner than F at 17 dB. If that holds, the paper's robustness claim is wrong as written | 1,040 | 2.5 | Done, `3cffc966` |
| N5 | **P9:** a paired reach test on every recorded cell, and τ = 0.82 where it's reachable. **Done 3 Nov:** `results/exp5/scores/probes/p9.md` | analysis | Robustness, compute columns | Under training time F reaches τ in 70–75 % of trials against 85–100 %, and its time "wins" count only the trials that reached τ | 0 | — | Done, `3cffc966`; not in the paper |
| | | | | | **3,000** | **~11.5** | 5 of 5 done |

---

## Should

The revision is clearly stronger with these, and a reviewer is likely to ask for most of them. Ten of the eleven are done; S10 is left.

| # | Run | Kind | Paper table | Why | Trials | ~h | Status |
|---|---|---|---|---|---|---|---|
| S1 | **5.1, 5.2, 5.5's FQ check and 5.7 to 40 trials.** 5.1 also flies its H1 route with an unbudgeted Pass 2 (320 of these trials), as the launcher runs the study whole | pre-registered | Design claims, C2–C4 | Near-claims: both deadline forms have CIs that exclude 0 at Holm p = 0.152; FedBuff has 13 seeds (p = 0.117) | 1,080 | 3.1 | Done, Group 1, `66410bd1` |
| S2 | **5.3's relaxed budget:** K = 1 to 40 trials, K = 3 new | pre-registered | Scheduler comparison | The paper says it was dropped, yet K = 1 flew at 20 trials on 16 Oct (F faster than all 7). It adds a third budget column | 600 | 1.9 | Done, Group 2, `35e75e41` |
| S3 | **5.8's 8-mission horizon** | pre-registered | Design claims, C5 | FedEx's route keeps updates fresher than F under stress at 4 missions (CI excludes 0, Holm p = 0.21); the age cap binds only from mission S on | 320 | 0.8 | Done, Group 1, `66410bd1` |
| S4 | **5.9's multi-mule cells:** K = 2 and 3 at N = 6, 12 and 24, jittery | pre-registered | Scale; mules | With more mules FedEx's route and H1 are already ahead of F on the mean (no claim). These are the pre-registered cells where a baseline is most likely to win outright | 840 | ~12.6 | Done (Group 5); the FedEx claim lost |
| S5 | **5.12 at the measured and 10 MB payloads; P5 at 2, 5 and 10 MB** | both | Robustness | The narrow band is slow: in the room, F left narrow at 2 MB and flew FedEx's stops at 5 MB. P5's 10 MB cell overlaps 5.12's for five arms | 600 + 480 | 1.4 + 1.2 | Done, Group 3, `654e095c` |
| S6 | **5.13's IID, Dirichlet 1.0 and quantity-skew partitions** | pre-registered | Robustness | The paper shows one partition | 300 | 0.7 | Done, Group 4, `e8d6f72b` |
| S7 | **5.15's adaptive-backhaul arms** (H1+L1, F+L1) | pre-registered | Robustness | Planned and dropped; they give H1 the adaptive backhaul | 180 | 0.4 | Done, Group 4, `e8d6f72b` |
| S8 | **P2:** 60 s and 45 s budgets at N = 6, every 5.3 arm | probe | Scheduler comparison | F's lead falls from 7 of 7 claims at the knee to 1 of 7 at stress; below stress it may flip | 400 | 1.0 | Done, Group 2, `35e75e41` |
| S9 | **P3:** MAX-AoI, Oort and FedCS in the scale cells | probe | Scale | Absent from the scale study; FedCS tied F at N = 24 in the room | 240 | 1.6 | Done (Group 5); no baseline beats F |
| S10 | **P6:** fields of ±200 and ±300 m with 1 and 10 MB; the room's 24-node, ±300 m, 10 MB, 600 s cell | probe | new rows | Every Exp 5 cell is a field the narrow band covers from the dock. The room's second-largest loss (0.55) is outside it | 1,360 | 6.3 | **Not done**; needs B1 and P6's budget decision (Group 3) |
| S11 | **5.6, when learning helps:** FX against E3 on six N = 12 cells with periodic channels | pre-registered | (a study the paper says did not run) | The one pre-registered study with nothing flown. FQ left it with 5.5's verdict | 480 | 1.9 | Done, Group 4, `e8d6f72b` |
| | | | | | **6,880** | **~33** | 8 of 11 done |

---

### The Should tier in groups

Added 4 Nov 2026. Run in this order. Rescore a study only once all its cells have flown, one study at a time. The last column is the worst case, with every added comparison finding nothing, because Holm runs across each study.

| Group | Runs | Question | Trials | ~h | Needs first | Claims at risk | Status |
|---|---|---|---|---|---|---|---|
| 1. Design claims | S1, S3, plus FQ and the pinned bands to 40 | Do the near-claims become real? | 1,500 | 4.2 | nothing | none; 5.1's family grows from 16 to 24 but it has no claims at tau 0.71 | Done; none lost |
| 2. Budget curve | S2, S8 (P2) | Does F's lead flip below stress or hold above the knee? | 1,000 | 2.9 | nothing | MAX-AoI at stress (0.0495 to 0.087) | Done; MAX-AoI at stress lost (Holm 0.068) |
| 3. Narrow-band limits | S5, S10 (P5, P6) | Where does F's reach stop paying off? | 2,440 | 8.9 | B1, P6's budget decision | 4 of 5.12's 7 claims | **Part done:** S5 flown, 3 of those claims lost; S10 (P6) not flown |
| 4. Completeness | S6, S7, S11 | Can we fill in what the paper says was dropped? | 960 | 3.0 | nothing | FX over F in the radio study; 2 non-IID claims | Done; FX over F lost on the default channel and at 8 dB, and the non-IID claims held |
| 5. Scale | S4, S9 (P3) | Do baselines overtake with more mules? | 1,080 | ~14 | Group 1 flown | the abstract's 24-device FedEx claim (0.039 to 0.35) | Done; the FedEx claim lost (Holm 0.29) |

List `s55 s57` whenever flying or scoring 5.9, as the Commands section does, or the launcher re-flies 80 trials it already has.

**Results by group, with no failures in any of them:**

- **Group 1** flew on `2ae0f08f` (results `66410bd1`). 5.1 went from 0 to 5 claims (FedBuff loses), 5.2 from 0 to 4 (the round deadline and the preferred duration lose on miss rate), 5.5 from 0 to 1 (FX beats F at the N = 12 knee, 428 against 461 s, Holm 0.032; FQ against FX stays no claim) and 5.8 from 5 to 10, with the five holding. 5.7 stays at 2 and 5.4 at 0.
- **Group 2** flew on `f1e71559` (results `35e75e41`). 5.3's family grew from 36 to 54 comparisons and its claims from 13 to 21. Its one risk came true: MAX-AoI at stress is no longer a claim (Holm 0.0494 to 0.068), and FX against F moved from 0.052 to 0.072, still no claim. Eleven of the 18 new tests copy a knee test exactly, because the 300 s budget does not change F, FX or the FedEx route, nor any arm but E3 with three mules. The family stays as the plan set it. At 300 s F stays faster than all seven baselines (179 against 229-248 s, all claims). P2 is in `results/exp5/scores/probes/P2.md`: at 60 and 45 s no baseline overtakes F, and nothing is a claim. The paper reports both in its text.
- **FQ's relaxed cells** then flew to 40 trials to fill a 300 s column in the scheduler table (60 trials, exploratory, results `72a75334`, in `results/exp5/scores/fills`): 168 s with one mule and 62 s with three, outside 5.3's family.
- **Group 3's payload runs** flew on `96a4406e` on 4-5 Nov (results `654e095c`; S5: 5.12's measured and 10 MB cells, 600 trials; P5 at 2, 5 and 10 MB, 480 trials). 5.12's family grew from 12 to 36 comparisons and its claims from 7 to 10. Three of the four at risk were lost: FX over F at 1 MB with no training time (Holm 0.047 to 0.177), and F over H1 (0.016 to 0.052) and FedCS (0.047 to 0.168) at 1 MB with training spread. F over the FedEx route there held at 0.041. The six new claims are F over H1, the FedEx route and FedCS at the measured payload, with no training time and with spread. F flies narrow at 18.8 KB and 1 MB, medium at 2 MB and wide from 5 MB. Its lead over H1 is 214 s at 18.8 KB and 88 s at 1 MB, and no claim from 2 MB up; no baseline beats it on time or reach at any payload. P5's report is `results/exp5/scores/probes/P5.md`. The paper reports both in its text, and the robustness table loses two marks.
- **Group 4** flew on `19519e46` (results `e8d6f72b`; S6, S7 and S11: 960 trials). 5.13's family grew from 4 to 16 comparisons and its claims from 3 to 9, and the three at Dir(0.1) held. 5.15's family grew from 10 to 22 and its claims from 2 to 4, counting the plan's two mission-backhaul tests the 9 Oct trim left out, which reuse 5.3's rows. Its risk came true: FX over F is no longer a claim on the default channel (Holm 0.0455 to 0.081) or with 8 dB shadowing (0.0297 to 0.051). Without those two tests the 8 dB test would pass at 0.049; the family stays as the plan set it. F+L1 beats F at 17 dB, n_pl 2.7 and 8 dB, and H1+L1 makes no claim. 5.6 ran for the first time: FX beats E3 by 78-89 s at 180 s in all three channels and leads without a claim at 90 s. E3 flew the checkpoint trained on the `jittery` family (P_c 60 s), as the plan set it, so the study reads E3 off its training period. The paper reports all three in its text, and the robustness table loses FX's two marks.
- **Group 5** flew on `9f20f7b2` on 5 Nov (results `cf07705b`; S4: 840 trials in 5.7 h; P3: 240 in 1.7 h). 5.9 is scored on the jittery channel only, with `s55 s57` listed. Its family grew from 8 to 72 comparisons and its claims from 1 to 8. Its risk came true: F over the FedEx route at N = 24 under stress is no longer a claim (661 against 752 s, Holm 0.039 to 0.29), and the abstract, the scale section and the conclusion still state it. Six of the eight claims are F's with one mule: over H1 and Whittle at the N = 24 knee (749 against 951 and 911 s), over H1, Whittle and FedEx at the N = 6 knee on its first 20 trials, and over FedEx at N = 6 under stress. The other two are FX over F at N = 12 with three mules (102 against 151 s, Holm 0.033), one flight counted twice: the stress budget does not change F, FX or the FedEx route there. FX's lead is uneven: it crosses the target within 45 s in 13 trials to F's 8, and six of the 20 trials carry 93 % of the gap. In these cells, with two or three mules, F has no claim over a baseline and no baseline has one over F; elsewhere, 5.3 keeps F's claim over FedEx with its own merge at N = 6 with three mules. Twelve of the 72 tests copy another exactly; without them Holm gives the same verdicts and the FedEx test reaches 0.24. The family stays as the plan set it. P3 makes 1 claim of 24: F over MAX-AoI at the N = 24 knee (749 against 948 s). No baseline beats F. F's leads over Oort at the knee and FedCS under stress, both at N = 24, stop at Holm 0.061. The report is `results/exp5/scores/probes/P3.md`. The paper reports both in its text. The scale table loses that mark and gains MAX-AoI, Oort and FedCS, marked exploratory, and the mule table gains N = 24. A later paper commit replaces the scale, cost and mule tables with one 3-panel figure; all three stay generated in `results/exp5/paper/exp5_tables.tex`.
- **Not run:** Group 3's second part (S10, P6).

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

| # | Run | Kind | Why it can wait | Trials | ~h | Status |
|---|---|---|---|---|---|---|
| O1 | **5.9's clean-channel cells**, at every N and K | pre-registered | A control: whether the order holds without the jittery channel the whole campaign flies. The costliest block (N = 24 dominates) | 1,760 | ~17.2 | **Not done**; ready |
| O2 | **P7, clustered layouts** (needs B2) | probe | The strongest optional run: FedEx beat F on all five of the room's Long-cluster layouts. It needs a new layout mode first | 1,200 | 4.8 | **Not done**; needs B2 and P7's budget decision |
| O3 | **P8, a hard budget** (needs B3) | probe | Today an overrun is counted, not cut; FedEx's route (85 % of stress missions over) would likely lose most | 300 | 1.4 | **Not done**; needs B3 and P8's rule |
| | | | | **3,260** | **~23.5** | 0 of 3 done |

## On acceptance

| Run | Why | Trials | ~h | Status |
|---|---|---|---|---|
| `quick`: a reviewer's reproduction of 5.3's headline cell | IPDPS requires a reproducibility appendix on acceptance (camera-ready 20 Feb 2027). It also needs the trace archives published and the anonymised artifact repository, which aren't runs | 320 | 0.8 | **Not done** |

## Not here

**5.10, real radios:** a validation on AERPAW, not a statistical study. It waits on testbed access. **Not done.**

---

## What has to happen before the runs

**Builds.** Each one is tested, then the full suite and `tests/golden/make_baseline.py compare` run before any run that uses it flies:

| Build | For | What | Size | Status |
|---|---|---|---|---|
| B0 | every probe, P9 | The probe scorer, `scripts/exp5/loss_probes.py`, after `fq_vs_e3.py`. It pairs each probe with its recorded reference cells by trial, applies Holm per probe, adds the reach test, and writes `results/exp5/scores/probes/` | small | Built 3 Nov |
| B1 | P6 | The field's half-width as a launcher setting, passed as `--h1-field-radius-m` | small | Not built |
| B2 | P7 | A clustered layout: the room's draw, with T_nom's reference layouts drawn the same way | medium | Not built |
| B3 | P8 | A hard budget, sim only, off by default | medium | Not built |

The pre-registered runs need no build: the launcher extends each 20-trial CSV in place, because its resume check ignores the trial count.

**Your decisions:**
1. **N1:** decided and flown; 5.3 is at 40 trials.
2. **The probes in the paper:** decided in part. The paper reports P1, P2, P3 and P5 as exploratory, and P4 and P9 stay out. Still open from the [probe plan](Experiment_5_Loss_Probes_Plan.md#decisions-for-you): P6 and P7's budgets, P8's rule and extra arms.
3. **5.1's bound-derived merge weights:** still open in the readiness doc. Batch 2 flew 5.1 with the hand-set constants. One option is a sensitivity check on them, about 240 trials and launcher settings only.
4. **How far down the list to go:** with NOMS due 9 Nov 2026, whether to fly P6 before submitting or after.

---

## Commands

Without `--yes`, each command lists its jobs and asks before starting. The pre-registered runs use `params.toml` as committed, so their cells fly at the planned trial counts.

**Still to fly:**

S10, P6, after B1, as given in the probe plan. **Optional:** `run batch2 --study s59x s55 s57 s56` (the clean cells; `s56` keeps two more shared cells); P7 after B2; P8 after B3. **On acceptance:** `run quick`.

**Flown, kept for the record:**

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x --set "s53x.budgets=['knee','stress']"
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s54
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s51 s52 s55 s57 s58 s513 s56
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x
```

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s512 s515
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s59x s55 s57 --set "s59x.contact_regimes=['jittery']"
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s59x --set "s59x.arms=['D1','D2','D5']" --set "s59x.N=[12,24]" --set "s59x.K=[1]" --set "s59x.contact_regimes=['jittery']"
```

The first `s53x` command is N1; the second, run after it, flies the relaxed cells. The last two are S4 and S9. P1, P2, P4 and P5 flew with the commands in the probe plan. S4 is scored with the same `--study` and `--set` as its flight; `score` rewrites `scores/b2/index.md` with only the studies it was given, so merge the other rows back.

---

## After the runs

1. **Score the pre-registered studies** once each study's cells are complete: `launch.py score batch2` and `score batch3`, with `params.toml` as committed. With the full plan, 5.3's score file holds every arm, D5, E3, H0 and the relaxed budget together. Never score with a probe's `--set`s.
2. **Score the probes** with B0, into `results/exp5/scores/probes/`.
3. **Regenerate the paper's tables and figures:** `scripts/exp5/paper_tables.py`, `paper_figures.py`.
4. **Rewrite the setup section's "What was run"** to match what flew, and say which results are exploratory.
5. **Pack and archive each stage's traces** (`launch.py pack`), then commit the results and scores.

## Calendar

These are IPDPS's dates, from when this list was written. The working target is now NOMS 2027: paper due 9 Nov 2026, notification 15 Feb 2027, camera-ready 28 Feb 2027. Whether the IPDPS paper is under review is still open.

| Date | Event | What should be done by then |
|---|---|---|
| 30 Nov 2026 | early rejects | — |
| 3 Dec 2026 | rebuttal | Needed (B0, N1–N5): about 11.5 h machine time. **Done** |
| 18 Jan 2027 | revised submission | Should (with B1): about 33 h more; Optional if time allows. **10 of 11 Should runs done;** B1 and about 6.3 h left |
| 20 Feb 2027 | camera-ready | `quick`, trace archives, the anonymised artifact |

---

## How the counts were made, and how far to trust the hours

- **The tracker** was checked on 5 Nov 2026 against `launch.py status`, the probe reports and the launcher's job lists. The trials left agree: 5.9's remaining 1,760 (O1), `quick`'s 320, and 2,860 of probes (P6, P7 and P8).
- **Trials** are the launcher's full plan, minus the trial rows on disk, per job. Shared cells are counted once.
- **Hours** come from the per-job wall times logged in batches 1–3 (per trial: 56 s at N = 6, 50 s at N = 12, 62 s at N = 24, more with several mules), packed on this host's 45 training processes.
- **Untimed runs:** the bigger fields and the clustered layouts have no logged timing, so their hours are rough. S4 took 5.7 h against its ~12.6 h, so O1's ~17.2 h, drawn the same way, is likely high.
- **The oracle:** its estimate is the smoke run's 50 min per oracle run, at 30 episodes.
- **One host, one stage at a time:** the totals assume one stage runs at a time on this host (the launcher's lock), with Windows Update paused.
- **The room's sweep:** two layouts with one seed each; the probes turn its findings into 20-trial cells.

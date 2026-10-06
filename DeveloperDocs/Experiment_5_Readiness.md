# Experiment 5 — Readiness

What is left before the Exp 5 sweep can run, stage by stage: what is ready, what is not, and why.

*As of 6 Oct 2026, about 10:00: batch 1 and `sens` done and scored (`ff90bbf`, `53bdfc5`), and the RL campaign paused after its calibration for a decision ([findings](Experiment_5_RL_Calibration_Findings.md)). Before that: the S\* pilot (commit `6e4b6c3`), the re-pin (`77880dc`), the smoke run of every stage (its fixes `b54185a`) and the code gate; the `"auto"` machine limits are `4b78194`. The live view is `exp5 status` (or `exp5 check`); the full procedure is the [reproducibility guide](Experiment_5_Reproducibility_Guide.md).*

---

## In one paragraph

The code, the launcher, every stage's definition, the scoring and the documentation are built. The first three pilots are done: the session timeout (TTL), the budget knees with τ, and the age cap S\* (2 at every N). **Batch 1 and batch 3's `pilot3` have every setting they need.** The FerrySim re-pin is done, so the RL stages can run: FerrySim's cells now fly the stack's budgets. Batch 2 needs the RL verdict, and batch 3 needs its own pilot. Every stage has passed a smoke run, and the code gate passed. From S\* on, the campaign runs on a second host; see "The host" below.

---

## Is batch 1 ready?

**Yes.**

| | |
|---|---|
| What it has | The knee and stress budgets per N, the session timeouts, τ = 0.71, S\* = 2 at every N, and every other batch 1 setting. All 70 of its stack jobs pass the runner's argument checks with the real values (`exp5 validate batch1`). |
| Size | 1,920 trials (5.3 core, 5.9 core, 5.11 (b), 5.14) plus 3 FerrySim jobs (5.11 (a)). About 6 h on the second host (the launcher estimates 4.4 h at its `"auto"` limits, 4.9 h at the first host's; the knee ran about 1.3 times its estimate). |
| Do first | Nothing: the re-pin (which renamed the FerrySim cells 5.11 (a) flies), the smoke run and the code gate are done. |
| Order choice | The campaign runs batch 1 after the RL verdict. That is not because its runs could become invalid: F, FX and FQ are separate arms, and batch 2 adds FQ to batch 1's own cells on the same seeds. The verdict decides which arm the paper calls FeRRy. If the learned score is kept, 5.3 and 5.9 get FQ in batch 2, while 5.11 (b) and 5.14 would show only F and FX unless extended. Batch 1 can therefore also run right after the re-pin and smoke test, before RL. |

---

## Every stage

| Stage | Status | Waiting on | Why |
|---|---|---|---|
| `ttl` | ✅ Done | — | Session timeouts 36, 34, 34, 23 s at N = 6, 12, 18, 24. |
| `knee` | ✅ Done | — | Knees 150, 180, 240, 262 s (stress half of each); 120 s at N = 6 with the measured payload; τ = 0.71. N = 6's knee is the grid's largest budget, accepted as is. |
| `sstar` | ✅ Done | — | S\* = 2 at N = 6, 12, 18 and 24: one mission covers 90% of layouts at each knee, two at each stress budget (N ≥ 12 by the tool's greedy bound). Ran on the second host; the tool has no timing in it. |
| `quick` | ⏳ After batch 1 | Batch 1's recorded CSVs to compare against | A reviewer's reproduction of 5.3's headline cell (320 trials). It needs a recorded batch 1 to compare with. |
| `batch1` | ▶️ Ready | — | See above. |
| `sens` | ⏳ After batch 1 | Batch 1's CSVs | 5.3's F, FX and H1 at the session timeout × 0.75 and × 1.5. × 1 is batch 1's own cell. |
| `rl-headroom`, `rl-calibrate` | ✅ Done, 6 Oct | — | ε = 0.01 everywhere (headroom 0.012–0.042). Calibration: `jittery56` flat, `clean` sanity-failed by 0.0005; the learned score sits at FX's level, below greedy_1. See [the findings](Experiment_5_RL_Calibration_Findings.md). |
| `rl-sweep`, `rl-e3` | ▶️ Running from 6 Oct 13:58 | — | Option A, decided 6 Oct: the sweep as pre-registered, then E3's trainings ([findings](Experiment_5_RL_Calibration_Findings.md#decision)). |
| `rl-s57` | ❌ Blocked | Study 5.5's verdict (`rl.gamma_star`, `rl.keep_learned`) | 5.7's scores train only if the verdict keeps the learned score. |
| `batch2` | ❌ Blocked | The RL verdict (`rl.keep_learned`, the `rl.checkpoints` paths); the 5.1 weights decision (below) | 5.1, 5.2, the rest of 5.3, 5.4, 5.5's stack check, 5.6–5.8, the rest of 5.9, 5.13: about 11,300 trials, roughly 50 h. FQ and E3 arms fly checkpoints that don't exist yet. |
| `pilot3` | ▶️ Ready | — | 5.15's interference levels, 5.12's training-time levels (p512) and 5.11 (c)'s FerrySim sweeps: 80 trials plus FerrySim. Its stack jobs fly H1 (no cap) and its FerrySim jobs the scale cells, which the re-pin leaves alone. |
| `batch3` | ❌ Blocked | `pilot3`'s outputs (`pilot_outputs.train_levels` by `--apply`; `[s515] harsher_amp_db`, `lossier_n_pl` and `[s511c] knee_s` by hand) | 5.12, 5.15, 5.11 (c): about 1,460 trials, roughly 3 h. |
| Study 5.10 | ❌ Not here | AERPAW access | Real radios; validation, not a statistical study. |

---

## What is left, in order

| # | Step | Command or action | Time |
|---|---|---|---|
| 1 | ✅ The S\* pilot and its output (S\* = 2 at every N); commit it | `exp5 run sstar --yes`, `exp5 report sstar --apply` | done |
| 2 | ✅ The re-pin: FerrySim's N = 6 cells to (75, 150) s and N = 12 to (90, 180) s; caps 2; Study 5.6's lags 27 and 34 s (periods 108/54 and 136/68 s); the hashes and the test pins. Two tests that held the old budgets as literals were fixed. Commit it, with `[rl] repinned = true` | `python scripts/exp5/repin.py --dry-run`, then `--workers 8`; the FerrySim tests | done (26 s, then about 5 min of tests) |
| 3 | ✅ A smoke run of every stage after the pilots, 5–6 Oct: one trial per job (RL trainings at 30 episodes), stand-ins for the undecided values (γ\* 0.9 with the learned score kept, the smoke verdict's checkpoints, pilot3's example values). 10 stages and 310 jobs, then batch 2's 361; every trial row ok. It found two problems, both in smoke runs alone, fixed in `b54185a`: smoke trainings stopped short of the learner's 1,000-transition warm-up, so Study 5.7's ablation checkpoints were one network and their evaluation refused them (smoke now trains past a warm-up of 64); and `report` on an RL stage did not show the verdict | `exp5 run <stage> --smoke --yes`, stage by stage, with `--set` stand-ins for the RL decisions and pilot3's values | done (about 3.5 h; Study 5.4's O1 oracle took 50 min of it at its full 30 episodes, and smoke now flies it at 2) |
| 4 | ✅ The code gate, 6 Oct, on `b54185a`: 6,375 tests, the five known failures with their recorded signatures, the same as all three baselines. None of the five last-bit `erfc` differences the guide expects on another host appeared | `python -m pytest tests …` then `tests/golden/make_baseline.py compare` | done (18 min) |
| 5 | ✅ Batch 1, then `sens`, 6 Oct 00:36–06:40 from `c461552`: 1,760 + 120 trials, every row ok; scored (`exp5 score batch1`, `score sens`) into `results/exp5/scores/`; committed `ff90bbf` (results) and `53bdfc5` (scores, headlines in its message); traces in `results/exp5/archives/{b1,sens}_traces.tar.gz` | `exp5 run batch1 --yes`, `exp5 run sens --yes` | done (6 h) |
| 6 | The RL campaign. It pauses after the calibration (a sanity check to read) and after Study 5.5's verdict. **Headroom and calibration done (6 Oct 06:42–09:17); paused for the decision recorded in [the findings](Experiment_5_RL_Calibration_Findings.md)** | `exp5 run rl --yes` | ~2.5 h done; the sweep ~10 h |
| 7 | Act on the verdict: set `rl.gamma_star`, `rl.keep_learned` and the `rl.checkpoints` paths; build FX-dwell and FX-cov (if FX is kept) or M1 (if the learned score is kept) | params.toml; a build | decisions + build |
| 8 | Batch 2 | `exp5 run batch2 --yes` | ~50 h |
| 9 | Batch 3's pilots, then their outputs | `exp5 run pilot3 --yes`, `exp5 report pilot3 --apply`, the rest by hand | ~1 h |
| 10 | Batch 3 | `exp5 run batch3 --yes` | ~3 h |
| 11 | Score each batch, archive each stage's traces, commit; publish the archives (with your go-ahead) | `exp5 score batches`, `exp5 pack all` | ~1 h |

Machine time from step 1 to step 11 is about 3.5–4 days, plus the pauses for reading reports and deciding. Every remaining stage runs on one host, the second ([guide §9](Experiment_5_Reproducibility_Guide.md#9-determinism-what-matches-across-machines)). Keep it awake and pause Windows Update for the long stages.

### The host

The TTL and knee pilots ran on the first host: 8 physical cores, 16 logical, 95 GB, Windows 10. From S\* on, the campaign runs on a second host by choice (5 Oct 2026): 12 physical cores, 20 logical (8 performance and 4 efficiency cores), 64 GB, Windows 11. The launcher's machine limits are now `"auto"`. On this host they come to 7 jobs, 45 training processes, a 51 GB memory budget and 18 RL trainings ([guide §2.1](Experiment_5_Reproducibility_Guide.md#21-the-machine)). Stack trials here fly the first host's session timeouts and knees, and `check` and `run` say so. Comparisons between arms stay valid, since every arm runs on this host. Only the timeouts and budgets were measured elsewhere. To make them this host's own, run `ttl` and `knee` here under a fresh `--out-root`, then S\* again: about 4 h, since the knee took 4 h and the TTL pilot 5 min on the first host.

---

## Decisions still open

| Decision | Needed before | Options |
|---|---|---|
| **5.1's bound-derived merge weights.** `agg:cutoff`'s age weights are FedAsync's hinge with hand-set constants; the "derived from the bound" side is a theory-track derivation not yet done. 5.7 cites it too. | Batch 2 (5.1 and 5.7) | Drop the derivation and add a small sensitivity check on the hand-set constants (~240 trials, launcher settings only); drop it with nothing added; or do the derivation (research work), after which it becomes a weight mode and a 5.1 variant. |
| **Batch 1 before or after RL** | Step 5 | After RL, as the campaign orders it; or right after the re-pin and smoke test. See "Order choice" above. |
| **The verdict's follow-ups:** γ\*, keep the learned score or not, the checkpoints; FX-dwell/FX-cov or M1 | Batch 2 | Read from `exp5 report rl-sweep`. |
| **pilot3's hand-set values:** 5.15's harsher interference amplitude and higher path-loss exponent, 5.11 (c)'s knees | Batch 3 | Read from `exp5 report pilot3`. |
| **Publishing the trace archives** (a GitHub release, Zenodo for the paper's artifact) | The paper | Nothing is published without your go-ahead. |

Decided already, for reference: τ from the knee pilot (0.71; 0.82 kept as a second τ); 4 missions per trial; the jittery contact channel everywhere; N = 6's knee accepted at 150 s; E3's γ = 0.99 (Chen's code); FedProx ρ = 0.01; 5.6 at 40 trials per cell; 5.12's levels from the p512 pilot; traces in per-stage archives, outside git; `agg:seq` out; p512's band read after the first mission; after the calibration, the 5.5 sweep as pre-registered (option A).

---

## What could still go wrong

- **The session timeouts and knees came from another host.** This CPU's cores are faster, so its fits are likely shorter and the timeouts more lenient than measured. That is a guess, not a measurement. See "The host" above.
- **p512's band and the first mission.** In the smoke run (one trial per level) even the shortest training time, 45 s, left 37.5% of Pass-1 contacts with no update ready, against the 10–30% band. Mission 1 accounts for most of it: every device starts its first fit at the mule's first takeoff, so mission 1's contacts, seconds later, find almost none ready (5 of 6 at 45 s), while missions 2–4 found 4 of 18 (22%, inside the band). Mission 1 is a quarter of the contacts, so it adds about 20 points at every level. **Decided 6 Oct 2026:** the band is read after each mule's first mission (the scorer's `not_ready_share_after_first`), and the share over every mission is reported beside it. On the smoke run's data the rule then picks 45 s.
- **A pilot rule can come back empty.** p512 refuses if no training-time level gives 10–30% not-ready contacts (change `[p512] cycle_factors` and rerun). (The re-pin's own check passed, narrowly: at 180 s the ratio bound is 1.48 against 1.5.)
- **τ = 0.71 is set by the larger sizes.** At N = 18 and 24, accuracy levels off near 0.71 whatever the budget. At N = 6, where 80% of trials reach 0.82, most trials reach 0.71 early, so time to τ separates the arms less there. Final accuracy and reach at 0.82 are reported beside it.
- **The N = 6 cells leave the learned score little to decide.** At their measured budgets one stop serves all six devices in most missions. At 150 s, every pair FX's in-flight slot weighed over 24 episodes was the dock. They are controls, so that is their role, but they add little training signal to the `jittery` family.
- **The code gate** must be rerun after the re-pin; host-specific differences are listed in the [guide §2.7](Experiment_5_Reproducibility_Guide.md#27-the-code-gate-path-b-and-before-reporting-results).

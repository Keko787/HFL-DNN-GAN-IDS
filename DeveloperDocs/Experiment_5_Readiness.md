# Experiment 5 — Readiness

What is left before the Exp 5 sweep can run, stage by stage: what is ready, what is not, and why.

*As of 5 Oct 2026, after the knee pilot (commit `25721608`). The live view is `exp5 status` (or `exp5 check`); the full procedure is the [reproducibility guide](Experiment_5_Reproducibility_Guide.md).*

---

## In one paragraph

The code, the launcher, every stage's definition, the scoring and the documentation are built. The first two pilots are done: the session timeout (TTL) and the budget knees, including τ. **Only the next pilots can run yet:** S\* and batch 3's `pilot3`. Every stage that flies the planner needs the age cap S\*, whose pilot takes minutes. The RL stages need the FerrySim re-pin, which is scripted and takes about 3 minutes plus tests. Batch 2 needs the RL verdict, and batch 3 needs its own pilot. No stage beyond the pilots has had a smoke run yet.

---

## Is batch 1 ready?

**Not yet: it is one short stage away.**

| | |
|---|---|
| What it needs | `pilot_outputs.s_star` at N = 6, 12, 18 and 24. Every plan arm (F, FX, F+L1, …) flies the age cap S\*, which is unset until the `sstar` stage runs and `exp5 report sstar --apply` writes it. |
| What it already has | The knee and stress budgets per N, the session timeouts, τ = 0.71, and every other batch 1 setting. With placeholder pilot values, all of its jobs pass the runner's argument checks. |
| Size | 1,920 trials (5.3 core, 5.9 core, 5.11 (b), 5.14) plus 3 FerrySim jobs (5.11 (a)). About 6.5 h on this machine (the launcher estimates 4.9 h; the knee ran about 1.3 times its estimate). |
| Do first | The re-pin (5.11 (a) flies FerrySim's cells by name, which the re-pin renames) and a smoke run. |
| Order choice | The campaign runs batch 1 after the RL verdict. That is not because its runs could become invalid: F, FX and FQ are separate arms, and batch 2 adds FQ to batch 1's own cells on the same seeds. The verdict decides which arm the paper calls FeRRy. If the learned score is kept, 5.3 and 5.9 get FQ in batch 2, while 5.11 (b) and 5.14 would show only F and FX unless extended. Batch 1 can therefore also run right after the re-pin and smoke test, before RL. |

---

## Every stage

| Stage | Status | Waiting on | Why |
|---|---|---|---|
| `ttl` | ✅ Done | — | Session timeouts 36, 34, 34, 23 s at N = 6, 12, 18, 24. |
| `knee` | ✅ Done | — | Knees 150, 180, 240, 262 s (stress half of each); 120 s at N = 6 with the measured payload; τ = 0.71. N = 6's knee is the grid's largest budget, accepted as is. |
| `sstar` | ▶️ Ready | — | 4 S\* tool runs, minutes. Then `exp5 report sstar --apply`. |
| `quick` | ⏳ After S\* | `pilot_outputs.s_star.6`, and batch 1's recorded CSVs to compare against | A reviewer's reproduction of 5.3's headline cell (320 trials). It needs a recorded batch 1 to compare with. |
| `batch1` | ⏳ After S\* | `pilot_outputs.s_star` | See above. Best after the re-pin and a smoke run. |
| `sens` | ⏳ After batch 1 | `pilot_outputs.s_star.6`, batch 1's CSVs | 5.3's F, FX and H1 at the session timeout × 0.75 and × 1.5. × 1 is batch 1's own cell. |
| `rl-headroom`, `rl-calibrate`, `rl-sweep`, `rl-e3` | ❌ Blocked | The re-pin (`[rl] repinned = true`) | The learned score must train on the budgets the stack flies. FerrySim's cells still fly placeholders (45/90 s at N = 6, 120/180 s at N = 12). `rl-calibrate` and `rl-sweep` also need `rl-headroom`'s report. |
| `rl-s57` | ❌ Blocked | The re-pin; Study 5.5's verdict (`rl.gamma_star`, `rl.keep_learned`) | 5.7's scores train only if the verdict keeps the learned score. |
| `batch2` | ❌ Blocked | S\*; the RL verdict (`rl.keep_learned`, the `rl.checkpoints` paths); the 5.1 weights decision (below) | 5.1, 5.2, the rest of 5.3, 5.4, 5.5's stack check, 5.6–5.8, the rest of 5.9, 5.13: about 11,300 trials, roughly 50 h. FQ and E3 arms fly checkpoints that don't exist yet. |
| `pilot3` | ▶️ Ready | — | 5.15's interference levels, 5.12's training-time levels (p512) and 5.11 (c)'s FerrySim sweeps: 80 trials plus FerrySim. Its stack jobs fly H1 (no cap) and its FerrySim jobs the scale cells, which the re-pin leaves alone. |
| `batch3` | ❌ Blocked | `pilot3`'s outputs (`pilot_outputs.train_levels` by `--apply`; `[s515] harsher_amp_db`, `lossier_n_pl` and `[s511c] knee_s` by hand); S\* | 5.12, 5.15, 5.11 (c): about 1,460 trials, roughly 3 h. |
| Study 5.10 | ❌ Not here | AERPAW access | Real radios; validation, not a statistical study. |

---

## What is left, in order

| # | Step | Command or action | Time |
|---|---|---|---|
| 1 | The S\* pilot, then write its output and commit | `exp5 run sstar --yes`, `exp5 report sstar --apply` | minutes |
| 2 | The re-pin: FerrySim's N = 6 cells to (75, 150) s and N = 12 to (90, 180) s, the caps, the 5.6 lags and periods, the hashes and the test pins | `python scripts/exp5/repin.py --dry-run`, then `--workers 8`; the FerrySim tests; commit; set `[rl] repinned = true` | ~3 min, then ~10 min of tests |
| 3 | A smoke run of every stage: one trial per job (RL trainings at 30 episodes), into `../exp5_smoke`, with placeholders for the values still undecided | `exp5 run <stage> --smoke --yes`, stage by stage, with `--set` placeholders for the RL decisions and pilot3's values | ~1–2 h |
| 4 | The code gate: the full suite against the recorded baselines | `python -m pytest tests …` then `tests/golden/make_baseline.py compare` | ~25 min |
| 5 | Batch 1, then `sens` (or after step 6; see the order choice) | `exp5 run batch1 --yes`, `exp5 run sens --yes` | ~7 h |
| 6 | The RL campaign. It pauses after the calibration (a sanity check to read) and after Study 5.5's verdict | `exp5 run rl --yes` | ~12–20 h |
| 7 | Act on the verdict: set `rl.gamma_star`, `rl.keep_learned` and the `rl.checkpoints` paths; build FX-dwell and FX-cov (if FX is kept) or M1 (if the learned score is kept) | params.toml; a build | decisions + build |
| 8 | Batch 2 | `exp5 run batch2 --yes` | ~50 h |
| 9 | Batch 3's pilots, then their outputs | `exp5 run pilot3 --yes`, `exp5 report pilot3 --apply`, the rest by hand | ~1 h |
| 10 | Batch 3 | `exp5 run batch3 --yes` | ~3 h |
| 11 | Score each batch, archive each stage's traces, commit; publish the archives (with your go-ahead) | `exp5 score batches`, `exp5 pack all` | ~1 h |

Machine time from step 1 to step 11 is about 3.5–4 days, plus the pauses for reading reports and deciding. Every stage runs on this machine: the session timeout was measured on its CPU, and arms are compared within one host ([guide §9](Experiment_5_Reproducibility_Guide.md#9-determinism-what-matches-across-machines)). Keep it awake and pause Windows Update for the long stages.

---

## Decisions still open

| Decision | Needed before | Options |
|---|---|---|
| **5.1's bound-derived merge weights.** `agg:cutoff`'s age weights are FedAsync's hinge with hand-set constants; the "derived from the bound" side is a theory-track derivation not yet done. 5.7 cites it too. | Batch 2 (5.1 and 5.7) | Drop the derivation and add a small sensitivity check on the hand-set constants (~240 trials, launcher settings only); drop it with nothing added; or do the derivation (research work), after which it becomes a weight mode and a 5.1 variant. |
| **Batch 1 before or after RL** | Step 5 | After RL, as the campaign orders it; or right after the re-pin and smoke test. See "Order choice" above. |
| **The verdict's follow-ups:** γ\*, keep the learned score or not, the checkpoints; FX-dwell/FX-cov or M1 | Batch 2 | Read from `exp5 report rl-sweep`. |
| **pilot3's hand-set values:** 5.15's harsher interference amplitude and higher path-loss exponent, 5.11 (c)'s knees | Batch 3 | Read from `exp5 report pilot3`. |
| **Publishing the trace archives** (a GitHub release, Zenodo for the paper's artifact) | The paper | Nothing is published without your go-ahead. |

Decided already, for reference: τ from the knee pilot (0.71; 0.82 kept as a second τ); 4 missions per trial; the jittery contact channel everywhere; N = 6's knee accepted at 150 s; E3's γ = 0.99 (Chen's code); FedProx ρ = 0.01; 5.6 at 40 trials per cell; 5.12's levels from the p512 pilot; traces in per-stage archives, outside git; `agg:seq` out.

---

## What could still go wrong

- **The smoke run has not happened.** Every job passes the runner's argument checks, but none past the pilots has run. Expect it to find a few runtime problems.
- **A pilot rule can come back empty.** p512 refuses if no training-time level gives 10–30% not-ready contacts (change `[p512] cycle_factors` and rerun). `repin.py` refuses if Study 5.6's quarter and half cells' ratio ranges would meet, which was tight at 180 s in the trial (1.48 against 1.5).
- **τ = 0.71 is set by the larger sizes.** At N = 18 and 24, accuracy levels off near 0.71 whatever the budget. At N = 6, where 80% of trials reach 0.82, most trials reach 0.71 early, so time to τ separates the arms less there. Final accuracy and reach at 0.82 are reported beside it.
- **The code gate** must be rerun after the re-pin; host-specific differences are listed in the [guide §2.7](Experiment_5_Reproducibility_Guide.md#27-the-code-gate-path-b-and-before-reporting-results).

# Experiment 5 — energy cost and the time budget: findings

*16 Oct 2026, after the 8 Oct deadline. Two questions came up while building the results deck: how much energy FeRRy (F, FX, FQ) spends against the seven baselines, and whether FeRRy wins only because the mission budget handicaps the baselines. Neither analysis is pre-registered. The 300 s budget is Study 5.3's pre-registered third level (dropped in the trim); the 1800 s budget is an added check. Status of every stage: [Experiment_5_Readiness.md](Experiment_5_Readiness.md).*

---

## Where the outputs are

- **Script:** `scripts/exp5/energy_budget.py` (commit `6531c422`), tests in `tests/unit/test_exp5_energy_budget.py`. It needs the kept traces on disk (energy comes from the mule traces).
- **Reports:** `results/exp5/scores/b2/exploratory_energy.{md,csv}` and `exploratory_budget.{md,csv}`. The pre-registered 5.3 scores (`s53x.*`) are untouched.
- **Runs:**
  - 300 s: `results/exp5/b2/s53x/n6k1_relaxed__*` (commit `5ca8f3de`).
  - 1800 s: `results/exp5_relaxed1800/b2/s53x/n6k1_relaxed__*` (commit `6531c422`), from the same command with `--out-root results/exp5_relaxed1800 --set batch2.relaxed_factor=12`.
  - Both: N = 6, one mule, all ten arms plus FedEx's merge, 20 trials on the knee cell's seeds, every row ok.
- **Slides:** `Presentation Documents/FeRRy_Draft3.pptx`: slide 13 (the FX rules), 21 (energy), 22–23 (budget).

## How energy is counted

The energy is the simulator's rotary-wing model, not a measurement. Every scheduler draws about the same power (124–132 W at N = 6, one mule), so energy follows mission time.
- **Energy to τ:** the energy of the missions that started before the model first reached τ = 0.71. The mission in flight when τ is reached counts in full; with three mules, every mule's missions count.
- **Energy per merged update:** a trial's total energy over its merged updates. The arms table reports all energy over all updates; the tests pair the per-trial values.
- **Energy per mission:** the scorer's `sim_energy_j_mean`.

Comparisons use the scorer's own test (bootstrap CI, paired Wilcoxon). Holm runs across each FeRRy arm's seven baselines per cell and metric; FQ against FX is its own family of three. A claim needs the CI to exclude 0 and Holm p < 0.05.

---

## Energy: FeRRy reaches the target on 22–33% less

**N = 6, one mule, knee budget (150 s):**

| Arm | Energy to τ (kJ) | Per merged update (kJ) | Per mission (kJ) | Transit (s) | Dwell (s) |
|---|---|---|---|---|---|
| F | 37.4 | 6.36 | 22.5 | 15 | 107 |
| FX | 33.7 | 5.44 | 19.3 | 15 | 88 |
| FQ | 34.5 | 5.74 | 20.3 | 15 | 94 |
| 7 baselines | 48.0–50.6 | 7.68–8.38 | 26.8–28.7 | 134–146 | 11 |

- **Every FeRRy arm uses less energy to τ than every baseline: 21 of 21 claims.** FX saves 30–33% against each baseline, FQ 28–32%, F 22–26%. The baseline used more in 15–19 of the 20 paired runs.
- **The same holds per merged update and per mission** (7 of 7 for each FeRRy arm).
- **Why:** FeRRy's narrow band lets one stop cover most devices, so it flies about 15 s per mission and hovers about 90 s; the baselines fly 134–146 s between short stops. Power is nearly equal, so the shorter mission is the saving.

**Stress budget (75 s):**
- **Energy to τ:** FeRRy 33.2–34.3 kJ against 38.8–43.1 kJ for the budget-respecting baselines and 48.0 kJ for FedEx's route. Only the FedEx-route gap is a claim (1 of 7 for each FeRRy arm).
- **Per merged update:** still 7 of 7. FeRRy spends 5.6–6.3 kJ against 7.7–10.0 kJ, because the cut-short baseline tours merge fewer updates (2.1–2.7 per mission against 3.2).

**Three mules (knee):** times to τ tie (F 52 s against 55–65 s, no claims in 5.3), but energy does not. F and FX need 33.2 and 32.9 kJ to τ against 39.1 kJ for six baselines (6 of 7 claims each; FedEx's route, 35.8 kJ, is not a claim). Per update and per mission: 7 of 7.

### FQ against FX

| Budget | Per mission | FQ cheaper in | Holm p | Verdict |
|---|---|---|---|---|
| Knee | +1.05 kJ (+5.5%) | 3 of 20 | 0.16 | no claim |
| Stress | +1.36 kJ (+7.5%) | 1 of 20 | 0.009 | FX |

- **Energy to τ:** no claim at either budget (+0.6 to +0.7 kJ).
- **Why FQ costs more:** FQ flies 6–8 s longer per mission for the same updates.
- **Why it was never pushed to save:** its training reward weighted energy at zero (c_e = 0).
- **Why FX saves by design:** its band rule picks the fastest band that reaches the same devices.

---

## The budget: a tight one erases FeRRy's lead; a loose one does not

Mean time to τ (s) over the trials that reached it, N = 6, one mule, 20 paired seeds:

| Arm | 75 s (stress) | 150 s (knee) | 300 s | 1800 s |
|---|---|---|---|---|
| F | 178 | 194 | 194 | 194 |
| FX | 175 | 178 | 178 | 178 |
| FQ | 182 | 180 | 179 | 179 |
| 6 baselines that obey the budget | 198–238 | 266–282 | 261–290 | 261–290 |
| FedEx route (ignores the budget) | 264 | 264 | 264 | 264 |
| F faster than (of 7) | 1 | 7 | 7 | 7 |

The stress and knee claims are 5.3's (`s53x.md`). The 300 s and 1800 s claims come from this script, with Holm across each cell's ten.
- **With room, FeRRy keeps its lead.** At 300 s nobody overruns. F is 67–96 s faster than every baseline, Holm p ≤ 0.002.
  - F's plan is the same as at the knee: all 20 times to τ are identical.
  - The baselines fly full wide-band tours and get no faster.
- **1800 s repeats 300 s.** Every arm except E3 gives the same time to τ in all 20 trials. E3, the only arm whose choices shift with the budget value, moves 0.2 s on average. **The budget stops mattering between 150 and 300 s.**
- **A tight budget erases the lead.** At 75 s the baselines' tours are cut short. Their missions shrink from about 225 s to about 175 s, and since τ = 0.71 needs only one or two rounds, they reach it sooner. F's lead falls to 20–61 s and only the FedEx-route gap (that route ignores the budget) stays a claim.
  - Same at N = 24: F beats H1 and the Whittle index at the knee (749 against 911–951 s) and ties them under stress (661 against 651–662 s).
- **Three mules also erase it.** Each mule covers about two devices, so flying is short for everyone and every scheduler takes 52–65 s.
- **The cost on FeRRy's side.** Under stress, 30% of F's missions overrun, by about 28 s. The baselines that obey the budget overrun in at most 9%. FeRRy keeps its long narrow-band hover, so signal dips push it past the budget.

**So:** FeRRy's advantage is the band choice, not the budget. It pays off when one mule covers a sparse field and the budget leaves room for a full mission. It shrinks to a tie when time is very tight or mules are plentiful.

---

## Caveats

- **Energy is simulated** (a rotary-wing model), not measured.
- **Neither analysis is pre-registered.**
  - The 300 s cell is a pre-registered cell, but it was flown after the deadline and scored as its own family.
  - The 1800 s cell is an added check. FQ is exploratory throughout.
- **Do not quote "FX faster than F" from the relaxed cells.** Each relaxed family marks it as a claim (16 s, Holm p 0.023). That repeats the knee data (identical times) in a family of ten; in 5.3's study-wide family it is not a claim.
- **No truly unbudgeted run was flown.** The launcher always passes a number; with no budget at all, the feasibility test would switch off and FeRRy's plan could choose differently.
- **The energy-to-τ figures need the traces.** The script reads them from disk; they are not in git (`launch.py pack` / `unpack`).

# Experiment 5 — loss probes: where FeRRy could lose

*Draft, 29 Oct 2026. **Status, 5 Nov 2026:** P1, P4 and P9 have run (scorer `72b8e59c`, results `3cffc966`), and so have P2 (results `35e75e41`) and P5 (results `654e095c`: F flies medium at 2 MB and wide from 5 MB, no baseline beats it on time or reach, and its lead is no claim from 2 MB up); reports in `results/exp5/scores/probes/`. The other probes have not. None of it is in Exp 5's pre-registered analysis plan: every probe is exploratory, added after the CARE room's sweep (21 Oct) and Exp 5's own near-misses. This page is the probes' analysis plan; change it before a probe flies, never after. Status of the pre-registered stages: [Experiment_5_Readiness.md](Experiment_5_Readiness.md).*

---

## Why

Exp 5 tests FeRRy's design claims. It was not built to find where FeRRy loses, and its cells all sit where FeRRy wins:

- **One field.** Every stack cell flies the fixed ±100 m square (`campaign.field_ref_n = 0`) at every N. The narrow band reaches 3.87 × 60 m ≈ 232 m and the square's corner is 141 m from the dock, so one narrow hover at the dock can reach every device. That is the mechanism behind F's N = 6 win (15 s of transit against the baselines' 134–146 s).
- **Partial baselines.** The radio study (5.15) flies only F, FX and H1; the scale study (5.9) has no MAX-AoI, Oort or FedCS; the band-pinning study (5.4) compares F only with itself pinned to each band.

### What the CARE room measured (21 Oct 2026)

Eleven scenarios, flown through the room's bridge (`scripts/ferry_bridge.py` in CARE-Sit-Rep-Cave): Ferry's own stack in process, Exp 5's run settings, the real model, T_nom priced on the scenario's field, on the room's node positions. F against MAX-AoI, Oort, Whittle, FedEx and FedCS; two layouts, one seed each, 132 runs. The margin is the best baseline's time to 71 % over F's (geometric mean of the two layouts; above 1, F first). Sources: the room's `Docs/checklists/FERRY_MODE_CHECKLIST.md` ("Real Training; … presets where Ferry wins and where it loses") and `Assets/_Project/Scripts/Ferry/FerryScenarios.cs`. The 132 runs themselves were not kept.

| Scenario | Margin | What the room saw |
|---|---|---|
| Near/far: 6 nodes, half beyond wide's 60 m reach, 150 s | 1.50 | F flies medium or narrow and reaches the far nodes from fewer hovers |
| Ferry's N = 6 cell, 150 s | 1.41 | F narrow, one stop a round |
| Ferry's N = 12 cell, 180 s | 1.26 | Exp 5's 20 trials had FedEx level with F (459 against 486 s): trust Exp 5 |
| N = 6 cell at 75 s | 1.15 | MAX-AoI tied F on layout 1 |
| Mid cluster: 3 groups of 4, 150 m out, 180 m across | 1.10 | F first on all five layouts measured |
| 12 nodes over ±300 m | 1.00 | tie with FedEx |
| Ferry's N = 24 cell, 262 s | 0.99 | tie with FedCS |
| Tight clusters: 3 groups, 150 m out, 60 m across | 0.93 | one wide hover covers a group; F flies medium; Whittle first |
| Long cluster: 3 groups, 250 m out, 180 m across | 0.91 | F flies narrow, one hover a group, and pays for the slow link; FedEx first on all five layouts |
| Long stress: 24 nodes over ±300 m, 10 MB, 600 s | 0.55 | F flies wide, keeps to the limit, leaves far nodes for later rounds; with no limit it ties FedEx |
| Harsh radio: N = 6 cell, 17 dB interference | 0.30 | F keeps its long narrow hover; every baseline reaches 71 % in 65–89 s, F in 169 and 342 s |

Also from the room:
- **Payload:** on the N = 6 cell at 2 MB F flies medium; at 5 MB, wide with FedEx's stops.
- **A far field (15 Oct):** three nodes 0.7–1.0 km out, 210 m reach: F 44:02 against FedCS 11:42 and FedEx 11:43; two of F's four rounds collected nothing. Measured before the 20 Oct fix that prices T_nom on the room's field; not re-measured.

**How far to trust it.** The bridge reproduces an Exp 5 trial exactly (trial 0: F 282.953 s, FedEx 375.655 s), so the path is faithful. Two layouts and one seed are a demonstration, not a study, as the room's own note says. The room's 52 s for F on the N = 6 cell is inside Exp 5's spread there (F's fastest of 40 trials 29 s, lower quartile 98 s).

### Exp 5's own near-misses (20 trials, no claim)

- With several mules, FedEx's route is ahead of F at N/K = 12/1 (459 against 486 s), 12/2 (222 against 226 s) and 12/3 (128 against 151 s), and H1 at 18/3 (149 against 186 s). Holm p = 1.0.
- Under stress at 4 missions, FedEx's route keeps updates fresher than F (age 0.72 against 0.78 missions; CI excludes 0, Holm p = 0.21).
- Under stress F's lead falls from 7 of 7 claims to 1 of 7, and F overruns 30 % of its missions (by 28 s on average) against at most 9 % for the baselines that keep to the budget.
- With training time, F reaches τ in 70–75 % of trials against 85–100 % for H1, D4 and D5; its time-to-τ wins count only the trials that reached τ.

The remaining pre-registered runs cover the mule and freshness items (the scale study's K = 2 and 3 cells, the fairness study's 8-mission horizon). The probes below cover the rest.

---

## Rules every probe follows

- **Its own family.** Each probe is scored on its own, with Holm across its comparisons, and reported as exploratory. Nothing is folded into the pre-registered score files.
- **What counts as a loss.** The reference is F; each baseline is compared with F in each cell. Primary metric: time to τ = 0.71, paired by seed over the trials where both reach it. Reach at τ is tested too (exact McNemar on the paired reached / not-reached outcomes). F loses a cell when a baseline is significantly faster or reaches τ significantly more often (CI excludes 0, Holm p < 0.05). FX is reported beside F, in its own family.
- **Trials.** 20 paired trials per cell, on the campaign's base seed. A trial's seed is sha256(base seed | cell | trial), where the cell is the grid axes (N, range, missions, regime); budget, payload and radio flags are not in it. So a new arm pairs with a recorded one in the same N cell, whatever its out-root.
- **Where a probe writes.** A probe that only adds arms, or cells with tags of their own, writes into `results/exp5` beside the recorded references, as the FQ-against-E3 check did (`scripts/exp5/fq_vs_e3.py`). A probe that changes a setting the job tag doesn't name (a budget factor, the payload, the backhaul, the field, the layout) gets its own `--out-root`, or the resume guard refuses it.
- **Keep out-root names short.** Long paths are off on this host, so a kept trace's deepest path must stay under 250 characters; `launch.py plan` prints it. `results/exp5_probe_harsh_missionbh` made 5.15's traces 254 characters, so P1 writes to `results/exp5_p1mbh` (238).
- **Never run `launch.py score` with a probe's `--set`s.** It would overwrite the official scores. Score with the probe scorer (B0 below).
- **Hours** come from the per-job wall times logged in batches 1–3 (per trial: 56 s at N = 6, 50 s at N = 12, 62 s at N = 24), packed on the 45 training processes of this host. Nothing at N = 24 with 10 MB, on a bigger field or on a clustered layout has been timed; treat those hours as rough.
- **Commands** are given without `--yes`, so the launcher lists the jobs and asks first.

---

## The probes

| # | Probe | Tests | Build | Trials | ~h |
|---|---|---|---|---|---|
| **Tier 1** | | | | | |
| P1 | Harsh radio, every baseline | the room's widest deficit (0.30) | B0 | 1,040 | 2.5 |
| P2 | Budgets tighter than stress | F's lead fading as the budget tightens | B0 | 400 | 1.0 |
| P3 | MAX-AoI, Oort and FedCS at scale | the room's FedCS tie at N = 24 | B0 | 240 | 1.6 |
| P4 | Baselines in the far-device and short-reach cells | the room's widest lead (1.50), and a narrower narrow band | B0 | 360 | 0.9 |
| P5 | Bigger uploads at the N = 6 cell | the narrow band's slow link | B0 | 480 | 1.2 |
| **Tier 2** | | | | | |
| P6 | Fields beyond the narrow band's reach | Long stress (0.55), 12 nodes over ±300 m (1.00) | B0, B1 | 1,360 | 6.3 |
| **Tier 3** | | | | | |
| P7 | Clustered layouts | Long cluster (0.91), Tight clusters (0.93), Mid cluster (1.10) | B0, B2 | 1,200 | 4.8 |
| P8 | A hard budget | F's 30 % overrun under stress | B0, B3 | 300 | 1.4 |
| P9 | Analysis on recorded data | reach as a tested metric; τ = 0.82 | B0 | 0 | — |
| | **All** | | | **5,380** | **~20** |

### P1 — Harsh radio, every baseline

**Question.** Does every baseline beat F when the interference swings 17 dB, as the room found? Where between the jittery 5 dB and 17 dB does the order flip?

**Cells.** N = 6, one mule, knee (150 s), 1 MB.
- On 5.15's seconds backhaul (its own cells): the default (5 dB), a new 11 dB level, 17 dB, path-loss exponent 2.7 and shadowing 8 dB. MAX-AoI, Oort, Whittle, FedEx and FedCS (D1–D5) in all five cells (500 trials); F, FX and H1 at 11 dB (60). The other cells' F, FX and H1 are recorded.
- On the mission backhaul, as the room flew it: the default, 17 dB and path-loss 2.7, with F, FX, H1 and D1–D5 (480 trials). The default cell repeats batch 1's knee cells on the same seeds; flights are deterministic (the 300 s and 1800 s cells gave identical times), so it must match them, which is a check.

**Expected (the room).** On the mission backhaul at 17 dB, every baseline 65–89 s against F's 169 and 342 s. Exp 5 at 17 dB on the seconds backhaul: F 255 s (95 % reach), FX 209 s, H1 332 s.

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s515 --set "s515.arms=['F','FX','H1','D1','D2','D3','D4','D5']" --set "s515.amps_db=[11.0]" --set "s515.mission_backhaul_arms=[]"
```

```bash
py -3.11 scripts/exp5/launch.py run batch3 --study s515 --out-root results/exp5_p1mbh --set "s515.backhaul='mission'" --set "s515.arms=['F','FX','H1','D1','D2','D3','D4','D5']" --set "s515.amps_db=[]" --set "s515.shadow_sigmas_db=[]" --set "s515.mission_backhaul_arms=[]"
```

### P2 — Budgets tighter than stress

**Question.** Below the stress budget, does a baseline overtake F?

**Cells.** N = 6, one mule, 1 MB, at 60 s and 45 s (0.4 and 0.3 × the 150 s knee). In the knee pilot H1 served 28 % and 24 % of devices there, against 58 % at the knee. Every 5.3 arm: F, FX, H1, D1–D5, E3 and FedEx with its merge (10 jobs a budget, 400 trials).

**Expected.** F's lead was 7 of 7 claims at 150 s and 1 of 7 at 75 s. The room's 75 s cell had MAX-AoI level with F on one layout.

The budget rides on 5.3's "relaxed" level (a multiple of the knee), so each factor needs its own out-root; the job tags will read `relaxed`.

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x --out-root results/exp5_probe_tight40 --trials 20 --set batch2.relaxed_factor=0.4 --set "s53x.budgets=['relaxed']" --set "s53x.K=[1]" --set s53x.h0=false
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x --out-root results/exp5_probe_tight30 --trials 20 --set batch2.relaxed_factor=0.3 --set "s53x.budgets=['relaxed']" --set "s53x.K=[1]" --set s53x.h0=false
```

### P3 — MAX-AoI, Oort and FedCS at scale

**Question.** The scale study flies F, FX, H1, Whittle and FedEx. Do the other three baselines catch F as the field fills?

**Cells.** N = 12 and 24, one mule, knee and stress, jittery: D1, D2 and D5 (240 trials). They pair with the recorded F, FX, H1, D3 and D4 cells.

**Expected (the room).** At the N = 24 cell FedCS tied F (541 and 496 s against 527 and 519 s).

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s59x --set "s59x.arms=['D1','D2','D5']" --set "s59x.N=[12,24]" --set "s59x.K=[1]" --set "s59x.contact_regimes=['jittery']"
```

### P4 — Baselines in the far-device and short-reach cells

**Question.** Does the room's widest lead hold over 20 trials, against every baseline? And when the narrow band reaches less far, does F lose its edge?

**Cells.** 5.4's 1 MB knee cells.
- **Far shares** 25, 50 and 75 % (devices beyond wide's 60 m reach): H1 and D1–D5 in each (360 trials; 480 if the base cell doesn't share batch 1's CSVs). F's far cells are 5.4's pre-registered runs, still to fly; the command below flies them too, and they count toward 5.4.
- **Narrow range ratios** 2 and 3: at ratio 2 the narrow band reaches 120 m, short of the square's 141 m corner, so one hover no longer covers the field. The ratio changes only the narrow class and the baselines fly wide, so their recorded knee cells already are their ratio cells (same seeds). Only F flies here, in 5.4's own pre-registered run; the probe compares it with those recorded cells.

**Expected (the room).** Near/far at 50 %: F 29 and 58 s against the best baseline's 48 and 79 s.

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s54 --set "s54.arms=['F','H1','D1','D2','D3','D4','D5']" --set "s54.payloads=[1000000]" --set "s54.narrow_range_ratios=[]" --set s54.o1=false
```

### P5 — Bigger uploads at the N = 6 cell

**Question.** The narrow band is slow. At what payload does reach stop paying for it?

**Cells.** N = 6, one mule, knee (150 s), at 2, 5 and 10 MB: F, FX, H1, D1–D5 (480 trials). The knee was measured at 1 MB, so at 10 MB some arms may serve no one in 150 s; reach shows that. The 10 MB cell overlaps 5.12's remaining 10 MB cell with no training time (F, FX, H1, D4, D5).

**Expected (the room).** F flies medium at 2 MB, and wide with FedEx's stops at 5 MB.

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s53x --out-root results/exp5_probe_2mb --trials 20 --set s53x.payload_bytes=2000000 --set "s53x.budgets=['knee']" --set "s53x.K=[1]" --set s53x.h0=false --set s53x.d4_faithful=false --arms F FX H1 D1 D2 D3 D4 D5
```

`--arms` selects by the runner's arm, and FedEx with its merge flies as arm D4, so `d4_faithful=false` is what leaves it out.

The same at 5 MB (`results/exp5_probe_5mb`, `s53x.payload_bytes=5000000`) and 10 MB (`results/exp5_probe_10mb`, `10000000`).

### P6 — Fields beyond the narrow band's reach

**Question.** On a field the narrow band can't cover from the dock, with a big model and a budget, do the baselines beat F? Is it the budget (the room: with no limit F ties FedEx)?

**Needs B1** (the field's half-width as a launcher setting).

**Pilot.** The knee at N = 12 on ±200 m and ±300 m: H1 on wide, 1 MB, budgets 180, 300, 450, 600, 900 and 1200 s, 20 trials each, by the knee rule (240 trials). Then S\* at each field.

**Cells.**
- N = 12, one mule, on ±200 m and ±300 m: 1 MB at the field's knee, 10 MB at the knee, and 10 MB at 12 × the knee (no limit in effect, as the 1800 s check). F, FX, H1, D1–D5 (960 trials).
- The room's own cell: N = 24, ±300 m, 10 MB, 600 s, the same eight arms (160 trials).

**Expected (the room).** Long stress: F 3184 and 3215 s against FedCS 1692 s and Oort 1835 s; with no limit F ties FedEx. 12 nodes over ±300 m: a tie.

Each (field, payload) pair gets its own out-root (the job tag names neither). Sketch after B1, for one pair:

```bash
py -3.11 scripts/exp5/launch.py run knee --out-root results/exp5_probe_w300_1mb --set campaign.field_half_width_m=300 --set "knee.cells=[[12,1000000]]" --set "knee.budgets_s.12=[180,300,450,600,900,1200]"
```

```bash
py -3.11 scripts/exp5/launch.py run batch2 --study s59x --out-root results/exp5_probe_w300_1mb --set campaign.field_half_width_m=300 --set pilot_outputs.knee_s.12=KNEE --set "s59x.arms=['F','FX','H1','D1','D2','D3','D4','D5']" --set "s59x.N=[12]" --set "s59x.K=[1]" --set "s59x.contact_regimes=['jittery']" --set "s59x.budgets=['knee']"
```

`KNEE` is the pilot's knee for that field. The 10 MB pairs add `s59x.payload_bytes=10000000`, and their no-limit cell `s59x.budgets=['knee','relaxed']` with `batch2.relaxed_factor=12`.

**Also.** Before this probe, re-fly the room's 15 Oct far-field race through the bridge, now that T_nom is priced on the room's field: a few minutes. If F still loses about 4×, add a sparse far cell here.

### P7 — Clustered layouts

**Question.** When devices sit in groups, wide's single hover can cover a group and reach buys nothing. Do the baselines win there, and is it the groups' size or their distance?

**Needs B2** (a clustered layout).

**Cells.** N = 12, one mule, 1 MB, three groups of four. Geometries (distance from the dock, group radius): tight (150 m, 30 m), mid (150, 90), long (250, 90), and a new long-tight (250, 30) that separates distance from group size.
- All four with no limit in effect (1800 s, as the room flew them): F, FX, H1, D1–D5 (640 trials).
- Tight and long at their own knee: an H1 pilot as in P6 (240 trials), then the eight arms (320).

**Expected (the room).** Long cluster: FedEx first on all five layouts (340–370 s against F's 388–501 s). Tight: Whittle 183 and 188 s against F's 207 and 191 s. Mid: F first on all five.

### P8 — A hard budget

**Question.** Overruns are allowed today and only counted. If a mission had to end at its budget, would F's 30 % overrun turn into lost updates, and who would lose most?

**Needs B3** (a hard budget).

**Cells.** Stress budget, one mule, 1 MB: N = 6 with every 5.3 arm (200 trials); N = 24 with F, FX, H1, D3 and D4 (100).

**Expected.** FedEx's route overruns 85 % of stress missions and has no gate, so it likely loses most; F's lead under stress may shrink further.

### P9 — Analysis on recorded data

No flights.
- **Reach as a tested metric:** the paired reach test on every recorded cell. The cells to watch are 5.12's, where F reaches τ in 70–75 % of trials against 85–100 % for H1, D4 and D5. (In 5.13's Dirichlet 0.1 cell every arm reaches it in only 50–55 %.)
- **τ = 0.82:** a rescore where it's reachable. In the knee pilot, H1 reached 0.82 in 18 of 20 trials at N = 6, at most 4 of 20 at N = 12, and none at N = 18 or 24.

---

## Builds

| | Build | What it is | Size |
|---|---|---|---|
| B0 | Probe scorer | `scripts/exp5/loss_probes.py`, after `fq_vs_e3.py` and `energy_budget.py`. It runs the scorer on each probe's CSVs and pairs them with the reference CSVs by trial, across out-roots. It applies Holm per probe, adds the reach test, and writes `results/exp5/scores/probes/<probe>.{md,csv}`. Unit tests. | small |
| B1 | The field's half-width | A `campaign.field_half_width_m` setting that `_runner` passes as `--h1-field-radius-m` (the runner switch exists, and the driver draws T_nom's reference layouts on the same field). Recorded in the manifest; a test. Check that the S\* tool takes the field; if not, S stays 2 and the probe says so. | small |
| B2 | A clustered layout | The room's cluster draw ported to `experiments/exp4/topology_builder.device_positions`: G group centres at distance d from the dock at even angles, the first due north, and each device uniform in a disc of radius r round its group's centre, dealt round the groups in turn, from the trial's seed. Runner flags; the driver draws T_nom's reference layouts the same way (the room found that pricing T_nom on the wrong field changed F's band choice); launcher settings; goldens unchanged at the default. Tests. | medium |
| B3 | A hard budget | A sim-only switch, default off (Freeze Rule 1). At the budget's end the mule stops its contact and flies home; a transfer in progress is lost. Adds a drop reason and a CSV column for the updates lost. Tests. | medium |

---

## Order and time

| Step | What | ~h |
|---|---|---|
| 1 | B0, then P9 | 0 machine |
| 2 | Tier 1: P1, P2, P3, P4 (with 5.4's F cells), P5; read the results | 7.2 |
| 3 | The room's far-field re-check | minutes |
| 4 | B1, P6's pilots, then P6 | 6.3 |
| 5 | B2, then P7 | 4.8 |
| 6 | B3, then P8 | 1.4 |

The probes share the one host with the remaining pre-registered runs (about 45 h), so they queue behind or between them. Only P4 depends on them, for 5.4's F cells, which P4's command flies anyway.

---

## Decisions for you

1. **The paper.** Report the probes in the revision (due 18 Jan 2027) as exploratory, motivated by the room's sweep, or keep them for a follow-up paper. Either way, the conclusion's limitations would then cite measured losses.
2. **P6 and P7's budgets.** Each field's pilot knee plus no limit, as drafted; or the room's budgets only, which saves the 480 pilot trials (about 1.9 h) but leaves F without a knee on those fields.
3. **P8's rule.** Cut at the budget, as drafted, or keep a reserve for the flight home.
4. **More arms.** E3 and FedEx with its merge fly only in P2; adding them elsewhere costs about 20 % more.
5. **Where arm-only probes write.** Beside the recorded references in `results/exp5`, as drafted (P1's first command, P3, P4); or every probe in its own out-root, re-flying the references there, which costs more but keeps the campaign's folders as they were.

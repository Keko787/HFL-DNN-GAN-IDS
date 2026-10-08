# Methods audit: the paper against the code (8 Oct 2026)

**What this is:** a review of whether the methodology section
(`methods.tex`), with the setup section (`setup.tex`), describes everything
significant in the code base and the evaluated configuration.

**How it was done:**
- Three code reviews ran in parallel, one per area: the radio and transport
  layers, the scheduler and planner, and aggregation, the processes and the
  simulators.
- Each review checked its findings against the code. I re-checked the
  high-severity ones myself.
- The baseline is code at `1f529c8` on `main`.
- The **evaluated configuration** is `COMMON_FLAGS` and `_runner` in
  `scripts/exp5/launch.py`, with `scripts/exp5/params.toml`.

**Verdict:** the core is faithful. The plan clock is right: demand, per-class
clustering, the three search modes and their thresholds, the plan score and
its constants, the plan key, the guard fold and the age cap. So are every L1
equation and constant of the main configuration and the merge formulas
(hinge, cutoff, minimum participation of one).

The holes were around that core:
- FX was described with FQ's machinery.
- Two studies flew an undescribed backhaul.
- FerrySim and the three-mule staleness were understated.
- The baselines, the field and the accuracy target's floor were missing.
- Several rules were stated wrongly.

**Status key:**
- **fixed:** the paper now says it.
- **appendix:** the detail belongs in the reproducibility appendix (IPDPS
  requires one on acceptance) or the artifact. Details are given below so the
  appendix can be written from this file.
- **no change:** checked and matches.

The fixes are in `methods.tex`, `setup.tex`, `evaluation.tex` and
`scripts/exp5/paper_tables.py` (two captions). They make the paper longer: about
11,800 counted words of prose, against about 10,600 before.

## A. Statements the code contradicted (all fixed)

| # | Finding | Code evidence | Fix in the paper |
|---|---|---|---|
| A1 | **FX was described with FQ's machinery.** The text said both fillings share the arrival-time mask, fallback and scope check and "differ only in how they rank the admitted pairs". In fact FX picks its band on arrival (fastest covering class) and its next stop at the following departure, after the departure check, from the realized clock (nearest stop whose move to the front passes the fold, else the plan's next). The mask, fallback and scope check exist only for FQ, which picks band and next stop together on arrival from the priced dwell. | `hermes/mule/mule_main.py:1968-2010`, `:2048-2078`; `hermes/scheduler/policies/cross_heuristic.py:221-274`; `hermes/scheduler/policies/pair_slot.py:49-56` ("none of them the fixed arm of its name"), `:762-839` | Methods: flight-slot section rewritten (F, FX, FQ paragraphs); Fig. `fig:clocks` caption; Algorithm 1's band and next-stop lines per arm. Results: a sentence in `sec:res:rl` that FX and FQ do not decide on the same information |
| A2 | **Study 5.14's backhaul test (secF vs secFL1) and every Study 5.15 cell flew a time-varying backhaul**, not the stated 2% flat loss. SNR_c(t) = 6 + g_c + 5 sin(2π(t/P_bh + φ_c)) + 1.5 ξ_c(t) dB, with P_bh = 4 T_nom; loss = 1/(1+e^{(SNR−3)/2}). F closes 79% of rounds on it (secF; 5.15 cells 72–84%), against 97% in the main configuration. 5.15's "default" cell is therefore not the headline's default: F takes 216 s there, against 194 s. | `scripts/exp5/launch.py:205-206`, `:1164-1166`; `params.toml` `[s514]` comment, `[s515] backhaul = "seconds"`; `hermes/l1/channel_model.py:157-165` (loss), `:779-902` (BackhaulChannel); `results/exp5/scores/b1/s514_arms.csv`, `b3/s515_arms.csv` | Methods: the "Backhaul" paragraph defines both models. Setup: Q4 and Q6 say which studies fly it. Results: robustness text and the robust table's caption note 5.15's default |
| A3 | **The adaptive backhaul controller (F+L1) is close to an oracle.** It picks the carrier from the realized per-carrier SNR at the upload instant, the same SNR that decides the loss. U = log2(1+SNR) above 0 dB (else 0), κ = 0, λ = 0.5 in rate units; switching costs no time. The stated formula had γ1(t)+g(c) and no constants. | `hermes/l1/channel_utility.py:28-85`; `channel_model.py:886-902`; `hermes/mule/ferry.py:1294-1335` | Methods: formula and constants corrected, with the oracle note. Results: the claims paragraph qualifies "0.88 to 0.69". Claims-table note |
| A4 | **FerrySim differs from the full system in more than "two settings":** one mule only; scale cells grow the field (half-widths 200, 282.8 and 400 m at N = 24, 48, 96) with budgets from FerrySim's own pilot; stub device (13-parameter θ plus N(0, 0.01) noise, n_i = 10, zero fit time); 3 s session timeout. | `experiments/ferrysim/inprocess.py:334-369`, `:839-840`; `experiments/ferrysim/cells.py:77-92`, `:339-357` | Methods: FerrySim paragraph lists them. Setup: the field sentence points to it |
| A5 | **Algorithm 1's "pull ⟨M, θ, v⟩ from the server" at the dock does not happen.** A mule flies Pass 1 with θ staged at its previous inter-pass dock. With quorum 1 each upload closes its own round, so with K = 3 the other mules' merges make partials stale: 160 of 198 applied merges at n6k3_knee (F, 20 trials) had partial age 1–3 (s_m = 1/2 to 1/4), and the server applies no cutoff. | `hermes/mule/mule_main.py:1611-1615`, `:1818-1826`; `hermes/cluster/host_cluster.py:464-473`; `hermes/mission/aggregation_rules.py:498-505`; traces in `results/exp5/b1/s53` | Methods: Algorithm 1's first line; "Merge across mules" adds the staleness; the relation paragraph says merge age zero is the mule's |
| A6 | **The re-plan does reorder.** The trim flies every stop holding a capped member first, then the rest in plan order. | `hermes/scheduler/fl_scheduler.py:1808-1916`; `plan/member_subset.py:604-738`; `stages/s3d_age_cap.py:326-339` (`priority_first`) | Methods: departure-check paragraph |
| A7 | **A stop's own deadline is the earliest among its uncapped members**, not all members (a stop with none is exempt). | `stages/s3d_age_cap.py:278-300` (`stop_deadline`); `plan/plan_search.py:453-456` | Methods: age-cap paragraph and the feasibility test's deadline clause |
| A8 | **Which outcomes widen by β_partial:** any timeout after the device answered (failed push, lost reply, no update ready) widens by β_partial = 1.25. β_timeout = 1.5 applies only to silent, out-of-range or dropped (synthetic) devices. | `stages/s3_deadline.py:271-295`, docstring `:118-150`; `hermes/mission/host_mission.py:1586-1626` | Methods: deadline paragraph |
| A9 | **A device whose reply is lost still receives the push**, at full airtime S, and adopts the pushed model. The 1 s listen is charged once per stop with any missing reply, not per device. | `host_mission.py:1316-1343`, `:1525-1541`; `hermes/mission/client_mission.py:430-447` | Methods: contact-outcomes paragraph |
| A10 | **The budget is soft.** It is checked on mean-SNR predictions at commit and at each departure while stops remain; nothing re-checks after the last Pass-1 stop, so realized missions overrun (30% of F's under the stress budget). The text said the mule "must land … within T_miss". | `mule_main.py:1976`, `:2712-2713`; `cross_heuristic.py:50-52` | Methods: Setting paragraph and the end of the feasibility-test section |
| A11 | **Pass 2 can skip a device:** one below the decoding floor on arrival is not delivered to (2 of 474 in the FX knee cell). The text said "no skipping". | `hermes/mule/ferry.py` `contact_plan` (≈1121-1170); `hermes/mission/contact_plan.py:340-372` | Methods: Delivery paragraph |
| A12 | **FerrySim's reward:** the shortfall counts only devices the plan committed to and left uncollected; a sortie that flew no Pass-1 stop is not charged; training credits a targeted device at its reliability ρ_j instead of the realized draw. | `experiments/ferrysim/reward.py:1-63`, `:375-383`; `experiments/ferrysim/train.py:170-174` | Methods: FQ paragraph |

## B. Omissions added to the paper (fixed)

| # | Finding | Code evidence | Where it is now |
|---|---|---|---|
| B1 | **How the baselines fly.** They share S1, S3a clustering at the wide range, the S3b pricing and the unbudgeted nearest-first Pass 2. They serve whole stops only, have no age cap, and use the additive law (−5σ s after a clean contact, +10σ s after a miss, floor 5σ s, no ceiling).<br>**H1:** EDF admission under deadline and budget; flown by bucket (NEW first), then distance from the dock; trim in flight.<br>**D1–D3:** greedy budget walk in rank order, budget-only.<br>**D5:** FedCS greedy by least marginal time.<br>**D1–D3 and D5 in flight:** budget-only departure check, then re-admission by their own rule.<br>**D4:** visit-all, no budget gate, CARP split at K = 3.<br>**E3:** admits all stops, picks the next by Q under a budget-only safety mask, no re-plan; trained at K = 1 (lr 5e-4, γ 0.99), flown per mule at K = 3. | `experiments/exp4/driver.py:516-561`, `:1156-1169`; `fl_scheduler.py:942-1031`, `:1097-1170`, `:1311-1339`; `policies/budget_walk.py`, `fedcs_degraded.py`, `fedex_carp.py`, `chen_dqn.py:1-50`; `params.toml` `[rl.e3]` | Setup: "How the baselines fly" paragraph after the arms table |
| B2 | **The field:** static devices drawn uniformly on a 200×200 m square centred on the dock (new layout per trial), the same square at every N (`field_ref_n = 0`), positions pre-seeded in the registry and used by the planner, slices fixed for the trial. | `experiments/exp4/topology_builder.py:85-145`; `params.toml` `field_ref_n`; `hermes/processes/cluster.py:717-760`; `host_cluster.py:661-669` | Methods: Setting paragraph. Setup: Cells paragraph |
| B3 | **The accuracy target is near the majority-class rate.** The test set is 4,000 benign + 2,000 attack (`reduce_attack_samples`), so all-benign scores 0.667 against τ = 0.71. H0's final 0.686 is barely above it. | `experiments/exp4/model_task.py:609-629`; `Config/DatasetConfig/CICIOT2023_Sampling/ciciot2023DatasetLoadV2.py:245-251` | Methods: learning task. Setup: τ sentence. Results: H0 sentence |
| B4 | **The radio only prices time.** Each session's rate is read once at its start. Sessions at a stop run in sequence. A started session always completes, at the floor rate if the SNR has fallen below it. The TCP exchange is lossless loopback (no channel emulator configured). | `host_mission.py:1484-1530`; `contact_plan.py:414-436`; `hermes/processes/mule.py:608`, `device.py:171`; `hermes/transport/tcp_rf_link.py:163` | Methods: "Rate and dwell" |
| B5 | **A lost upload** (main configuration): Bernoulli draw from a per-mule seeded stream, no retry. The server returns the unchanged θ, Pass 2 re-delivers it, and the planner, which ran `record_merged` before the upload, counts the devices as merged. | `hermes/processes/cluster.py:913-935`, `:1357-1375`; `mule_main.py:1774-1783`; `hermes/mule/client_cluster.py:410-440` | Methods: "Backhaul" paragraph |
| B6 | **Wall-clock timeout expiries become simulated timeouts.** This is a host-load dependence; the sensitivity stage at 0.75, 1 and 1.5 times the timeout gave identical results. | `host_mission.py:1163`, `:1198`, `:1765-1770`; `results/exp5/scores/sens` | Methods: contact-outcomes paragraph |
| B7 | **In-session fits:** a device with no prepared update (first contact, or after a missed delivery) fits inside the Pass-1 contact at zero simulated cost. Train-ahead needs `pass_2_budget`, which plan mode refuses. | `client_mission.py:477-520`; `mule_main.py:722`; `hermes/processes/config.py:903` | Methods: learning task |
| B8 | **Time to τ is stamped at the simulated completion of the closing upload,** before Pass 2 delivers the model. | `host_cluster.py:247-280`; `experiments/.../traces_scorer.py:692-760` | Methods: learning task |
| B9 | **Scope-outs:** integrity checks only (unkeyed SHA-256; `verify_up_bundle` never called), no authentication, robust aggregation or privacy. Unused code: ChannelDDQN, S3c mission-window adaptation, the legacy S3.5 learned selector, the beacon path, the S2A/S2B readiness gates. Flight is planar at fixed altitude with no takeoff/landing time or energy; energy is propulsion only. | `hermes/types/signatures.py:1-12`; `client_cluster.py:402`, `:503-509`; `hermes/processes/mule.py:715-724`; `driver.py:677`; `hermes/l1/mission_clock.py:96-112`, `:355-370` | Methods: Implementation-and-scope and Mission-clock paragraphs |

## C. Deferred to the reproducibility appendix

These are needed to reproduce the system but not to follow the paper.

**Radio constants**
- κ_b = 0.754 / 0.734 / 0.732 for the three classes (`hermes/l1/contact_link.py:399-401`).
- CQI SNR thresholds from −6.7 to 22.7 dB, from the AERPAW digital twin (Hossen et al. 2025, arXiv:2503.07935), paired with TS 36.213 Table 7.2.3-1 efficiencies (`contact_link.py:322-338`).
- The carrier is 3.32 GHz, and d³ᴰ is floored at 1 m.
- The 60 m wide-band anchor is an assumed link budget, implying an EIRP of about −11.3 dBm (`contact_link.py:117-145`).

**Channel generation**
- Shadowing is keyed by time, not position: it varies while the mule hovers.
- It is built by linear interpolation of independent normals on 7.4 s bins, so the correlation is 0.29 at 7.4 s and 0 beyond 14.8 s. It is not exponential.
- A position-keyed option exists but is unused (`channel_model.py:265-275`, `:395-448`, `:694-710`).

**Interference**
- φ_b is a per-trial seeded permutation of {0, 1/3, 2/3} across classes.
- ξ_b uses 1 s bins.
- t runs from the trial's first takeoff, shared by all mules and arms (`channel_model.py:501-529`, `:712-723`).

**Main-configuration upload timing**
- The upload is timed at 6 dB + max g_c on the wide class's CQI table: about 0.35 s for 1 MB (`ferry.py:526-536`, `:889-895`).

**Planner details**
- **S3a anchor order:** −delivery_priority, then bucket (NEW first), then earliest deadline, with ties broken by slice order (`stages/s3a_cluster.py:77-152`).
- **Re-clustering:** clustering is redone each mission on the current deadlines. delivery_priority rises after a Pass-2 miss (`hermes/cluster/device_registry.py:106-119`).
- **NEW devices:** a NEW device keeps the NEW bucket for up to 3 misses. ι(j) = 0 before its first clean contact (`s3_deadline.py:76`, `:356-366`, `:427-465`).
- **Member order for greedy reduction:** capped members first, then w_j/dwell_j descending, then device id (`plan/member_subset.py:236-276`).
- **Local search:** up to three first-improvement scans, the first ranked by V and the later ones by the plan key, sharing the 50-pass and 2000-evaluation bounds. The start tour is cheapest insertion plus 2-OPT, as a fixed-end path back to the dock (`plan/plan_search.py:64-117`, `:643-801`; `routing/two_opt.py`).
- **Hover-stop point:** on the dock–device segment, within R_b and at or above the floor. It minimizes the device's alone mission and is found by a grid plus refinement (`plan/hover.py:1-80`).

**FQ: features and training**
- **Features:** 36 columns (`selector/pair_features.py:22-57`, `:246-284`):
  - travel, deadline slack (log-scaled), exempt, capped and home flags, plan age / S, on-time rate;
  - member share, clock left against the budget, energy left;
  - remainder and weight shares, least slack;
  - current and previous per-class offsets, with the previous reading's age and phase.
- **Learner:** Huber loss, Adam lr 1e-3, gradient clip 10, hard target copy every 500 updates, batch 64, replay 50k, warm-up 1000, one update per decision (`selector/pair_q.py:19-47`, `:353-500`).
- **Behaviour policy:** ε-greedy around FX's pair, 0.3 for the first 500 episodes, then 0.3 to 0.05 over the first half.
- **Training schedule:** 10,000 episodes, validation every 1,000 on 400 episodes, early stop after 3 without improvement (`experiments/ferrysim/train.py:176-180`; `params.toml:280-282`).
- **Cells (jittery56):** N = 6 at 75/150 s, N = 12 at 90/180 s, plus four N = 12 cells with P_c ∈ {54, 68, 108, 136} s (`experiments/ferrysim/cells.py:274-370`).
- **Reward:** G_k = Σw/(n_ref·N), with the time cost scaled by T_nom (`reward.py`).
- **The flown checkpoint:** γ = 0.75, seed 3.

**E3's training**
- lr 5e-4, ε from 1.0, γ 0.99, 5 seeds; the median-validation seed is flown (`params.toml` `[rl.e3]`).

**Data and model**
- **Data draw:** the trial seed picks 3 training CSV parts and 1 test part out of 169. Rows are sampled within files with a fixed random_state = 47, so two files usually fill the quota and the attack mix follows them (`model_task.py:604-607`; `ciciot2023DatasetLoadV2.py:136-208`).
- **Scaling:** one MinMax scaler is fit on the pooled 20,000 training rows before partitioning, so test values can exceed 1 (`datasetPreprocess.py:146-153`). Central fitting is a simplification FL would not allow.
- **Cleaning:** no NaN or duplicate removal.
- **Seeds:** the initial θ seed is 12345 in every trial and arm (`driver.py:615`). The device training seed is a hash of the device id (`hermes/processes/device.py:151-158`).
- **Model state:**
  - each device keeps its Keras model and Adam state across fits (`model_task.py:239-281`);
  - deltas cover all weights, including BatchNorm moving statistics, which are therefore averaged (`client_mission.py:514-538`).

**Wire protocol**
- Pickled frames with a magic, version and length header, capped at 256 MiB (`hermes/transport/wire.py`).
- An update is checked for round, byte count, SHA-256, update form and wall receipt age (`host_mission.py:1734-1770`).

**Empty missions**
- With K = 1 an empty Pass 1 does not dock or fly Pass 2, re-flies the same θ next mission, and still counts as one of the four missions (`mule_main.py:1740-1770`).
- With K > 1 an empty Pass 1 still docks and is logged as an expired merge (`driver.py:1631-1636`).
- All 3,780 trial rows in b1–b3 are status ok.

**Server feedback**
- The DOWN bundle carries each device's delivery_priority, raised after a failed Pass-2 delivery and reported one mission late (`host_cluster.py:720-729`).

**Dock timing**
- `dock_wait` (Lamport sync) is 0 s in every evaluated K = 3 run (`mission_clock.py`).

## D. Checked and matching (no change)

- **L1 constants:**
  - bandwidths and N_RB;
  - planar ranges 60 / 119.5 / 232.2 m;
  - M_sh = 5.13 dB, floor −6.7 dB;
  - σ_sh = 4 dB, 7.4 s;
  - A = 5 dB, σ_I = 1.5 dB, P_c = 60 s;
  - the p_out σ_eff;
  - 5 m/s, 143.6 / 168.5 W, 30 s turnaround, 1 s listen, 2% loss.
- **Search:** modes and thresholds (exact ≤ 6 demanded devices, stop-subsets ≤ 6 stops, local otherwise); 50 passes and 2000 evaluations; at most 1957 sequences per class.
- **Plan score V:** form and constants (c1 = 1, c2 = c3 = κN with κ = 1, c4 = 0.1).
- **Plan key:** matches, with unstated tie-breaks on stop identity after −V and the class index.
- **Guard fold and drops:** the guard fold, plan-drop labelling and synthetic timeouts.
- **Age cap and coverage weight:** the plan age, the coverage weight max(a, 1)(1 + μ), and the cap at a ≥ S − L.
- **Multi-mule slicing:** angular sectors after the widest empty arc, sizes within one (`topology_builder.py:171-209`). D4's CARP split is the exception, now in B1.
- **Merges:** Eqs. `eq:mergeweight`, `eq:mulemerge` and `eq:clustermerge`, the FedAsync hinge, the deadline-derived cutoff, and the minimum participation of one.
- **The RF layer learns nothing:** "nothing in the RF layer is learned" holds; ChannelDDQN is never wired to a mule.

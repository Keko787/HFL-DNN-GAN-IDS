# Joint RL Methods — document set

*7 Oct 2026. Start with the overview, [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) (one page per joint optimisation, by layer). The nine documents below are the dedicated detail, one per component or decision. Each was written against the code and the records; claims it could not verify are marked "unverified" in the text, and each ends with the discrepancies it found between older documents and the code.*

| # | Document | What it covers | Lines |
|---|---|---|---|
| 01 | [RF band selection](01_RF_Band_Selection.md) | Contact band classes, plan-time band class b̄, per-arrival band, backhaul controller U(c, t), retired ChannelDDQN, channel model | 401 |
| 02 | [Job / target selection](02_Job_Target_Selection.md) | S1–S3a pipeline, deadline laws, the S3.5 selector, age-cap admission, baselines D1–D5, the 2 × 2 | 328 |
| 03 | [Feasibility gate and mission clock](03_Feasibility_Gate_and_Mission_Clock.md) | S3b predicate, simulated clock, departure check and re-plan, scope guard, TTL and knee pilots, the three cuts | 321 |
| 04 | [Age and staleness weight](04_Age_Staleness_Weight.md) | Age definitions, merge weight and cutoff, aggregation rules, FedProx, age cap S\*, Network AoU | 264 |
| 05 | [Objective and reward](05_Objective_and_Reward.md) | Plan score V, flight reward, expected-availability credit, Study 5.7 | 258 |
| 06 | [Route and plan search](06_Route_and_Plan_Search.md) | Candidate family, plan key, search modes, member subsets, hover, 2-OPT, oracle O1, decision cost | 374 |
| 07 | [Flight pair decision and learner](07_Flight_Pair_Decision_and_Learner.md) | The (band, next stop) decision, FX / FQ / E3, pair_v1 features, double DQN, FerrySim, Study 5.5 | 455 |
| 08 | [Decision register](08_Decision_Register.md) | 39 runtime decisions and 54 hand-set constants, who or what decides each | 315 |
| 09 | [Work record](09_Work_Record.md) | Chronology, phase ledger, Freeze amendments, studies ledger, defects, decisions, what remains | 312 |

**Reading order by question**

- *What does the system decide, and what is learned?* → 08, then 07.
- *Why is the learning claim a null?* → 07, then 05.
- *How does the radio layer enter?* → 01, then 03.
- *What do the layers exchange?* → 03, 04, 05.
- *What has been done and what is left?* → 09.

**Findings that cut across the documents** (each is sourced inside the document named)

1. **One learned runtime component.** The in-flight pair score FQ (07, 08). In the arms Exp 5 calls FeRRy (F, FX) the register tallies 27 fixed, 9 heuristic, 2 search and 0 learned decisions. Learned code that nothing runs: E3 (a competitor), the H2 selector, `ChannelDDQN`, `select_server`.
2. **The learned score matched FX.** It trailed `greedy_1`; looking ahead added nothing (07). That the learner copied FX is a hypothesis, untested (07).
3. **F's lead is not shown to come from any single mechanism.** At N = 6 H1 and D1 tie, with three mules H1 to D3 tie, D4 ties F at N = 12 and 24, and most 5.14 ablations are inert at N = 6. Study 5.4 (reach) has not run (02, 06, 09).
4. **"One objective" is a shared design idea, not one quantity.** The merge weight falls with age and the plan score's coverage weight rises with it (04, 05).
5. **Gates that never run.** The S2A/S2B readiness gates and the cluster's deadline overrides have no live effect (08).
6. **Stale older records.** The build plan, Learned Pair Score note, Matrix Results and Reproducibility Guide §11 each disagree with the code in places. The list is in section 9 of document 09 and the closing section of each document. None of those records was edited.

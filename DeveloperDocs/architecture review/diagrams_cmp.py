"""Diagrams for the CEDA / HERMES comparative analysis."""
from __future__ import annotations

from typing import List, Sequence

from diagrams import _t, arrow, band, box, figure, label, Row

SM = "t-sm"


# --------------------------------------------------------------------------- #
# D7 — how much does the policy decide?
# --------------------------------------------------------------------------- #

def d7_policy_authority() -> str:
    P = "d7"
    s: List[str] = []
    COLW = 306
    X = [22, 356, 690]
    TOP, PANEL_H = 54, 348

    heads = [
        ("CEDA", "monolithic CTDE DQN", "bad-band"),
        ("hermes_rl", "hybrid heuristic + DQN", "band"),
        ("HERMES", "gated stages + tie-break DQN", "ok-band"),
    ]
    for i, (name, sub, cls) in enumerate(heads):
        s.append(band(X[i], TOP, COLW, PANEL_H, "", cls=f"band {cls}"))
        s.append(_t(X[i] + COLW / 2, TOP + 26, name, "", anchor="middle"))
        s.append(_t(X[i] + COLW / 2, TOP + 40, sub, "t-lbl", anchor="middle"))

    def inner(x: float, y: float, rows: Sequence[Row], h: float, cls: str) -> str:
        return box(x + 16, y, COLW - 32, h, rows, cls=cls, lh=14)

    # --- CEDA ---
    s.append(_t(X[0] + COLW / 2, TOP + 68, "LEARNED", "t-hd", anchor="middle"))
    s.append(inner(X[0], TOP + 78, [
        "one DQN · 140-dim obs",
        ("routing to patients", SM),
        ("task assignment", SM),
        ("navigation + collision", SM),
        ("hazard avoidance", SM),
        ("energy management", SM),
        ("triage ordering", SM),
    ], 140, "bx hl"))
    s.append(_t(X[0] + COLW / 2, TOP + 246, "DETERMINISTIC", "t-hd", anchor="middle"))
    s.append(inner(X[0], TOP + 256, [("— nothing —", SM)], 30, "bx"))
    s.append(label(X[0] + COLW / 2, TOP + 312,
                   ["six reward terms carry every",
                    "constraint simultaneously"]))

    # --- hermes_rl ---
    s.append(_t(X[1] + COLW / 2, TOP + 68, "LEARNED", "t-hd", anchor="middle"))
    s.append(inner(X[1], TOP + 78, [
        "DQN · 9 actions",
        ("which base station", SM),
        ("which channel", SM),
    ], 66, "bx hl"))
    s.append(_t(X[1] + COLW / 2, TOP + 172, "DETERMINISTIC", "t-hd", anchor="middle"))
    s.append(inner(X[1], TOP + 182, [
        "heuristic navigator",
        ("which waypoint next", SM),
        ("collect / transit rules", SM),
    ], 66, "bx"))
    s.append(label(X[1] + COLW / 2, TOP + 280,
                   ["action space cut 45 to 9;",
                    "replay seeded with expert",
                    "trajectories from episode 1"]))

    # --- HERMES ---
    s.append(_t(X[2] + COLW / 2, TOP + 68, "LEARNED", "t-hd", anchor="middle"))
    s.append(inner(X[2], TOP + 78, [
        "DDQN rank · 11 features",
        ("order within ONE bucket", SM),
        ("only when 2+ candidates", SM),
    ], 66, "bx hl"))
    s.append(_t(X[2] + COLW / 2, TOP + 172, "DETERMINISTIC", "t-hd", anchor="middle"))
    s.append(inner(X[2], TOP + 182, [
        "S1 to S2A/S2B to S3 to S3a",
        ("eligibility · readiness", SM),
        ("deadline · bucket · cluster", SM),
    ], 66, "bx"))
    s.append(label(X[2] + COLW / 2, TOP + 280,
                   ["a scope guard raises if the",
                    "selector ever sees a candidate",
                    "the gates did not admit"]))

    # authority axis
    ay = 436
    s.append(arrow([(X[0] + 40, ay), (X[2] + COLW - 40, ay)], f"{P}-a"))
    s.append(_t(X[0] + 40, ay - 9, "policy decides everything", "t-lbl",
                anchor="start"))
    s.append(_t(X[2] + COLW - 40, ay - 9, "policy decides one ranking", "t-lbl",
                anchor="end"))
    s.append(_t((X[0] + X[2] + COLW) / 2, ay + 19,
                "less to learn · more to verify · easier to explain a decision",
                "t-lbl", anchor="middle"))

    return figure(
        "".join(s), view="0 0 1018 466", prefix=P,
        aria=("Three systems compared by how much authority is given to the learned "
              "policy. CEDA puts routing, assignment, navigation, hazard avoidance, "
              "energy and triage ordering inside one DQN with a 140-dimensional "
              "observation, with nothing deterministic outside it. hermes_rl splits "
              "the problem: a heuristic chooses the next waypoint while a DQN "
              "chooses base station and channel, cutting the action space from 45 to "
              "9. HERMES runs four deterministic scheduler stages and gives its DDQN "
              "only the ordering within a single bucket, and only when at least two "
              "candidates remain."),
        caption=("<b>The same lab, three answers to one question.</b> All three "
                 "schedule a UAV against time-varying channels and deadline-bound "
                 "tasks. They differ almost entirely in how much of the decision is "
                 "delegated to the network — which is the axis worth arguing about, "
                 "because it decides what can be verified, ablated, and explained "
                 "after the fact."),
    )


# --------------------------------------------------------------------------- #
# D8 — CEDA's cross-layer ablation, mapped onto HERMES
# --------------------------------------------------------------------------- #

def d8_ablation() -> str:
    P = "d8"
    s: List[str] = []
    X0, BARX, BARW = 24, 236, 300
    SCALE = 62.0
    rows = [
        ("Full CEDA", 1.40, "baseline", "bx hl"),
        ("minus network layer", 1.50, "HERMES L1 ChannelDDQN", "bx"),
        ("minus wind / physical", 1.55, "HERMES S3a + RF range", "bx"),
        ("minus patient timers", 3.10, "HERMES S3 deadline math", "bx warnbx"),
        ("minus triage weights", 3.75, "HERMES S3 bucket priority", "bx warnbx"),
        ("minus battery state", 4.10, "HERMES mule_energy feature", "bx badbx"),
    ]

    s.append(_t(X0, 26, "CROSS-LAYER INFORMATION ABLATION", "t-hd", anchor="start"))
    s.append(_t(X0, 44,
                "W3 Critical patients unserved per episode — lower is better",
                "t-lbl", anchor="start"))
    s.append(_t(BARX + BARW + 44, 44, "NEAREST HERMES ANALOGUE", "t-hd",
                anchor="start"))

    y = 62
    for name, val, mapping, cls in rows:
        s.append(_t(X0, y + 15, name, "t-sm", anchor="start"))
        w = val * SCALE
        s.append(f'<rect x="{BARX}" y="{y + 3}" width="{w:.0f}" height="18" '
                 f'rx="4" class="{cls}"/>')
        s.append(_t(BARX + w + 8, y + 16, f"{val:.2f}", "t-sm", anchor="start"))
        s.append(_t(BARX + BARW + 44, y + 16, mapping, "t-lbl", anchor="start"))
        y += 30

    s.append(f'<line x1="{BARX}" y1="{y + 2}" x2="{BARX + 4.4 * SCALE}" '
             f'y2="{y + 2}" class="ln"/>')
    for tick in (0, 1, 2, 3, 4):
        tx = BARX + tick * SCALE
        s.append(f'<line x1="{tx}" y1="{y + 2}" x2="{tx}" y2="{y + 7}" class="ln"/>')
        s.append(_t(tx, y + 19, str(tick), "t-lbl"))

    s.append(label(X0, y + 46,
                   ["Removing the network layer costs about 10 percent triage "
                    "efficiency. Removing battery state nearly triples "
                    "critical-patient mortality."],
                   anchor="start"))

    return figure(
        "".join(s), view="0 0 1000 302", prefix=P,
        aria=("Bar chart of CEDA's cross-layer information ablation, measured as W3 "
              "critical patients left unserved per episode. Full CEDA scores 1.40. "
              "Removing the network layer gives 1.50 and removing the wind or "
              "physical layer 1.55, both close to baseline. Removing patient timers "
              "gives 3.10 and removing triage weights 3.75. Removing battery state "
              "is worst at 4.10. Each row is annotated with its nearest HERMES "
              "equivalent."),
        caption=("<b>The layer everyone builds first matters least.</b> CEDA's "
                 "ablation ranks the network layer as the <i>least</i> load-bearing "
                 "of the three and energy state as the most. HERMES has never run "
                 "the equivalent experiment — its A1–A4 ablation varies the "
                 "<i>policy</i>, not the <i>information available to it</i>, so it "
                 "cannot produce this ranking."),
    )

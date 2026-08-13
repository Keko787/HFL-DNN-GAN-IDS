"""Hand-authored inline SVG replacements for the Mermaid blocks.

Self-contained: no scripts, no external images, no <style> inside the SVG.
All colour comes from page-level CSS classes bound to the house theme
variables, so light and dark both work with no per-diagram overrides.

Layout is verified analytically by check_svg.py (monospace advance widths),
which catches clipping, text collisions, and text straddling box borders.
"""
from __future__ import annotations

from html import escape
from typing import List, Optional, Sequence, Tuple, Union

Row = Union[str, Tuple[str, str]]   # text, or (text, css-class)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def _t(x: float, y: float, s: str, cls: str = "", anchor: str = "middle") -> str:
    a = f' text-anchor="{anchor}"' if anchor else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x}" y="{y}"{a}{c}>{escape(s)}</text>'


def _split(row: Row, default_cls: str) -> Tuple[str, str]:
    if isinstance(row, tuple):
        return row[0], row[1]
    return row, default_cls


def lines(x: float, y: float, rows: Sequence[Row], *, cls: str = "",
          lh: float = 13.0, anchor: str = "middle") -> str:
    out = []
    for i, row in enumerate(rows):
        text, rcls = _split(row, cls)
        out.append(_t(x, y + i * lh, text, rcls, anchor))
    return "".join(out)


def box(x: float, y: float, w: float, h: float, rows: Sequence[Row], *,
        cls: str = "bx", tcls: str = "", rx: float = 7, lh: float = 13.0) -> str:
    """Rounded rect with vertically centred text rows."""
    n = len(rows)
    y0 = y + h / 2 - (n - 1) * lh / 2 + 4
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'
            + lines(x + w / 2, y0, rows, cls=tcls, lh=lh))


def band(x: float, y: float, w: float, h: float, label: str, *,
         cls: str = "band", rx: float = 10) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'
            + _t(x + 12, y + 17, label, "t-hd", anchor="start"))


def arrow(pts: Sequence[Tuple[float, float]], marker: str, *, cls: str = "ln") -> str:
    d = " ".join(f"{px},{py}" for px, py in pts)
    return f'<polyline points="{d}" class="{cls}" marker-end="url(#{marker})"/>'


def label(x: float, y: float, rows: Sequence[Row], *, anchor: str = "middle",
          cls: str = "t-lbl", lh: float = 11.5) -> str:
    return lines(x, y, rows, cls=cls, lh=lh, anchor=anchor)


def defs(prefix: str) -> str:
    def m(name: str, cls: str) -> str:
        return (f'<marker id="{prefix}-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" class="{cls}"/></marker>')
    return ("<defs>" + m("a", "mk") + m("bad", "mk-bad")
            + m("dead", "mk-dead") + m("soft", "mk-soft") + "</defs>")


def figure(svg_body: str, *, view: str, aria: str, caption: str,
           prefix: str, extra_class: str = "") -> str:
    cls = ("dg " + extra_class).strip()
    return (
        '<figure class="dgfig">'
        f'<div class="dgscroll"><svg class="{cls}" viewBox="{view}" role="img" '
        f'aria-label="{escape(aria)}" xmlns="http://www.w3.org/2000/svg">'
        f'{defs(prefix)}{svg_body}</svg></div>'
        f'<figcaption>{caption}</figcaption>'
        '</figure>'
    )


# --------------------------------------------------------------------------- #
# D1 — layer model and dependency direction
# --------------------------------------------------------------------------- #

def d1_layers() -> str:
    P = "d1"
    s: List[str] = []
    LX, LW = 46, 518          # hermes column  46..564
    GX = 564                  # gutter         564..660
    RX, RW = 660, 336         # legacy column  660..996
    SM = "t-sm"

    # ---- hermes bands ----
    s.append(band(LX, 60, LW, 74, "L4 · EXPERIMENT / ANALYSIS"))
    s.append(box(LX + 14, 86, 152, 38,
                 ["experiments/", ("exp1 · exp3 · exp4", SM)], lh=12))
    s.append(box(LX + 180, 86, 152, 38,
                 ["experiments/runner", ("grid · csv_log", SM)], lh=12))
    s.append(box(LX + 346, 86, 158, 38,
                 ["experiments/analysis", ("figures · stats", SM)], lh=12))

    s.append(band(LX, 166, LW, 74, "L3 · PROCESS / ORCHESTRATION"))
    s.append(box(LX + 14, 192, 152, 38,
                 ["MultiProcess", ("Orchestrator", SM)], lh=12))
    s.append(box(LX + 180, 192, 152, 38,
                 ["processes.cluster", ("processes.mule", SM)], lh=12))
    s.append(box(LX + 346, 192, 158, 38, ["processes.device"], lh=12))

    s.append(band(LX, 272, LW, 96, "L2 · PROGRAM LAYER — 7 HERMES PROGRAMS",
                  cls="band ok-band"))
    prog = ["HFLHostCluster", "MuleSupervisor", "FLScheduler", "HFLHostMission",
            "ClientCluster", "ClientMission", "ChannelDDQN"]
    for i, name in enumerate(prog):
        col, row = i % 4, i // 4
        s.append(box(LX + 14 + col * 124, 298 + row * 32, 114, 26,
                     [(name, SM)], lh=12))

    s.append(band(LX, 400, LW, 78, "L1 · TRANSPORT + TYPES + OBSERVABILITY",
                  cls="band ok-band"))
    s.append(box(LX + 14, 426, 186, 40,
                 ["hermes.transport", ("RFLink / DockLink ABCs", SM)], lh=12))
    s.append(box(LX + 214, 426, 146, 40,
                 ["hermes.types", ("bundles · messages", SM)], lh=12))
    s.append(box(LX + 374, 426, 130, 40,
                 ["hermes.", ("observability", SM)], lh=12))

    # ---- legacy column ----
    s.append(band(RX, 60, RW, 418, "HIFINS LEGACY STACK", cls="band bad-band"))
    s.append(box(RX + 18, 90, 300, 38,
                 [("App/TrainingApp · App/InferenceApp", SM)], lh=12))
    s.append(box(RX + 18, 162, 300, 44,
                 ["Config/SessionConfig", ("argparse → 41-arg dispatch", SM)], lh=12))
    s.append(box(RX + 18, 246, 300, 46,
                 ["Config/ModelTrainingConfig",
                  ("47 classes / 45 files · 21.3K LOC", SM)], lh=12))
    s.append(box(RX + 18, 328, 142, 40,
                 ["Config/", ("DatasetConfig", SM)], lh=12))
    s.append(box(RX + 176, 328, 142, 40,
                 ["Config/", ("modelStructures", SM)], lh=12))
    s.append(box(RX + 18, 404, 300, 38, [("ModelArchive/ · Analysis/", SM)], lh=12))
    for y0, y1 in ((128, 158), (206, 242)):
        s.append(arrow([(RX + 168, y0), (RX + 168, y1)], f"{P}-a"))
    s.append(arrow([(RX + 100, 292), (RX + 100, 324)], f"{P}-a"))
    s.append(arrow([(RX + 250, 292), (RX + 250, 324)], f"{P}-a"))

    # ---- healthy downward dependencies ----
    s.append(arrow([(LX + 160, 134), (LX + 160, 162)], f"{P}-a"))
    s.append(arrow([(LX + 160, 240), (LX + 160, 268)], f"{P}-a"))
    s.append(arrow([(LX + 160, 368), (LX + 160, 396)], f"{P}-a"))
    s.append(label(LX + 170, 388, ["depends on ABCs only"], anchor="start"))

    # L4 bypasses L3 straight into L2 (exp3 drives the scheduler in-process)
    s.append(arrow([(LX + 8, 100), (LX - 20, 100), (LX - 20, 322), (LX + 8, 322)],
                   f"{P}-a"))
    s.append(label(LX - 20, 208, ["drives"]))

    # ---- bad edge 1: inverted import (runs up the gutter) ----
    s.append(arrow([(GX + 30, 196), (GX + 30, 112)], f"{P}-bad", cls="ln-bad"))
    s.append(label(GX + 46, 146, ["INVERTED", "IMPORT"], anchor="start", cls="t-bad"))
    s.append(label(LX + 330, 152,
                   ["hermes.processes imports experiments.exp4 — cluster.py:113"],
                   anchor="middle"))

    # ---- bad edge 2: the dead --mode hermes bridge (routed under both columns) ----
    s.append(arrow([(RX + 14, 109), (GX + 72, 109), (GX + 72, 520),
                    (LX + 400, 520), (LX + 400, 372)], f"{P}-dead", cls="ln-dead"))
    s.append(label(LX + 300, 512, ["--mode hermes · STUB, never executes"]))
    s.append(_t(LX + 300, 536, "✕", "t-x"))

    # ---- orphan ----
    s.append(box(RX, 508, 300, 46,
                 ["FlightFramework/ — 5,853 LOC",
                  ("0 importers · documented as reused", SM)],
                 cls="bx dead", lh=13))

    # ---- legend ----
    ly = 578
    s.append(f'<line x1="{LX}" y1="{ly}" x2="{LX + 34}" y2="{ly}" class="ln"/>')
    s.append(_t(LX + 42, ly + 4, "import / depends on", "t-lbl", anchor="start"))
    s.append(f'<line x1="{LX + 190}" y1="{ly}" x2="{LX + 224}" y2="{ly}" class="ln-bad"/>')
    s.append(_t(LX + 232, ly + 4, "inverted dependency", "t-lbl", anchor="start"))
    s.append(f'<line x1="{LX + 380}" y1="{ly}" x2="{LX + 414}" y2="{ly}" class="ln-dead"/>')
    s.append(_t(LX + 422, ly + 4, "declared but dead", "t-lbl", anchor="start"))

    return figure(
        "".join(s), view="0 0 1010 600", prefix=P,
        aria=("Layer diagram. Dependencies flow downward from the experiment layer "
              "through process orchestration, the seven HERMES programs, and the "
              "transport and types layer. Two edges break the rule: hermes.processes "
              "imports the experiments package, and the legacy App stack's "
              "--mode hermes branch points at the program layer but never executes. "
              "FlightFramework sits unconnected with zero importers."),
        caption=("<b>Dependencies flow downward — except twice.</b> The HERMES column "
                 "(green) is acyclic and leaf-clean. The red edge is "
                 "<code>hermes.processes</code> importing <code>experiments.exp4</code>; "
                 "the grey edge is the <code>--mode hermes</code> bridge that never "
                 "executes. <code>FlightFramework/</code> has no edges at all."),
    )


# --------------------------------------------------------------------------- #
# D2 — mission sequence
# --------------------------------------------------------------------------- #

def d2_sequence() -> str:
    P = "d2"
    s: List[str] = []
    cols = [
        (104, "Orchestrator", "host"),
        (330, "ClusterService", "Tier 2"),
        (600, "MuleSupervisor", "Tier 2-mobile"),
        (838, "DeviceService", "Tier 1"),
    ]
    TOP, BOT = 46, 964

    def msg(y: float, a: int, b: int, text: str, *, sub: str = "") -> str:
        x1, x2 = cols[a][0], cols[b][0]
        fwd = x2 > x1
        out = arrow([(x1 + (6 if fwd else -6), y), (x2 - (6 if fwd else -6), y)],
                    f"{P}-a")
        mid = (x1 + x2) / 2
        out += _t(mid, y - 7, text, "t-sm")
        if sub:
            out += _t(mid, y + 12, sub, "t-lbl")
        return out

    def selfmsg(y: float, a: int, rows: Sequence[str], *, left: bool = False) -> str:
        """Self-call loop with its note beside the lifeline.

        ``left`` places the note to the left — needed on the last column so
        the text cannot run past the viewBox.
        """
        x = cols[a][0]
        sign = -1 if left else 1
        out = arrow([(x + 6 * sign, y - 8), (x + 46 * sign, y - 8),
                     (x + 46 * sign, y + 8), (x + 10 * sign, y + 8)],
                    f"{P}-soft", cls="ln-soft")
        tx = x + (54 * sign)
        anchor = "end" if left else "start"
        y0 = y - 4 - (5 if len(rows) > 1 else 0)
        out += lines(tx, y0, rows, cls="t-lbl", anchor=anchor, lh=11.5)
        return out

    # phase bands
    s.append(band(20, 62, 936, 172, "BOOTSTRAP"))
    s.append(band(20, 246, 936, 274, "PASS 1 · COLLECT", cls="band ok-band"))
    s.append(band(20, 532, 936, 148, "INTER-PASS DOCK"))
    s.append(band(20, 692, 936, 250, "PASS 2 · DELIVER", cls="band ok-band"))

    # lifelines + heads
    for x, name, tier in cols:
        s.append(f'<rect x="{x - 74}" y="10" width="148" height="34" rx="7" class="bx"/>')
        s.append(_t(x, 25, name, "", anchor="middle"))
        s.append(_t(x, 37, tier, "t-lbl", anchor="middle"))
        s.append(f'<line x1="{x}" y1="{TOP}" x2="{x}" y2="{BOT}" class="life"/>')

    # bootstrap
    s.append(msg(92, 0, 1, "spawn --config --port-out",
                 sub="parent reads back the bound dock port"))
    s.append(msg(128, 0, 2, "spawn (dock_port)"))
    s.append(msg(154, 0, 3, "spawn (mule rf_port)"))
    s.append(msg(186, 3, 2, "TCP connect + registration"))
    s.append(msg(210, 2, 1, "TCP connect + registration"))
    s.append(msg(230, 1, 2, "DownBundle #0 — slice, θ_disc, synth"))

    # pass 1
    s.append(selfmsg(276, 2, ["BundleDistributor fans out: slice + amendment",
                              "→ FLScheduler · θ + synth → staged"]))
    s.append(selfmsg(320, 2, ["build_contact_queue(rf_range_m)",
                              "S1 eligibility → S3 bucket + deadline →",
                              "S3a cluster → S3.5 DDQN rank"]))
    s.append('<rect x="392" y="352" width="540" height="122" rx="8" class="loopbx"/>')
    s.append(_t(404, 366, "loop  per contact waypoint", "t-hd", anchor="start"))
    s.append(msg(392, 2, 3, "FLOpenSolicit(COLLECT) — broadcast"))
    s.append(msg(416, 3, 2, "FLReadyAdv(state, utility)"))
    s.append(msg(440, 2, 3, "DiscPush(θ_disc, synth)"))
    s.append(msg(464, 3, 2, "GradientSubmission(Δθ, n, checksum)"))
    s.append(selfmsg(496, 2, ["verify: round · byte_count · checksum · TTL",
                              "→ RoundCloseDelta onto the scheduler bus"]))

    # dock
    s.append(selfmsg(556, 2, ["close_round() → partial_fedavg(Σ wᵢ Δθᵢ)"]))
    s.append(msg(588, 2, 1, "UpBundle(aggregate, round-close + delivery reports)"))
    s.append(selfmsg(616, 1, ["ingest → cross_mule_fedavg → close_cluster_round"]))
    s.append(msg(654, 1, 2, "DownBundle(θ_disc′, synth′, enriched amendment)"))

    # pass 2
    s.append(selfmsg(718, 2, ["build_pass_2_queue — entire slice,",
                              "nearest-first, selector bypassed"]))
    s.append('<rect x="392" y="756" width="540" height="80" rx="8" class="loopbx"/>')
    s.append(_t(404, 770, "loop  per contact waypoint", "t-hd", anchor="start"))
    s.append(msg(796, 2, 3, "FLOpenSolicit(DELIVER) + DiscPush(θ′)"))
    s.append(msg(820, 3, 2, "DeliveryAck"))
    s.append(selfmsg(862, 3, ["set θ basis → train_offline() →",
                              "Δθ prepared for the NEXT Pass 1"], left=True))
    s.append(selfmsg(912, 2, ["close_pass_2() → MissionDeliveryReport,",
                              "stashed for the NEXT mission's UP bundle"]))

    return figure(
        "".join(s), view="0 0 976 986", prefix=P,
        aria=("Sequence diagram of one HERMES mission. The orchestrator spawns "
              "cluster, mule and device processes and they register over TCP. In "
              "Pass 1 the mule broadcasts a collect solicitation per contact, "
              "receives readiness adverts, pushes the discriminator weights, and "
              "collects gradient submissions which it verifies and aggregates with "
              "partial FedAvg. At the inter-pass dock it uploads the aggregate and "
              "receives freshly cross-mule-averaged weights. In Pass 2 it delivers "
              "those weights to every device in the slice, each acknowledging and "
              "immediately starting offline training for the next mission."),
        caption=("<b>One mission, end to end.</b> The two-pass structure is the "
                 "point: every Δθ collected in Pass 1 was trained against the θ "
                 "delivered by the <i>previous</i> mission's Pass 2, which makes "
                 "cross-mule FedAvg exact rather than approximate — async-FL drift "
                 "becomes structurally impossible."),
    )


# --------------------------------------------------------------------------- #
# D3 — target architecture
# --------------------------------------------------------------------------- #

def d3_target() -> str:
    P = "d3"
    s: List[str] = []
    SM = "t-sm"
    s.append(band(24, 48, 700, 300,
                  "hifins/ — ONE INSTALLABLE DISTRIBUTION (pyproject.toml)"))

    s.append(box(48, 84, 274, 66,
                 ["hermes/", ("unchanged public API", SM),
                  ("+ transport & persistence hardening", SM)], lh=12.5))
    s.append(box(48, 178, 274, 66,
                 ["hifins_models/", ("Keras structures + ONE", SM),
                  ("parameterised trainer family", SM)], lh=12.5))
    s.append(box(48, 272, 274, 60,
                 ["hifins_data/", ("loaders · preprocessing", SM),
                  ("· repo-anchored paths", SM)], lh=12.5))
    s.append(box(422, 84, 278, 66,
                 ["hermes_adapters/", ("KerasGeneratorHost : GeneratorHost", SM),
                  ("KerasLocalTrain : LocalTrainFn", SM)], cls="hl", lh=12.5))
    s.append(box(422, 178, 278, 66,
                 ["hifins_config/", ("frozen dataclasses,", SM),
                  ("not 41-arg positional tuples", SM)], lh=12.5))
    s.append(box(422, 272, 278, 60,
                 ["experiments/", ("depends on the package —", SM),
                  ("nothing depends on it", SM)], lh=12.5))

    s.append(arrow([(418, 117), (326, 117)], f"{P}-a"))
    s.append(label(372, 110, ["implements"]))
    s.append(arrow([(561, 174), (561, 154)], f"{P}-a"))
    s.append(arrow([(418, 205), (326, 205)], f"{P}-a"))
    s.append(label(372, 198, ["configures"]))
    s.append(arrow([(561, 268), (561, 248)], f"{P}-a"))
    s.append(arrow([(326, 300), (418, 300)], f"{P}-a"))
    s.append(arrow([(185, 154), (185, 174)], f"{P}-a"))
    s.append(arrow([(185, 248), (185, 268)], f"{P}-a"))

    return figure(
        "".join(s), view="0 0 748 372", prefix=P,
        aria=("Target architecture. One installable distribution holds hermes "
              "unchanged, a consolidated model and data layer, a frozen-dataclass "
              "config layer, and a new hermes_adapters package that implements the "
              "two existing HERMES protocols against the real Keras models. The "
              "experiments package depends on the distribution and nothing depends "
              "on the experiments package."),
        caption=("<b>One new package does the load-bearing work.</b> "
                 "<code>hermes_adapters/</code> (highlighted) implements the two "
                 "Protocols <code>hermes</code> already defines, so the stacks join "
                 "with <i>zero</i> changes to <code>hermes/</code> — and the "
                 "<code>experiments</code> arrow finally points only inward."),
    )


# --------------------------------------------------------------------------- #
# D4 — tiers and programs
# --------------------------------------------------------------------------- #

def d4_tiers() -> str:
    P = "d4"
    s: List[str] = []
    SM = "t-sm"

    # Tier 1
    s.append(band(20, 150, 180, 154, "TIER 1 · DEVICE"))
    s.append(box(34, 182, 152, 106,
                 ["ClientMission", ("· flagger", SM),
                  ("· offline trainer", SM), ("· FL client", SM)], lh=15))

    # Tier 2-mobile
    s.append(band(280, 46, 310, 314, "TIER 2-MOBILE · MULE NUC",
                  cls="band ok-band"))
    s.append(box(296, 78, 278, 30, ["MuleSupervisor"], lh=12))
    s.append(box(296, 124, 132, 52,
                 ["ChannelDDQN", ("L1 · band choice", SM)], lh=13))
    s.append(box(442, 124, 132, 52,
                 ["FLScheduler", ("S1→S3a→S3.5", SM)], lh=13))
    s.append(box(296, 200, 132, 52,
                 ["HFLHostMission", ("FL server", SM)], lh=13))
    s.append(box(442, 200, 132, 52,
                 ["ClientCluster", ("dock client", SM)], lh=13))
    s.append(label(435, 292,
                   ["the only host holding a server role",
                    "and a client role in one process"]))

    # Tier 2
    s.append(band(670, 150, 230, 190, "TIER 2 · EDGE SERVER"))
    s.append(box(686, 182, 198, 146,
                 ["HFLHostCluster", ("· DeviceRegistry", SM),
                  ("· cross-mule FedAvg", SM), ("· θ_gen + synth", SM),
                  ("· slice dispatch", SM)], lh=16))

    # Tier 3
    s.append(band(670, 30, 230, 82, "TIER 3 · CLOUD"))
    s.append(box(686, 58, 198, 40,
                 ["Tier3Coordinator", ("θ_gen refinement", SM)], lh=13))

    # inter-tier links
    s.append(arrow([(190, 218), (276, 218)], f"{P}-a"))
    s.append(arrow([(276, 234), (190, 234)], f"{P}-a"))
    s.append(label(233, 202, ["RFLink"], cls="t-hd"))
    s.append(label(233, 254, ["Solicit/Adv", "Push/Δθ/Ack"]))

    s.append(arrow([(594, 218), (666, 218)], f"{P}-a"))
    s.append(arrow([(666, 234), (594, 234)], f"{P}-a"))
    s.append(label(630, 202, ["DockLink"], cls="t-hd"))
    s.append(label(630, 254, ["UpBundle", "DownBundle"]))

    s.append(arrow([(770, 146), (770, 116)], f"{P}-a"))
    s.append(arrow([(786, 116), (786, 146)], f"{P}-a"))
    s.append(label(908, 126, ["CloudLink"], anchor="start", cls="t-hd"))
    s.append(label(908, 140, ["pickle over", "plain HTTP"],
                   anchor="start", cls="t-bad"))

    # intra-NUC buses
    s.append(arrow([(362, 200), (362, 182)], f"{P}-soft", cls="ln-soft"))
    s.append(arrow([(508, 200), (508, 182)], f"{P}-soft", cls="ln-soft"))
    s.append(label(348, 194, ["RoundCloseDelta"]))
    s.append(label(516, 194, ["slice + amendment"]))
    s.append(arrow([(362, 124), (362, 112)], f"{P}-soft", cls="ln-soft"))
    s.append(arrow([(508, 124), (508, 112)], f"{P}-soft", cls="ln-soft"))

    # legend
    s.append('<line x1="20" y1="392" x2="54" y2="392" class="ln"/>')
    s.append(_t(62, 396, "inter-tier link (transport ABC)", "t-lbl", anchor="start"))
    s.append('<line x1="300" y1="392" x2="334" y2="392" class="ln-soft"/>')
    s.append(_t(342, 396, "intra-NUC bus (in-process callable)", "t-lbl",
                anchor="start"))

    return figure(
        "".join(s), view="0 0 1020 412", prefix=P,
        aria=("Four-tier topology. The edge device runs ClientMission. The mule NUC "
              "runs MuleSupervisor over ChannelDDQN, FLScheduler, HFLHostMission and "
              "ClientCluster, connected to devices by the RF link and to the edge "
              "server by the dock link. The edge server runs HFLHostCluster holding "
              "the authoritative device registry, and reaches the Tier 3 cloud "
              "coordinator over an HTTP link."),
        caption=("<b>Every tier boundary is a transport ABC.</b> Programs never see a "
                 "socket, which is why the whole topology runs in-process under "
                 "loopback links in tests. The mule NUC is the only host holding a "
                 "server role and a client role at once."),
    )


# --------------------------------------------------------------------------- #
# D5 — HIFINS loader chain
# --------------------------------------------------------------------------- #

def d5_loader() -> str:
    P = "d5"
    s: List[str] = []
    SM = "t-sm"
    CX, W = 120, 300           # main column 120..420
    CM = CX + W / 2            # 270

    def step(y: float, h: float, rows: Sequence[Row], cls: str = "bx") -> str:
        return box(CX, y, W, h, rows, cls=cls, lh=14)

    s.append(box(CX + 90, 20, 120, 30, ["argv"], lh=12))
    s.append(arrow([(CM, 50), (CM, 64)], f"{P}-a"))

    s.append(step(64, 56, ["parse_training_client_args()",
                           ("154 lines · 21 dead AC-GAN flags", SM)]))
    s.append(label(CX + W + 14, 92, ["ArgumentConfigLoad.py:8"], anchor="start"))

    # mode gate
    s.append(f'<path d="M {CM} 136 L {CM + 104} 166 L {CM} 196 '
             f'L {CM - 104} 166 Z" class="bx"/>')
    s.append(_t(CM, 170, "args.mode", "", anchor="middle"))

    # dead branch (right)
    s.append(arrow([(CM + 104, 166), (CM + 176, 166)], f"{P}-dead", cls="ln-dead"))
    s.append(label(CM + 140, 158, ["hermes"]))
    s.append(box(CM + 180, 138, 250, 56,
                 ["_run_hermes_main()", ("local_train raises RuntimeError", SM),
                  ("DEAD BRANCH", SM)], cls="bx badbx", lh=13))

    s.append(arrow([(CM, 196), (CM, 214)], f"{P}-a"))
    s.append(label(CM + 12, 210, ["legacy (default)"], anchor="start"))

    s.append(step(214, 66, ["datasetLoadProcess(args)",
                            ("loadCICIOT / IOTBOTNET / IoT / live", SM),
                            ("→ preprocess_*", SM)]))
    s.append(label(CX + W + 14, 240,
                   ["DATASET_DIRECTORY ="], anchor="start"))
    s.append(label(CX + W + 14, 252,
                   ["'../../../../datasets/…'", "← CWD-relative"],
                   anchor="start", cls="t-bad"))
    s.append(arrow([(CM, 280), (CM, 298)], f"{P}-a"))

    s.append(step(298, 52, ["hyperparameterLoading(args, X_train)",
                            ("→ 20-element positional tuple", SM)]))
    s.append(arrow([(CM, 350), (CM, 368)], f"{P}-a"))

    s.append(step(368, 62, ["modelCreateLoad(13 positional args)",
                            ("375 lines · complexity 89", SM),
                            ("→ (nids, disc, gen, GAN)", SM)]))
    s.append(arrow([(CM, 430), (CM, 448)], f"{P}-a"))

    s.append(step(448, 62, ["modelCentralTrainingConfigLoad(41 args)",
                            ("modelFederatedTrainingConfigLoad(41 args)", SM),
                            ("if / elif over model_type × train_type", SM)]))

    # split
    s.append(arrow([(CX + 70, 510), (CX + 70, 548)], f"{P}-a"))
    s.append(arrow([(CX + 230, 510), (CX + 230, 548)], f"{P}-bad", cls="ln-bad"))

    s.append(box(CX - 60, 548, 240, 52,
                 [("one of ~14 trainer classes", SM),
                  ("NIDS · GAN · WGAN-GP · AC-GAN", SM)], lh=14))
    s.append(box(CX + 196, 548, 234, 52,
                 [("client = None", SM),
                  ("CANGAN · NIDS-IOT ×3", SM)], cls="bx badbx", lh=14))

    s.append(arrow([(CX + 60, 600), (CX + 60, 622)], f"{P}-a"))
    s.append(arrow([(CX + 313, 600), (CX + 313, 622)], f"{P}-bad", cls="ln-bad"))

    s.append(box(CX - 60, 622, 240, 50,
                 [("client.fit()", SM),
                  ("client.evaluate() · save()", SM)], lh=14))
    s.append(box(CX + 196, 622, 234, 50,
                 [("AttributeError:", SM),
                  ("'NoneType' has no 'fit'", SM)], cls="bx badbx", lh=14))
    s.append(label(CX + 313, 690, ["…after the full dataset load"], cls="t-bad"))

    return figure(
        "".join(s), view="0 0 700 712", prefix=P,
        aria=("Flow of the legacy training loader chain. Argument parsing feeds a "
              "mode gate whose hermes branch is a dead stub. The legacy branch loads "
              "and preprocesses the dataset, builds a twenty-element hyperparameter "
              "tuple, calls a 375-line model factory, then a 41-parameter dispatcher "
              "that either returns one of about fourteen trainer classes or returns "
              "None for four advertised model types, which crashes with an "
              "AttributeError after the dataset has already been loaded."),
        caption=("<b>Every stage widens the parameter surface.</b> That is the "
                 "mechanical reason 21 CLI hyperparameters were declared and never "
                 "wired: threading one value through this chain means editing six "
                 "call sites. Both red terminals are reachable from the documented "
                 "<code>--help</code> output."),
    )


# --------------------------------------------------------------------------- #
# D6 — refactoring phase dependencies
# --------------------------------------------------------------------------- #

def d6_phases() -> str:
    P = "d6"
    s: List[str] = []
    SM = "t-sm"
    BW, BH = 148, 64

    def phase(x: float, y: float, rows: Sequence[Row], cls: str = "bx") -> str:
        return box(x, y, BW, BH, rows, cls=cls, lh=14)

    # columns
    X0, X1, X2, X3, X4, X5 = 20, 200, 380, 560, 740, 920
    MID, TOP, BOT = 112, 30, 194

    s.append(phase(X0, MID, ["Phase 0", ("Stop the bleeding", SM),
                             ("3 d · very low risk", SM)]))
    s.append(phase(X1, MID, ["Phase 1", ("Foundations", SM),
                             ("1 w · low risk", SM)]))
    s.append(phase(X2, TOP, ["Phase 2", ("HERMES hardening", SM),
                             ("1.5 w · low–med", SM)]))
    s.append(phase(X2, BOT, ["Phase 3 ⚠", ("Correctness", SM),
                             ("3 d · approval-gated", SM)], cls="bx badbx"))
    s.append(phase(X3, MID, ["Phase 4", ("Bridge the stacks", SM),
                             ("2 w · medium", SM)]))
    s.append(phase(X4, MID, ["Phase 5", ("Training consolidation", SM),
                             ("4 w · HIGH RISK", SM)], cls="bx warnbx"))
    s.append(phase(X5, MID, ["Phase 6", ("Scale-out", SM),
                             ("3 w · medium", SM)]))

    cy, ty, by = MID + BH / 2, TOP + BH / 2, BOT + BH / 2

    # straight hops
    s.append(arrow([(X0 + BW, cy), (X1 - 4, cy)], f"{P}-a"))
    s.append(arrow([(X3 + BW, cy), (X4 - 4, cy)], f"{P}-a"))
    s.append(arrow([(X4 + BW, cy), (X5 - 4, cy)], f"{P}-a"))

    # P1 fans out to P2 and P3
    s.append(arrow([(X1 + BW, cy - 10), (X1 + BW + 16, cy - 10),
                    (X1 + BW + 16, ty), (X2 - 4, ty)], f"{P}-a"))
    s.append(arrow([(X1 + BW, cy + 10), (X1 + BW + 16, cy + 10),
                    (X1 + BW + 16, by), (X2 - 4, by)], f"{P}-a"))

    # P2 and P3 converge on P4
    s.append(arrow([(X2 + BW, ty), (X3 - 20, ty), (X3 - 20, cy - 10),
                    (X3 - 4, cy - 10)], f"{P}-a"))
    s.append(arrow([(X2 + BW, by), (X3 - 20, by), (X3 - 20, cy + 10),
                    (X3 - 4, cy + 10)], f"{P}-a"))

    # P1 -> P5, the characterization gate (routed under everything)
    s.append(arrow([(X1 + BW / 2, MID + BH), (X1 + BW / 2, 306),
                    (X4 + BW / 2, 306), (X4 + BW / 2, MID + BH + 4)],
                   f"{P}-bad", cls="ln-bad"))
    s.append(label(X3 + 20, 300,
                   ["R-08 characterization harness is the gate — phase 5 "
                    "cannot start before it is green"], cls="t-bad"))

    # P2 -> P6, routed over the top
    s.append(arrow([(X2 + BW / 2, TOP), (X2 + BW / 2, 14),
                    (X5 + BW / 2, 14), (X5 + BW / 2, MID - 4)], f"{P}-a"))

    # legend
    ly = 336
    s.append(f'<rect x="{X0}" y="{ly - 8}" width="14" height="11" rx="3" '
             f'class="bx badbx"/>')
    s.append(_t(X0 + 22, ly, "behaviour-changing — approve separately", "t-lbl",
                anchor="start"))
    s.append(f'<rect x="{X0 + 300}" y="{ly - 8}" width="14" height="11" rx="3" '
             f'class="bx warnbx"/>')
    s.append(_t(X0 + 322, ly, "high risk — gated on a characterization suite",
                "t-lbl", anchor="start"))

    return figure(
        "".join(s), view="0 0 1090 356", prefix=P,
        aria=("Phase dependency graph for the refactor. Phase 0 precedes phase 1. "
              "Phase 1 unblocks phases 2 and 3, which both feed phase 4, which feeds "
              "phase 5. Phase 2 also feeds phase 6, as does phase 5. A separate "
              "gating edge runs from phase 1 to phase 5: the R-08 characterization "
              "harness must be green before the training-stack consolidation starts. "
              "Phase 3 is behaviour-changing and approval-gated; phase 5 is high "
              "risk."),
        caption=("<b>Two constraints drive the ordering.</b> Phase 3 is isolated so "
                 "its approval conversation covers three specific behaviour changes "
                 "rather than a large PR. Phase 5 is gated on the phase-1 "
                 "characterization harness — consolidating 21.3 K LOC of untested "
                 "training code without it is a rewrite, not a refactor."),
    )


DIAGRAMS = {
    "System_Architecture_Overview.md": [d1_layers, d2_sequence, d3_target],
    "HERMES_Architecture.md": [d4_tiers],
    "HIFINS_Architecture.md": [d5_loader],
    "01_Refactoring_Strategy.md": [d6_phases],
}

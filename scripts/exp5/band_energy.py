"""Exploratory: does changing band in flight save energy?

    py -3.11 scripts/exp5/band_energy.py

Added 6 Nov 2026; not pre-registered. F flies the plan's band at every stop. FX and FQ
(gamma = 0.75) share F's plan and may fly another band at a Pass-1 stop, so a seed's F
trial is the same mission without the change. Everything comes from the kept mule traces,
paired by (trial, seed), at 1 MB, one mule, N = 6, 12 and 24, knee and stress.

1. Whole trial: energy per mission and per merged update, arm minus F, with a bootstrap
   CI and a sign-flip p, Holm across each arm's six cells and metric. Energy is split into
   moving (transit, return) and hovering (dwell, listen) at the model's two powers.
2. Mission 1: nothing has diverged yet, so the seed's two missions start from one plan.
   Pairs are split by whether the arm left the plan's band. Those that kept it are the
   control and should match F.
3. First change: the first stop in mission 1 where the arm left the plan's band, against
   the same stop in F. It holds the time at the stop, and the rule's arrival price of both
   bands: every target, both directions, at the rate read at the arrival.

Uplink drops are a keyed draw that ignores the band (FerryRuntime.uplink_drops), so a
faster band pays for a weaker link only through reach and the SNR floor. Read a saving here
with that in mind.

It writes results/exp5/scores/b2/exploratory_band_energy.{md,csv}.
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "exp5" / "scores" / "b2"

P_MOVE_W = 143.6
P_HOVER_W = 168.5
PAYLOAD_BYTES = 1_000_000
ARMS = {"FX": "FeRRy-Adapt", "FQ": "FeRRy-RL"}

_R = "results/exp5"
CELLS = [
    ("6, knee", 6, f"{_R}/b1/s514/n6k1_knee__capS_traces", f"{_R}/b1/s53/n6k1_knee__FX_traces",
     f"{_R}/b2/s53x/n6k1_knee__FQ-g75_traces"),
    ("6, stress", 6, f"{_R}/b1/s514/n6k1_stress__capS_traces", f"{_R}/b1/s53/n6k1_stress__FX_traces",
     f"{_R}/b2/s53x/n6k1_stress__FQ-g75_traces"),
    ("12, knee", 12, f"{_R}/b1/s59/n12k1_knee__F_traces", f"{_R}/b1/s59/n12k1_knee__FX_traces",
     f"{_R}/b2/s55/n12k1_knee__FQ-g75_traces"),
    ("12, stress", 12, f"{_R}/b2/s55/n12k1_stress__F_traces", f"{_R}/b2/s55/n12k1_stress__FX_traces",
     f"{_R}/b2/s55/n12k1_stress__FQ-g75_traces"),
    ("24, knee", 24, f"{_R}/b1/s59/n24k1_knee__F_traces", f"{_R}/b1/s59/n24k1_knee__FX_traces",
     f"{_R}/b2/s55/n24k1_knee__FQ-g75_traces"),
    ("24, stress", 24, f"{_R}/b2/s59x/n24k1_stress__F_traces", f"{_R}/b2/s59x/n24k1_stress__FX_traces",
     f"{_R}/b2/s55/n24k1_stress__FQ-g75_traces"),
]


# ---------------------------------------------------------------------- traces
def read_missions(path: Path) -> List[Dict[str, Any]]:
    """One mule trace's completed missions, in mission order."""
    out = []
    with Path(path).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"mission_completed"' not in line:
                continue
            e = json.loads(line)
            if e.get("event") != "mission_completed":
                continue
            ledger = e.get("sim_ledger") or {}
            move_s = ledger.get("transit", 0.0) + ledger.get("return", 0.0)
            hover_s = ledger.get("dwell", 0.0) + ledger.get("listen", 0.0)
            out.append({
                "round": e.get("mission_round"), "plan": e.get("band"),
                "planned": [sorted(s.get("devices") or []) for s in (e.get("pass_1_plan") or [])
                            if isinstance(s, dict)],
                "energy_j": float(e.get("energy_j") or 0.0),
                "move_j": P_MOVE_W * move_s, "hover_j": P_HOVER_W * hover_s,
                "updates": int(e.get("pass_1_merged_updates") or 0),
                "stops": [s for s in (e.get("pass_1_flown") or []) if isinstance(s, dict)],
            })
    return sorted(out, key=lambda m: m["round"])


def read_arm(root: Path) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    """Every trial of one job, keyed (trial, seed)."""
    out = {}
    for path in Path(root).glob("*/mule-exp4-mule.jsonl"):
        m = re.search(r"__t(\d+)__s(\d+)$", path.parent.name)
        if m:
            out[(int(m.group(1)), m.group(2))] = read_missions(path)
    return out


def left_plan(mission: Dict[str, Any]) -> bool:
    return any(s.get("band") != mission["plan"] for s in mission["stops"])


def same_plan(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["plan"] == b["plan"] and a["planned"] == b["planned"]


def _same_stop(x: Dict[str, Any], y: Dict[str, Any]) -> bool:
    return ([round(c, 2) for c in x["position"]] == [round(c, 2) for c in y["position"]]
            and sorted(x.get("devices") or []) == sorted(y.get("devices") or []))


def first_change(plan: Dict[str, Any], arm: Dict[str, Any]) -> Optional[Tuple[Dict, Dict]]:
    """The first stop where the arm's band differs, while both missions still fly one route."""
    for x, y in zip(plan["stops"], arm["stops"]):
        if not _same_stop(x, y):
            return None
        if x.get("band") != y.get("band"):
            return x, y
    return None


def arrival_price_s(stop: Dict[str, Any], payload_bytes: int = PAYLOAD_BYTES) -> float:
    """Every target, both directions, at the rate read at the arrival: the rule's own price."""
    rates = stop.get("rate_bps") or {}
    return sum(2 * 8.0 * payload_bytes / rates[d] for d in stop.get("targets") or []
               if rates.get(d, 0.0) > 0.0)


# ---------------------------------------------------------------------- statistics
def boot_ci(diffs: Sequence[float], n: int = 4000, seed: int = 7) -> Tuple[float, float]:
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def sign_flip_p(diffs: Sequence[float], n: int = 20000, seed: int = 11) -> float:
    rng = random.Random(seed)
    observed = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12
               for _ in range(n))
    return (hits + 1) / (n + 1)


def holm(ps: Sequence[float]) -> List[float]:
    """Holm-adjusted p values, in the input's order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def _kj(x: float) -> float:
    return x / 1000.0


def _mean(xs) -> float:
    return statistics.fmean(xs)


# ---------------------------------------------------------------------- views
def whole_trial(cell: str, arm: str, f_trials, a_trials) -> Dict[str, Any]:
    keys = sorted(k for k in set(f_trials) & set(a_trials) if f_trials[k] and a_trials[k])
    per = lambda ms, name: _mean(m[name] for m in ms)  # noqa: E731
    d_e = [per(a_trials[k], "energy_j") - per(f_trials[k], "energy_j") for k in keys]
    d_m = [per(a_trials[k], "move_j") - per(f_trials[k], "move_j") for k in keys]
    d_h = [per(a_trials[k], "hover_j") - per(f_trials[k], "hover_j") for k in keys]
    d_u, f_u, a_u = [], [], []
    for k in keys:
        uf, ua = sum(m["updates"] for m in f_trials[k]), sum(m["updates"] for m in a_trials[k])
        if uf and ua:
            ef = sum(m["energy_j"] for m in f_trials[k]) / uf
            ea = sum(m["energy_j"] for m in a_trials[k]) / ua
            f_u.append(ef), a_u.append(ea), d_u.append(ea - ef)
    stops = [s for k in keys for m in a_trials[k] for s in m["stops"]]
    off = sum(s.get("band") != m["plan"] for k in keys for m in a_trials[k] for s in m["stops"])
    lo, hi = boot_ci(d_e)
    ulo, uhi = boot_ci(d_u)
    return {"view": "whole trial", "cell": cell, "arm": arm, "pairs": len(keys),
            "stops": len(stops), "stops_off_plan": off,
            "f_kj": _kj(_mean(per(f_trials[k], "energy_j") for k in keys)),
            "arm_kj": _kj(_mean(per(a_trials[k], "energy_j") for k in keys)),
            "diff_kj": _kj(_mean(d_e)), "ci_lo": _kj(lo), "ci_hi": _kj(hi), "p": sign_flip_p(d_e),
            "move_diff_kj": _kj(_mean(d_m)), "hover_diff_kj": _kj(_mean(d_h)),
            "f_kj_per_update": _kj(_mean(f_u)), "arm_kj_per_update": _kj(_mean(a_u)),
            "update_diff_kj": _kj(_mean(d_u)), "update_ci_lo": _kj(ulo), "update_ci_hi": _kj(uhi),
            "update_p": sign_flip_p(d_u)}


def mission_one(cell: str, arm: str, f_trials, a_trials) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Tuple[Dict, Dict]]] = defaultdict(list)
    for k in sorted(set(f_trials) & set(a_trials)):
        if f_trials[k] and a_trials[k] and same_plan(f_trials[k][0], a_trials[k][0]):
            b = a_trials[k][0]
            groups["left the band" if left_plan(b) else "kept the band"].append((f_trials[k][0], b))
    rows = []
    for name, pairs in sorted(groups.items()):
        d = [b["energy_j"] - a["energy_j"] for a, b in pairs]
        lo, hi = boot_ci(d)
        rows.append({"view": "mission 1", "cell": cell, "arm": arm, "group": name, "pairs": len(d),
                     "f_kj": _kj(_mean(a["energy_j"] for a, _ in pairs)),
                     "arm_kj": _kj(_mean(b["energy_j"] for _, b in pairs)),
                     "diff_kj": _kj(_mean(d)), "ci_lo": _kj(lo), "ci_hi": _kj(hi), "p": sign_flip_p(d),
                     "largest_abs_diff_kj": _kj(max(abs(x) for x in d)),
                     "f_updates": _mean(a["updates"] for a, _ in pairs),
                     "arm_updates": _mean(b["updates"] for _, b in pairs)})
    return rows


def change_stops(n: int, arm: str, f_trials, a_trials) -> Dict[Tuple[str, int, str], List[Dict]]:
    out: Dict[Tuple[str, int, str], List[Dict]] = defaultdict(list)
    for k in set(f_trials) & set(a_trials):
        if not (f_trials[k] and a_trials[k]) or not same_plan(f_trials[k][0], a_trials[k][0]):
            continue
        hit = first_change(f_trials[k][0], a_trials[k][0])
        if hit:
            x, y = hit
            out[(arm, n, f"{x['band']} to {y['band']}")].append({
                "plan_s": float(x["dwell_s"]), "arm_s": float(y["dwell_s"]),
                "plan_price_s": arrival_price_s(x), "arm_price_s": arrival_price_s(y)})
    return out


def change_row(key: Tuple[str, Any, str], stops: List[Dict]) -> Dict[str, Any]:
    arm, n, change = key
    plan_s, arm_s = _mean(s["plan_s"] for s in stops), _mean(s["arm_s"] for s in stops)
    return {"view": "first change", "arm": arm, "N": n, "change": change, "stops": len(stops),
            "plan_band_s": plan_s, "chosen_band_s": arm_s,
            "stop_energy_diff_kj": _kj(P_HOVER_W * (arm_s - plan_s)),
            "plan_band_price_s": _mean(s["plan_price_s"] for s in stops),
            "chosen_band_price_s": _mean(s["arm_price_s"] for s in stops),
            "price_said_faster": sum(s["arm_price_s"] < s["plan_price_s"] for s in stops),
            "was_faster": sum(s["arm_s"] < s["plan_s"] for s in stops)}


# ---------------------------------------------------------------------- report
def _sig(lo: float, hi: float) -> str:
    return "" if lo < 0.0 < hi else " *"


def report(whole, first, changes, overall, worst_split) -> List[str]:
    lines = ["# Exploratory: does changing band in flight save energy?", "",
             "Not pre-registered (6 Nov 2026). `scripts/exp5/band_energy.py`, from the kept mule "
             "traces, 1 MB, one mule, paired by trial and seed against F, which flies the same plan "
             "on the plan's band. Difference = arm - F, so negative is a saving. Holm across each "
             "arm's six cells, per metric. Energy is simulated: "
             f"{P_MOVE_W} W moving and {P_HOVER_W} W hovering on every band, with no radio power. "
             "Uplink drops ignore the band, so a faster band pays for a weaker link only through "
             "reach and the SNR floor.", "",
             f"Largest gap between a mission's recorded energy and the split below: {worst_split:.3%}.",
             "", "## Whole trial", "",
             "| arm | devices, budget | pairs | stops off plan | F kJ per mission | arm | "
             "difference [95% CI] | Holm p | moving | hovering | kJ per merged update, difference "
             "[95% CI] | Holm p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in whole:
        lines.append(
            f"| {ARMS[r['arm']]} | {r['cell']} | {r['pairs']} | {r['stops_off_plan']} of {r['stops']} | "
            f"{r['f_kj']:.2f} | {r['arm_kj']:.2f} | {r['diff_kj']:+.2f} [{r['ci_lo']:+.2f}, "
            f"{r['ci_hi']:+.2f}]{_sig(r['ci_lo'], r['ci_hi'])} | {r['holm_p']:.4f} | "
            f"{r['move_diff_kj']:+.2f} | {r['hover_diff_kj']:+.2f} | {r['update_diff_kj']:+.2f} "
            f"[{r['update_ci_lo']:+.2f}, {r['update_ci_hi']:+.2f}]"
            f"{_sig(r['update_ci_lo'], r['update_ci_hi'])} | {r['update_holm_p']:.4f} |")
    lines += ["", "## Mission 1: one plan, band kept or left", "",
              "| arm | devices, budget | group | pairs | F kJ | arm kJ | difference [95% CI] | p | "
              "largest difference, kJ | merged updates F, arm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in first:
        lines.append(
            f"| {ARMS[r['arm']]} | {r['cell']} | {r['group']} | {r['pairs']} | {r['f_kj']:.2f} | "
            f"{r['arm_kj']:.2f} | {r['diff_kj']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
            f"{r['p']:.4f} | {r['largest_abs_diff_kj']:.2f} | {r['f_updates']:.2f}, {r['arm_updates']:.2f} |")
    head = ("| arm | devices | change | stops | seconds at the stop, plan band | chosen band | stop energy, "
            "kJ | arrival price, plan band, s | chosen band | price said faster | was faster |")
    lines += ["", "## First change in mission 1, against the same stop in F", "", head,
              "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(r):
        return (f"| {ARMS[r['arm']]} | {r['N']} | {r['change']} | {r['stops']} | {r['plan_band_s']:.1f} | "
                f"{r['chosen_band_s']:.1f} | {r['stop_energy_diff_kj']:+.2f} | {r['plan_band_price_s']:.1f} | "
                f"{r['chosen_band_price_s']:.1f} | {r['price_said_faster']} | {r['was_faster']} |")
    lines += [row(r) for r in changes]
    lines += ["", "## First change, all device counts", "", head,
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [row(r) for r in overall]
    return lines + [""]


def main() -> None:
    whole, first, grouped = [], [], defaultdict(list)
    worst = 0.0
    for cell, n, f_dir, fx_dir, fq_dir in CELLS:
        f_trials = read_arm(REPO / f_dir)
        for trials in [f_trials]:
            for ms in trials.values():
                for m in ms:
                    if m["energy_j"]:
                        worst = max(worst, abs(m["move_j"] + m["hover_j"] - m["energy_j"]) / m["energy_j"])
        for arm, root in (("FX", fx_dir), ("FQ", fq_dir)):
            a_trials = read_arm(REPO / root)
            whole.append(whole_trial(cell, arm, f_trials, a_trials))
            first.extend(mission_one(cell, arm, f_trials, a_trials))
            for key, stops in change_stops(n, arm, f_trials, a_trials).items():
                grouped[key].extend(stops)
    for arm in ARMS:
        rows = [r for r in whole if r["arm"] == arm]
        for r, q, u in zip(rows, holm([r["p"] for r in rows]), holm([r["update_p"] for r in rows])):
            r["holm_p"], r["update_holm_p"] = q, u
    whole.sort(key=lambda r: r["arm"] != "FX")
    first.sort(key=lambda r: r["arm"] != "FX")
    changes = [change_row(k, v) for k, v in sorted(grouped.items(), key=lambda kv: (kv[0][0] != "FX", kv[0][1], kv[0][2]))]
    pooled = defaultdict(list)
    for (arm, _n, change), stops in grouped.items():
        pooled[(arm, "all", change)].extend(stops)
    overall = [change_row(k, v) for k, v in sorted(pooled.items(), key=lambda kv: (kv[0][0] != "FX", kv[0][2]))]

    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(report(whole, first, changes, overall, worst))
    (OUT / "exploratory_band_energy.md").write_text(text, encoding="utf-8")
    rows = whole + first + changes + overall
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with (OUT / "exploratory_band_energy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(text)


if __name__ == "__main__":
    main()

"""Capture of the legacy exp4 channel (``experiments/exp4/channel.py`` at afa9526).

Phase 3 (unit U2) moves ``ChannelModel``, ``loss_from_snr``, ``BackhaulPlan``
and ``backhaul_plan`` verbatim to ``hermes/l1/channel_model.py`` and leaves
``experiments/exp4/channel.py`` as a re-export shim. These cases pin what the
code computes, bit for bit: the per-band SNR trace, the loss schedule, the
chosen bands and the chosen-band mean SNR that the driver hands the mule as its
``rf_prior_snr_db``, over seeds x regime x mission count x band count, for the
fixed-band (H1/H2) and the adaptive (H3) controller.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tests.golden._canon import canon, f

#: Seeds: small ones, and 32-bit ones like the runner's paired seeds (two are
#: seeds of recorded C2 trials). The recorded C1/C2 trials add 120 more below.
SEEDS = (0, 1, 7, 42, 12345, 2191267877, 3959840510, 4294967295)
REGIMES = ("clean", "jittery")
N_MISSIONS = (1, 2, 3, 4, 6, 8)
N_BANDS = (3,)
#: Extra band counts (the driver's ``l1_channel_bands``) on a few seeds.
EXTRA_BANDS = ((2, (0, 7, 2191267877)), (4, (0, 7, 2191267877)))
#: Non-default controller settings (``switch_cost``, ``channel_use_cost``).
CONTROLLER_VARIANTS = (
    ("switch0", dict(switch_cost=0.0)),
    ("switch2", dict(switch_cost=2.0)),
    ("usecost", dict(channel_use_cost=(0.0, 0.5, 1.0))),
)

#: ``loss_from_snr`` sample points, and the non-default logistic parameters.
LOSS_GRID = tuple(x / 4.0 for x in range(-60, 161))   # -15 dB .. 40 dB
LOSS_PARAMS = ((3.0, 2.0), (0.0, 1.0), (6.5, 3.25))

REPO = Path(__file__).resolve().parents[2]
RECORDED_TRACE_SETS = (
    REPO / "results" / "exp4_matrix" / "C_traces",
    REPO / "results" / "exp4_matrix" / "C2_traces",
)


def _channel_module():
    from experiments.exp4 import channel
    return channel


def plan_record(plan) -> Dict[str, Any]:
    """The afa9526 ``BackhaulPlan`` field set, exact."""
    return canon(plan)


def model_record(model) -> Dict[str, Any]:
    return {
        "fields": canon(model),
        "snr_by_mission": canon(model.snr_by_mission()),
    }


def _cell_key(seed: int, regime: str, n_missions: int, n_bands: int) -> str:
    return f"seed={seed}|{regime}|m={n_missions}|b={n_bands}"


def iter_grid() -> Iterator[Tuple[int, str, int, int]]:
    for seed in SEEDS:
        for regime in REGIMES:
            for n in N_MISSIONS:
                for b in N_BANDS:
                    yield seed, regime, n, b
    for b, seeds in EXTRA_BANDS:
        for seed in seeds:
            for regime in REGIMES:
                for n in (1, 4, 6):
                    yield seed, regime, n, b


def build_cell(module, seed: int, regime: str, n_missions: int, n_bands: int) -> Dict[str, Any]:
    model = module.ChannelModel(
        n_bands=n_bands, n_missions=n_missions, seed=seed, jittery=(regime == "jittery"),
    )
    out: Dict[str, Any] = {"model": model_record(model)}
    out["fixed"] = plan_record(module.backhaul_plan(model, adaptive=False))
    out["adaptive"] = plan_record(module.backhaul_plan(model, adaptive=True))
    return out


def build_cases(module=None) -> Dict[str, Any]:
    module = module or _channel_module()
    cases: Dict[str, Any] = {}
    for seed, regime, n, b in iter_grid():
        cases["cell:" + _cell_key(seed, regime, n, b)] = build_cell(module, seed, regime, n, b)
    for name, kwargs in CONTROLLER_VARIANTS:
        for seed in (0, 7, 2191267877):
            for regime in REGIMES:
                model = module.ChannelModel(n_bands=3, n_missions=6, seed=seed,
                                            jittery=(regime == "jittery"))
                cases[f"controller:{name}|seed={seed}|{regime}"] = {
                    "adaptive": plan_record(module.backhaul_plan(model, adaptive=True, **kwargs)),
                    "fixed": plan_record(module.backhaul_plan(model, adaptive=False, **kwargs)),
                }
    cases["loss_from_snr:default"] = [f(module.loss_from_snr(x)) for x in LOSS_GRID]
    for mid, scale in LOSS_PARAMS:
        cases[f"loss_from_snr:mid={mid}|scale={scale}"] = [
            f(module.loss_from_snr(x, mid=mid, scale=scale)) for x in LOSS_GRID
        ]
    # Zero missions: no trace, an empty schedule and a 0.0 mean.
    empty = module.ChannelModel(n_bands=3, n_missions=0, seed=5, jittery=True)
    cases["edge:zero_missions"] = {
        "snr_by_mission": canon(empty.snr_by_mission()),
        "fixed": plan_record(module.backhaul_plan(empty, adaptive=False)),
        "adaptive": plan_record(module.backhaul_plan(empty, adaptive=True)),
    }
    return cases


# --------------------------------------------------------------------------- #
# The recorded C1/C2 L1-channel trials (results/exp4_matrix/C*_traces)
# --------------------------------------------------------------------------- #

_DIR_RE = re.compile(r"^(?P<cell>.+)__(?P<arm>[A-Z]\d)__t(?P<trial>\d+)__s(?P<seed>\d+)$")


def parse_trace_dir(name: str) -> Optional[Dict[str, Any]]:
    """Cell params, arm, trial and seed from a kept trace directory's name.

    The driver writes ``<cell_id>__<arm>__t<trial>__s<seed>`` with the cell
    id's ``|`` sanitised to ``-``, e.g. ``N=6-dead_zone=0.6-...-rrf=60.0``.
    """
    m = _DIR_RE.match(name)
    if m is None:
        return None
    params: Dict[str, str] = {}
    for part in re.split(r"-(?=[A-Za-z_]+=)", m.group("cell")):
        key, _, val = part.partition("=")
        params[key] = val
    return {
        "cell": m.group("cell"),
        "arm": m.group("arm"),
        "trial": int(m.group("trial")),
        "seed": int(m.group("seed")),
        "params": params,
    }


def recorded_trials() -> List[Path]:
    out: List[Path] = []
    for root in RECORDED_TRACE_SETS:
        if root.is_dir():
            out.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return out


def rederive_recorded(trace_dir: Path, module=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(recorded, re-derived) channel fields of one kept trial.

    Recorded: the cluster's ``backhaul_loss_schedule`` and
    ``backhaul_rng_seed`` and the mule's ``rf_prior_snr_db``. Re-derived: the
    same three from today's code, the way ``Exp4Driver.run_trial`` computes
    them for an L1-channel cell (3 bands, H3 adaptive, the rest fixed).
    """
    module = module or _channel_module()
    info = parse_trace_dir(trace_dir.name)
    if info is None:
        raise ValueError(f"unparseable trace directory name {trace_dir.name!r}")
    cluster = json.loads((trace_dir / "cluster.json").read_text(encoding="utf-8"))
    mule_files = sorted(trace_dir.glob("mule-*.json"))
    mule = json.loads(mule_files[0].read_text(encoding="utf-8"))
    n_missions = int(info["params"]["n_missions"])
    regime = info["params"]["regime"]
    model = module.ChannelModel(
        n_bands=3, n_missions=n_missions, seed=info["seed"], jittery=(regime == "jittery"),
    )
    plan = module.backhaul_plan(model, adaptive=(info["arm"] == "H3"))
    recorded = {
        "backhaul_loss_schedule": canon(cluster["backhaul_loss_schedule"]),
        "backhaul_rng_seed": cluster["backhaul_rng_seed"],
        "rf_prior_snr_db": canon(mule["rf_prior_snr_db"]),
    }
    derived = {
        "backhaul_loss_schedule": canon(plan.loss_schedule),
        "backhaul_rng_seed": info["seed"] ^ 0x0BACC0DE,
        "rf_prior_snr_db": canon(plan.mean_chosen_snr_db),
    }
    return recorded, derived

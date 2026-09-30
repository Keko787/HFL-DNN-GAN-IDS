"""Legacy pin (Freeze Rule 1): the feasibility walks at afa9526.

Unit U4 rebuilds S3b, the D-arm budget walks, FedCS, the FedEx tour and the
mule's in-flight check as folds over one predicate. With the Phase 3 switches
at their defaults every kept/dropped list and every diagnostic must match what
afa9526 computes, over 2,400 seeded random instances (positions, deadlines,
budgets, cost models, miss priorities, poses, clocks) and the exact-boundary
cases the existing tests pin. Fixture: ``data/feasibility.json``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import pytest

from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import ContactWaypoint

from tests.golden import _build_feasibility as B
from tests.golden._canon import BASE_COMMIT, canon, diff, digest, load

GOLDEN = load("feasibility")
RANDOM_KEYS = [k for k in GOLDEN["cases"] if k.startswith("random:")]
BOUNDARY_KEYS = [k for k in GOLDEN["cases"] if k.startswith("boundary:")]


def test_fixture_was_captured_at_the_base_commit():
    assert GOLDEN["_meta"]["base_commit"] == BASE_COMMIT
    assert len(RANDOM_KEYS) >= 2000


def test_exact_boundary_cases():
    current = {f"boundary:{k}": v for k, v in B.build_boundary_cases().items()}
    bad = []
    for key in BOUNDARY_KEYS:
        problems = diff(GOLDEN["cases"][key], current.get(key, "<missing>"))
        if problems:
            bad.append(f"{key}: " + "; ".join(problems[:3]))
    assert not bad, f"{len(bad)} boundary case(s) changed:\n  " + "\n  ".join(bad)


def test_random_instances():
    """Every instance's results, by digest; the first ones also in full."""
    bad = []
    for key in RANDOM_KEYS:
        seed = int(key.split(":")[1])
        golden = GOLDEN["cases"][key]
        inst = B.make_instance(seed)
        if B.instance_digest(inst) != golden["in"]:
            bad.append(
                f"{key}: the instance no longer hashes as recorded, so its results "
                f"cannot be compared. Either this test's generator changed, or a field "
                f"ContactWaypoint/FeasibilityModel had at afa9526 changed its value or "
                f"type (added default-valued fields are ignored). Do not regenerate the "
                f"fixture from changed code: its results are the afa9526 pins")
            continue
        result = B.run_instance(inst)
        if digest(result) != golden["out"]:
            detail = golden.get("detail")
            if detail is not None:
                why = "; ".join(diff(detail, result)[:4])
            else:
                why = f"results now {result}"
            bad.append(f"{key}: {why}")
    assert not bad, (
        f"{len(bad)} of {len(RANDOM_KEYS)} random instance(s) changed since afa9526:\n  "
        + "\n  ".join(bad[:12])
    )


@pytest.mark.parametrize("walk", [
    "s3b", "s3b_prio", "greedy_edf", "greedy_order", "d1_max_aoi", "fedcs_unit",
    "fedcs_devices", "d2_oort", "d3_whittle_expected", "d3_whittle_literal",
    "fedex_closed", "fedex_dock", "inflight", "pass_2",
])
def test_each_walk_on_the_detailed_instances(walk):
    """Per walk, so a failure names which of them moved."""
    bad = []
    for key in RANDOM_KEYS:
        golden = GOLDEN["cases"][key].get("detail")
        if golden is None:
            continue
        result = B.run_instance(B.make_instance(int(key.split(":")[1])))
        problems = diff(golden[walk], result[walk])
        if problems:
            bad.append(f"{key}.{walk}: " + "; ".join(problems[:3]))
    assert not bad, "\n  ".join([f"{walk} changed:"] + bad[:10])


# --------------------------------------------------------------------------- #
# The instance digest is a projection on the afa9526 fields (critic A3)
# --------------------------------------------------------------------------- #

def test_the_afa9526_fields_still_exist():
    """A field the digest projects on must not be removed or renamed."""
    for cls in (ContactWaypoint, FeasibilityModel):
        have = [f.name for f in dataclasses.fields(cls)]
        missing = [n for n in B.AFA9526_FIELDS[cls.__name__] if n not in have]
        assert not missing, f"{cls.__name__} lost its afa9526 field(s) {missing}"


def _phase3_types():
    """The two dataclasses with the default-valued fields unit U4 plans.

    Design section 4.5 (``band``, ``range_m``, ``pred_snr_db``, all
    ``compare=False``) and section 3.1 (``ferry=None``). Subclasses under the
    same names, so only the added fields differ.
    """
    no_cmp = dict(default=None, compare=False)
    waypoint = dataclasses.make_dataclass(
        "ContactWaypoint",
        [("band", Optional[str], dataclasses.field(**no_cmp)),
         ("range_m", Optional[float], dataclasses.field(**no_cmp)),
         ("pred_snr_db", Optional[Tuple[float, ...]], dataclasses.field(**no_cmp))],
        bases=(ContactWaypoint,), frozen=True,
    )
    model = dataclasses.make_dataclass(
        "FeasibilityModel", [("ferry", object, dataclasses.field(default=None))],
        bases=(FeasibilityModel,), frozen=True,
    )
    return waypoint, model


def _with_phase3_fields(inst):
    waypoint, model = _phase3_types()
    out = dict(inst)
    out["contacts"] = [
        waypoint(position=c.position, devices=c.devices, bucket=c.bucket,
                 deadline_ts=c.deadline_ts, band="wide", range_m=60.0,
                 pred_snr_db=tuple(12.5 for _ in c.devices))
        for c in inst["contacts"]
    ]
    m = inst["model"]
    # Legacy mode keeps ``ferry=None``; the waypoints get values a ferry-mode
    # mule would attach, which the digest must ignore just the same.
    out["model"] = None if m is None else model(
        cruise_speed_m_s=m.cruise_speed_m_s, session_time_s=m.session_time_s)
    return out


def test_instance_digest_ignores_additive_fields():
    """Positive control: U4's added fields leave every recorded digest valid.

    (Hashing the whole canonical form, as before, broke on them: critic A3.)
    """
    for cls in _phase3_types():
        extra = {f.name for f in dataclasses.fields(cls)} - set(B.AFA9526_FIELDS[cls.__name__])
        assert extra, cls.__name__            # the stand-ins carry fields afa9526 lacked
    checked = 0
    for key in RANDOM_KEYS[:400]:
        inst = B.make_instance(int(key.split(":")[1]))
        if not inst["contacts"] or inst["model"] is None:
            continue
        extended = _with_phase3_fields(inst)
        assert canon(extended["contacts"][0])["band"] == "wide"
        assert B.instance_digest(extended) == GOLDEN["cases"][key]["in"], key
        checked += 1
    assert checked >= 100


def test_instance_digest_sees_the_afa9526_values():
    """Negative control: one ulp on an afa9526 field moves the digest."""
    inst = next(i for i in (B.make_instance(s) for s in range(200))
                if i["contacts"] and i["model"] is not None)
    base = B.instance_digest(inst)
    c0 = inst["contacts"][0]
    nudged = dict(inst, contacts=[dataclasses.replace(
        c0, deadline_ts=math.nextafter(c0.deadline_ts, math.inf))] + inst["contacts"][1:])
    assert B.instance_digest(nudged) != base
    m = inst["model"]
    slower = dict(inst, model=dataclasses.replace(
        m, cruise_speed_m_s=math.nextafter(m.cruise_speed_m_s, 0.0)))
    assert B.instance_digest(slower) != base

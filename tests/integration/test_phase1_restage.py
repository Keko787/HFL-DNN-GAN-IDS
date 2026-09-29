"""FeRRy Phase 1 — an empty mission restages θ with its version.

When Pass 1 collects nothing the mule skips the dock and Pass 2 and flies the
same θ again next mission. The version has to be restaged with it: θ alone
would go out unversioned, the next mission's partial would carry no
``base_version``, and every update in it would come back with no age — which
``agg:cutoff`` cannot judge (audit #18).

Whether the next mission merges does not show the regression: the cutoff
counts an unknown age as 0, so even ``a_max=0``, the strictest cap, keeps an
unversioned update. The test therefore asserts the restaged version and the
ages and basis versions themselves, which go missing when θ is restaged alone.
"""

from __future__ import annotations

from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec

from tests.integration.test_phase1_two_pass_ages import _TIGHT, _run_mission, _setup


def test_an_empty_mission_restages_the_theta_version():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=0)
    sup, cluster, devices = _setup(_TIGHT, spec)
    assert sup._next_theta_version == 0         # the bootstrap DOWN's issued_round

    # No device answers: Pass 1 collects nothing, so there is no dock and no
    # Pass 2, and the same θ is restaged for the next mission.
    r1 = sup.run_one_mission()
    assert r1.empty and r1.aggregate is None
    assert sup._next_theta is not None
    assert sup._next_theta_version == 0

    r2 = _run_mission(sup, cluster, devices)
    agg = r2.aggregate
    assert agg is not None and agg.base_version == 0
    assert agg.device_ages and None not in agg.device_ages
    assert agg.device_basis_versions and None not in agg.device_basis_versions
    lines = [l for l in r2.report.lines if l.outcome.is_on_time()]
    assert lines and all(l.basis_version == 0 and l.age == 0 for l in lines)

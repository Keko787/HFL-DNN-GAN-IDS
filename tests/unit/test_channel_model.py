"""FeRRy Phase 3, unit U2: the seconds-axis channel (``hermes/l1/channel_model.py``).

What is pinned here (design §1 D2, spec D2; critic A4, A6/C5, A8-ii, B4):

* **The move.** The legacy mission-indexed model now lives in ``hermes.l1``,
  and ``experiments/exp4/channel.py`` re-exports the same objects. The numbers
  themselves are pinned bit for bit by ``tests/golden/test_golden_channel.py``,
  on both import paths. ``hermes`` does not import ``experiments``.
* **Noise as a pure function of time.** It is reproducible, and paired: the
  same (t, band, link) gives the same value whatever the query order.
  Different seeds differ. It is N(0, 1) at every t, its correlation decays
  over the bin, and it has no horizon up to t = 1e7 s.
* **The contact channel's terms** (mean + shadowing + interference; shadowing
  keyed by time or, as an option, by position; interference phase and noise
  keyed by class name, so a class keeps its ``I_b(t)`` across the class-set
  sweep), and the **backhaul channel's** ``fixed_band() = argmax g_c``,
  periodicity, controller and loss magnitudes (critic A4).
* **Keyed Bernoulli helpers**, paired by mission rather than by draw count.
* **The causal RF-prior producer** (critic B4): 20 dB before the first upload.

Fast and in-process; the only subprocess checks the import graph.
"""

from __future__ import annotations

import ast
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes.l1 import channel_model as cm
from hermes.l1.channel_utility import AdaptiveChannelController
from hermes.l1.rf_prior import DEFAULT_RF_PRIOR_SNR_DB, RFPrior, RFPriorProducer, RFPriorStore

REPO = Path(__file__).resolve().parents[2]
EPOCH = cm.DEFAULT_EPOCH_S


def _corr(a, b) -> float:
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    return cov / math.sqrt(statistics.pvariance(a) * statistics.pvariance(b))


def _mean_fn(name: str, d_planar: float) -> float:
    """A stand-in for ``ContactLink.mean_snr_db`` that only accepts band names."""
    if not isinstance(name, str):
        raise AssertionError(f"the mean must be called with a band name, got {name!r}")
    return {"wide": 10.0, "medium": 14.0, "narrow": 20.0}[name] - 0.1 * d_planar


def _contact(seed: int = 5, **kw) -> cm.ContactChannel:
    return cm.ContactChannel(_mean_fn, salt=cm.ferry_salt(seed, cm.SALT_CONTACT), **kw)


def _backhaul(seed: int = 5, **kw) -> cm.BackhaulChannel:
    kw.setdefault("period_s", cm.backhaul_period_s(4, 219.0))
    return cm.BackhaulChannel(salt=cm.ferry_salt(seed, cm.SALT_BACKHAUL), **kw)


# --------------------------------------------------------------------------- #
# The move: legacy code in hermes.l1, experiments/exp4/channel.py a shim
# --------------------------------------------------------------------------- #

def test_shim_reexports_the_same_objects():
    import experiments.exp4.channel as shim
    import hermes.l1.channel_utility as util

    assert shim.ChannelModel is cm.ChannelModel
    assert shim.loss_from_snr is cm.loss_from_snr
    assert shim.BackhaulPlan is cm.BackhaulPlan
    assert shim.backhaul_plan is cm.backhaul_plan
    # The old module's own imports stay importable from it.
    assert shim.AdaptiveChannelController is util.AdaptiveChannelController
    assert shim.best_average_band is util.best_average_band
    assert sorted(shim.__all__) == sorted([
        "AdaptiveChannelController", "BackhaulPlan", "ChannelModel",
        "backhaul_plan", "best_average_band", "loss_from_snr",
    ])


def test_legacy_types_keep_their_names():
    """The golden fixtures record dataclasses by type name (``tests/golden/_canon.py``)."""
    assert cm.ChannelModel.__name__ == "ChannelModel"
    assert cm.BackhaulPlan.__name__ == "BackhaulPlan"


def test_legacy_plan_is_unchanged_through_the_shim():
    """A spot check next to the golden: both paths give the same plan."""
    from experiments.exp4.channel import ChannelModel, backhaul_plan

    for adaptive in (False, True):
        a = backhaul_plan(ChannelModel(n_bands=3, n_missions=6, seed=13, jittery=True),
                          adaptive=adaptive)
        b = cm.backhaul_plan(cm.ChannelModel(n_bands=3, n_missions=6, seed=13, jittery=True),
                             adaptive=adaptive)
        assert a == b


def test_the_legacy_helpers_are_patched_in_hermes_l1(monkeypatch):
    """The shim re-exports rather than wraps (a Rule 3 note in both docstrings):
    ``backhaul_plan`` resolves ``loss_from_snr`` in ``hermes.l1.channel_model``.
    Rebinding the name on ``experiments.exp4.channel`` no longer reaches it;
    patching it in ``hermes.l1.channel_model`` does, through either import path."""
    import experiments.exp4.channel as shim

    model = shim.ChannelModel(n_bands=3, n_missions=4, seed=7, jittery=True)
    before = shim.backhaul_plan(model, adaptive=False).loss_schedule
    assert shim.backhaul_plan.__globals__ is cm.__dict__
    monkeypatch.setattr(shim, "loss_from_snr", lambda snr_db, **kw: 0.5)
    assert shim.backhaul_plan(model, adaptive=False).loss_schedule == before
    monkeypatch.setattr(cm, "loss_from_snr", lambda snr_db, **kw: 0.5)
    assert shim.backhaul_plan(model, adaptive=False).loss_schedule == [0.5] * 4


@pytest.mark.parametrize("rel", ["hermes/l1/channel_model.py", "hermes/l1/rf_prior.py"])
def test_hermes_modules_do_not_import_experiments_directly(rel):
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(n == "experiments" or n.startswith("experiments.") for n in names), (
            f"{rel} imports {names} (finding A-01: hermes must not import experiments)")


def test_importing_the_channel_model_loads_no_experiments_module():
    code = ("import sys, hermes.l1.channel_model, hermes.l1.rf_prior; "
            "bad = [m for m in sys.modules if m == 'experiments' or m.startswith('experiments.')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_l1_package_exports_are_unchanged():
    """Design §2.1: ``hermes/l1/__init__.py`` stays as it is."""
    import hermes.l1 as l1

    assert sorted(l1.__all__) == ["CHANNEL_FREQS_GHZ", "ChannelDDQN", "RFPrior", "RFPriorStore"]


# --------------------------------------------------------------------------- #
# Salts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed,parts", [
    (0, ("ferry", "contact")),
    (2191267877, ("device_reliability", 6)),
    (4294967295, ("ferry", "backhaul_loss")),
    (7, ()),
    (-3, ("x", 1.5, None)),
])
def test_trial_salt_is_a_copy_of_model_task_u32(seed, parts):
    from experiments.exp4.model_task import _u32

    assert cm.trial_salt(seed, *parts) == _u32(seed, *parts)


def test_trial_salt_pinned_values():
    assert cm.trial_salt(0, "ferry", "contact") == 1763559150
    assert cm.trial_salt(2191267877, "device_reliability", 6) == 878576888
    assert cm.ferry_salt(42, cm.SALT_CONTACT) == 3071003727


def test_ferry_salts_are_distinct_per_stream_and_seed():
    streams = (cm.SALT_CONTACT, cm.SALT_BACKHAUL, cm.SALT_AVAILABILITY, cm.SALT_BACKHAUL_LOSS)
    salts = {(seed, s): cm.ferry_salt(seed, s) for seed in range(20) for s in streams}
    assert len(set(salts.values())) == len(salts)
    assert all(0 <= v < 2 ** 32 for v in salts.values())
    assert cm.ferry_salt(9, cm.SALT_CONTACT) == cm.trial_salt(9, "ferry", "contact")


# --------------------------------------------------------------------------- #
# smooth_normal: a pure, smooth function of time
# --------------------------------------------------------------------------- #

def test_smooth_normal_pinned_values():
    """Pins the hash layout and the Box-Muller branch. The tolerance absorbs
    last-bit differences between OS math libraries (critic A8-ii)."""
    cases = [
        ((12345, "shadow", "dev-3", EPOCH + 3.25, 7.4, EPOCH), -1.7463185005614525),
        ((12345, "interference", 1, EPOCH + 100.0, 1.0, EPOCH), 0.6119438910879597),
        ((7, "backhaul", 2, 17.0, 1.0, 0.0), 0.016017529800019675),
    ]
    for args, expected in cases:
        assert cm.smooth_normal(*args) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def _queries():
    rng = random.Random(11)
    out = []
    for i in range(300):
        out.append((rng.choice((1, 2, 3)), rng.choice(("shadow", "interference", "backhaul")),
                    rng.choice(("dev-0", "dev-1", 0, 1, 2)), EPOCH + rng.uniform(-50.0, 5000.0),
                    rng.choice((1.0, 7.4))))
    return out


def test_smooth_normal_is_pure_and_query_order_free():
    queries = _queries()
    forward = {q: cm.smooth_normal(*q, EPOCH) for q in queries}
    shuffled = list(queries)
    random.Random(3).shuffle(shuffled)
    backward = {}
    for q in reversed(shuffled):
        cm.smooth_normal(99, "noise", "other", q[3] + 0.5, 1.0, EPOCH)  # unrelated queries in between
        backward[q] = cm.smooth_normal(*q, EPOCH)
    assert backward == forward


def test_smooth_normal_differs_across_salt_stream_and_key():
    ts = [EPOCH + 0.37 + 2.7 * i for i in range(400)]
    base = [cm.smooth_normal(1, "s", "k", t, 1.0, EPOCH) for t in ts]
    for other in ([cm.smooth_normal(2, "s", "k", t, 1.0, EPOCH) for t in ts],
                  [cm.smooth_normal(1, "s2", "k", t, 1.0, EPOCH) for t in ts],
                  [cm.smooth_normal(1, "s", "k2", t, 1.0, EPOCH) for t in ts]):
        assert all(a != b for a, b in zip(base, other))
        assert abs(_corr(base, other)) < 0.15


def test_smooth_normal_is_continuous_and_hits_the_bin_draws():
    salt = 17
    for k in (0, 1, 5, 1000, -3):
        t = EPOCH + 7.4 * k
        at = cm.smooth_normal(salt, "shadow", "d", t, 7.4, EPOCH)
        assert cm.smooth_normal(salt, "shadow", "d", t - 1e-7, 7.4, EPOCH) == pytest.approx(at, abs=1e-6)
        assert cm.smooth_normal(salt, "shadow", "d", t + 1e-7, 7.4, EPOCH) == pytest.approx(at, abs=1e-6)
    # Inside a bin the value lies between the two bin draws (a convex mix, rescaled).
    z0 = cm.smooth_normal(salt, "s", "d", 10.0, 1.0, 0.0)
    z1 = cm.smooth_normal(salt, "s", "d", 11.0, 1.0, 0.0)
    mid = cm.smooth_normal(salt, "s", "d", 10.5, 1.0, 0.0)
    assert mid == pytest.approx((z0 + z1) / math.sqrt(2.0), rel=1e-12)


def test_smooth_normal_needs_an_explicit_epoch():
    """Design §2.1 gives ``epoch`` no default. A default of 0.0 put a direct
    caller that left it out on bins that do not line up with the channels'
    (epoch ``DEFAULT_EPOCH_S``), so its values were silently not paired."""
    ch = _contact()
    t = EPOCH + 50.0
    with pytest.raises(TypeError):
        cm.smooth_normal(ch.salt, "shadow", "dev-0", t, ch.shadow_corr_s)
    x = ch.shadow_db(t, "dev-0")
    assert x == ch.shadow_sigma_db * cm.smooth_normal(ch.salt, "shadow", "dev-0", t, 7.4, EPOCH)
    assert x == ch.shadow_sigma_db * cm.smooth_normal(ch.salt, "shadow", "dev-0", t, 7.4, epoch=EPOCH)
    # Another epoch shifts the 7.4 s bins (1e6 is not a multiple of 7.4): another value.
    assert x != ch.shadow_sigma_db * cm.smooth_normal(ch.salt, "shadow", "dev-0", t, 7.4, 0.0)


def test_smooth_normal_has_unit_variance():
    xs = [cm.smooth_normal(21, "s", "k", i * 2.6180339887, 1.0, 0.0) for i in range(20000)]
    assert abs(statistics.fmean(xs)) < 0.05
    assert statistics.pvariance(xs) == pytest.approx(1.0, abs=0.05)


def test_smooth_normal_keeps_unit_variance_mid_bin():
    """Plain linear interpolation would give variance 0.5 at u = 0.5."""
    xs = [cm.smooth_normal(21, "s", "k", k + 0.5, 1.0, 0.0) for k in range(20000)]
    assert statistics.pvariance(xs) == pytest.approx(1.0, abs=0.06)


def test_smooth_normal_correlation_decays_over_the_bin():
    """Theory for this interpolation: rho = 0.987, 0.736, 0.285, 0, 0 at lags of
    0.1, 0.5, 1, 2 and 3 bins (1/e at 0.9 bin)."""
    starts = [i * 3.3819660113 for i in range(20000)]
    base = [cm.smooth_normal(31, "c", "k", t, 1.0, 0.0) for t in starts]

    def rho(lag):
        return _corr(base, [cm.smooth_normal(31, "c", "k", t + lag, 1.0, 0.0) for t in starts])

    r = {lag: rho(lag) for lag in (0.1, 0.5, 1.0, 2.0, 3.0)}
    assert r[0.1] > 0.95
    assert r[0.5] == pytest.approx(0.736, abs=0.05)
    assert r[1.0] == pytest.approx(0.285, abs=0.05)
    assert abs(r[2.0]) < 0.04 and abs(r[3.0]) < 0.04
    assert r[0.1] > r[0.5] > r[1.0] > abs(r[2.0])


def test_smooth_normal_has_no_horizon():
    for t in (1.0e3, 1.0e5, 1.0e6, 1.0e7, 1.0e7 + 0.5, 1.0e9):
        for epoch in (0.0, EPOCH):
            v = cm.smooth_normal(5, "s", "k", t, 1.0, epoch)
            assert math.isfinite(v) and abs(v) < 9.0
    # Near t = 1e7 s the noise still varies bin to bin and is still N(0, 1).
    late = [cm.smooth_normal(5, "s", "k", EPOCH + 1.0e7 + i * 2.6180339887, 1.0, EPOCH)
            for i in range(5000)]
    assert len(set(late)) == len(late)
    assert abs(statistics.fmean(late)) < 0.08
    assert statistics.pvariance(late) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("kwargs,exc", [
    (dict(t=float("nan")), ValueError),
    (dict(t=float("inf")), ValueError),
    (dict(bin_s=0.0), ValueError),
    (dict(bin_s=-1.0), ValueError),
    (dict(epoch=float("nan")), ValueError),
    (dict(stream="a|b"), ValueError),
    (dict(stream="_u"), ValueError),
    (dict(stream=""), ValueError),
    (dict(salt=1.0), TypeError),
    (dict(salt=True), TypeError),
])
def test_smooth_normal_rejects_bad_inputs(kwargs, exc):
    args = dict(salt=1, stream="s", key="k", t=10.0, bin_s=1.0, epoch=0.0)
    args.update(kwargs)
    with pytest.raises(exc):
        cm.smooth_normal(**args)


# --------------------------------------------------------------------------- #
# Keyed Bernoulli helpers
# --------------------------------------------------------------------------- #

def test_keyed_uniform_pinned_and_pure():
    assert cm.keyed_uniform(99, "mule-0", 3) == 0.16995874104658593  # exact on every platform
    assert cm.u is cm.keyed_uniform
    keys = [(s, k, r) for s in (1, 2) for k in ("dev-0", "dev-1", 7) for r in range(6)]
    first = {q: cm.keyed_uniform(*q) for q in keys}
    again = {q: cm.keyed_uniform(*q) for q in reversed(keys)}
    assert first == again
    assert len(set(first.values())) == len(first)  # every part of the key matters
    assert all(0.0 <= v < 1.0 for v in first.values())


def test_keyed_uniform_is_uniform():
    salt = cm.ferry_salt(1, cm.SALT_AVAILABILITY)
    us = [cm.keyed_uniform(salt, f"dev-{i % 37}", i // 37) for i in range(20000)]
    assert statistics.fmean(us) == pytest.approx(0.5, abs=0.01)
    assert statistics.pvariance(us) == pytest.approx(1.0 / 12.0, abs=0.003)
    deciles = [0] * 10
    for v in us:
        deciles[int(v * 10)] += 1
    assert all(abs(c - 2000) < 200 for c in deciles), deciles


def test_keyed_bernoulli_rate_and_edges():
    salt = cm.ferry_salt(2, cm.SALT_BACKHAUL_LOSS)
    draws = [(f"mule-{i % 3}", i // 3) for i in range(15000)]
    assert not any(cm.keyed_bernoulli(0.0, salt, k, r) for k, r in draws)
    assert all(cm.keyed_bernoulli(1.0, salt, k, r) for k, r in draws)
    hits = [cm.keyed_bernoulli(0.3, salt, k, r) for k, r in draws]
    assert sum(hits) / len(hits) == pytest.approx(0.3, abs=0.02)
    assert hits == [cm.keyed_uniform(salt, k, r) < 0.3 for k, r in draws]
    # The inequality is strict: at p == u the outcome is False. With <=, p = 0
    # would be true whenever u is exactly 0.0.
    assert not any(cm.keyed_bernoulli(cm.keyed_uniform(salt, k, r), salt, k, r)
                   for k, r in draws[:200])


def test_keyed_draws_pair_by_mission_not_by_draw_count():
    """Design §1 D2: arm A docks every mission and arm B only on missions 2 and 4.
    Both see the same loss on missions 2 and 4. The legacy per-upload stream
    would hand B the draws of A's missions 1 and 2 instead."""
    salt = cm.ferry_salt(1234, cm.SALT_BACKHAUL_LOSS)
    p = 0.5
    arm_a = {m: cm.keyed_bernoulli(p, salt, "mule-0", m) for m in (1, 2, 3, 4)}
    arm_b = {m: cm.keyed_bernoulli(p, salt, "mule-0", m) for m in (2, 4)}
    assert all(arm_b[m] == arm_a[m] for m in arm_b)


@pytest.mark.parametrize("bad", [None, True, 1.0, "3"])
def test_keyed_uniform_needs_an_integer_round(bad):
    with pytest.raises(TypeError):
        cm.keyed_uniform(1, "dev-0", bad)


# --------------------------------------------------------------------------- #
# ContactChannel
# --------------------------------------------------------------------------- #

def test_contact_defaults_follow_the_spec():
    ch = _contact()
    assert ch.bands == ("wide", "medium", "narrow")
    assert ch.regime == "clean"
    assert (ch.interference_amp_db, ch.interference_sigma_db) == (1.0, 0.4)
    assert ch.interference_period_s == 60.0 and ch.period_multipliers == (1.0, 1.0, 1.0)
    assert ch.noise_bin_s == 1.0
    assert (ch.shadow_sigma_db, ch.shadow_corr_s, ch.shadow_keying) == (4.0, 7.4, "time")
    assert ch.epoch_s == EPOCH
    assert sorted(ch.phases) == [0.0, 1.0 / 3.0, 2.0 / 3.0]


def test_default_epoch_is_the_mission_clock_epoch():
    try:
        from hermes.l1 import mission_clock
    except ImportError:
        pytest.skip("hermes.l1.mission_clock (unit U1) is not there yet")
    assert cm.DEFAULT_EPOCH_S == mission_clock.SIM_EPOCH_S


def test_contact_pred_is_the_mean_alone():
    ch = _contact()
    for b in ch.bands:
        for d in (0.0, 12.5, 60.0):
            assert ch.pred_snr_db(b, d) == _mean_fn(b, d)


def test_contact_snr_is_mean_plus_shadow_plus_interference():
    ch = _contact()
    for i in range(50):
        t = EPOCH + 13.1 * i
        for b in ch.bands:
            expected = ch.pred_snr_db(b, 20.0) + ch.shadow_db(t, "dev-2") + ch.interference_db(t, b)
            assert ch.snr_db(t, b, 20.0, "dev-2") == expected
    assert ch.snr_by_band(EPOCH + 5.0, 20.0, "dev-2") == tuple(
        ch.snr_db(EPOCH + 5.0, b, 20.0, "dev-2") for b in ch.bands)


def test_contact_terms_follow_the_design_formula():
    ch = _contact(period_multipliers=(1.0, 2.0, 0.5))
    for i in range(40):
        t = EPOCH + 9.7 * i
        assert ch.shadow_db(t, "dev-1") == 4.0 * cm.smooth_normal(
            ch.salt, "shadow", "dev-1", t, 7.4, EPOCH)
        for b, m in zip(range(3), (1.0, 2.0, 0.5)):
            wave = 1.0 * math.sin(2.0 * math.pi * ((t - EPOCH) / (60.0 * m) + ch.phases[b]))
            assert ch.interference_db(t, b) == wave + 0.4 * cm.smooth_normal(
                ch.salt, "interference", ch.bands[b], t, 1.0, EPOCH)


FOUR_CLASSES = cm.CONTACT_BANDS + ("medium_wide",)   # U3's CLASSES_WITH_10MHZ: appended
SIXTHS = (1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0)


def test_a_class_keeps_its_interference_when_the_class_set_changes():
    """Design §1 D1 sweeps the class set (3 classes, or 4 with 10 MHz). The
    phase and the noise of ``I_b`` are keyed by class name, so the default
    classes keep their whole ``I_b(t)``, sine term included, in the four-class
    channel (U3's appended order), in a subset and in another order. Keying the
    phase by index re-drew it for every class when a class was added."""
    zero = lambda name, d: 0.0  # noqa: E731
    others = (FOUR_CLASSES, ("narrow", "wide"), ("narrow", "medium", "wide"),
              ("medium_wide", "medium"))
    for seed in range(40):
        salt = cm.ferry_salt(seed, cm.SALT_CONTACT)
        for regime in ("clean", "jittery"):
            three = cm.ContactChannel(zero, salt=salt, regime=regime)
            for bands in others:
                other = cm.ContactChannel(zero, salt=salt, regime=regime, bands=bands)
                kept = [name for name in bands if name in three.bands]
                assert [other.phases[other.band_index(n)] for n in kept] == [
                    three.phases[three.band_index(n)] for n in kept]
                for i in range(8):
                    t = EPOCH + 12.3 + 7.1 * i
                    for name in kept:
                        assert other.interference_db(t, name) == three.interference_db(t, name)
                    assert other.shadow_db(t, "dev-0") == three.shadow_db(t, "dev-0")


def test_an_extra_class_takes_a_gap_midpoint():
    """The 10 MHz class's phase is a midpoint between the default thirds, picked
    by its name: 1/6 of a period from its neighbours, the same in every set."""
    seen = set()
    for seed in range(300):
        salt = cm.ferry_salt(seed, cm.SALT_CONTACT)
        four = cm.ContactChannel(_mean_fn, salt=salt, bands=FOUR_CLASSES)
        assert sorted(four.phases[:3]) == [0.0, 1.0 / 3.0, 2.0 / 3.0]
        extra = four.phases[3]
        assert extra in SIXTHS
        seen.add(extra)
        ps = sorted(four.phases)
        gaps = [(ps[(i + 1) % 4] - ps[i]) % 1.0 for i in range(4)]
        assert min(gaps) == pytest.approx(1.0 / 6.0, abs=1e-12)
        alone = cm.ContactChannel(_mean_fn, salt=salt, bands=("medium_wide", "wide"))
        assert alone.phases == (extra, four.phases[0])
    assert seen == set(SIXTHS)


def test_the_four_class_link_keeps_the_default_classes_snr():
    """U2 x U3, the review's case: on U3's real links, the channel built on the
    four-class link gives wide, medium and narrow the same SNR as the one built
    on the default link, at every (t, band, link)."""
    contact_link = pytest.importorskip("hermes.l1.contact_link")
    link3 = contact_link.ContactLink(anchor_planar_m=60.0)
    link4 = contact_link.ContactLink(anchor_planar_m=60.0, classes=contact_link.CLASSES_WITH_10MHZ)
    assert tuple(link4.names[:3]) == tuple(link3.names)   # the 10 MHz class is appended
    for salt in (11, cm.ferry_salt(3, cm.SALT_CONTACT)):
        for regime in ("clean", "jittery"):
            three = cm.ContactChannel.from_link(link3, salt=salt, regime=regime)
            four = cm.ContactChannel.from_link(link4, salt=salt, regime=regime)
            assert four.bands == tuple(link4.names)
            for i in range(20):
                t = EPOCH + 12.3 + 7.1 * i
                for name in link3.names:
                    assert four.snr_db(t, name, 30.0, "dev-0") == three.snr_db(t, name, 30.0, "dev-0")


def test_contact_band_by_index_name_or_class_agree():
    ch = _contact()
    t = EPOCH + 42.0
    for i, name in enumerate(ch.bands):
        by_name = ch.snr_db(t, name, 30.0, "dev-0")
        assert ch.snr_db(t, i, 30.0, "dev-0") == by_name
        assert ch.snr_db(t, SimpleNamespace(name=name), 30.0, "dev-0") == by_name
    for bad, exc in (("ultra", ValueError), (3, ValueError), (-1, ValueError),
                     (True, TypeError), (1.0, TypeError)):
        with pytest.raises(exc):
            ch.snr_db(t, bad, 30.0, "dev-0")


def test_contact_channel_is_paired_across_arms_and_query_orders():
    """Two arms, each with its own instance, ask in different orders, one of them
    with unrelated queries in between: the same (t, band, link) gives the same value."""
    rng = random.Random(8)
    queries = [(EPOCH + rng.uniform(0.0, 3000.0), rng.choice((0, 1, 2)),
                rng.uniform(0.0, 60.0), rng.choice(("dev-0", "dev-1", "dev-2")))
               for _ in range(200)]
    arm_a, arm_b = _contact(seed=77), _contact(seed=77)
    a = {q: arm_a.snr_db(*q) for q in queries}
    b = {}
    for q in reversed(queries):
        arm_b.snr_db(q[0] + 1.3, (q[1] + 1) % 3, 5.0, "dev-9")
        b[q] = arm_b.snr_db(*q)
    assert a == b


def test_contact_channel_differs_across_seeds():
    one, two = _contact(seed=1), _contact(seed=2)
    ts = [EPOCH + 11.3 * i for i in range(200)]
    assert all(one.snr_db(t, "wide", 10.0, "dev-0") != two.snr_db(t, "wide", 10.0, "dev-0")
               for t in ts)


def test_contact_shadowing_is_shared_by_every_class():
    ch = _contact()
    for i in range(30):
        t = EPOCH + 17.0 * i
        x = ch.shadow_db(t, "dev-4")
        for b in ch.bands:
            residual = ch.snr_db(t, b, 25.0, "dev-4") - ch.pred_snr_db(b, 25.0) - ch.interference_db(t, b)
            assert residual == pytest.approx(x, abs=1e-9)


def test_contact_shadowing_statistics_and_time_correlation():
    ch = _contact(seed=3)
    starts = [EPOCH + 23.0 * i for i in range(5000)]   # > 2 correlation times apart
    xs = [ch.shadow_db(t, "dev-0") for t in starts]
    assert statistics.pstdev(xs) == pytest.approx(4.0, abs=0.2)
    near = [ch.shadow_db(t + 0.74, "dev-0") for t in starts]   # 0.1 correlation time
    far = [ch.shadow_db(t + 14.8, "dev-0") for t in starts]    # 2 correlation times
    assert _corr(xs, near) > 0.95
    assert abs(_corr(xs, far)) < 0.05
    other_link = [ch.shadow_db(t, "dev-1") for t in starts]
    assert abs(_corr(xs, other_link)) < 0.05


def test_contact_interference_is_periodic_and_bounded_without_noise():
    ch = _contact(interference_sigma_db=0.0, period_multipliers=(1.0, 1.5, 3.0))
    for b, m in zip(range(3), (1.0, 1.5, 3.0)):
        grid = [EPOCH + 0.25 * i for i in range(2000)]
        values = [ch.interference_db(t, b) for t in grid]
        assert max(values) <= 1.0 + 1e-12 and min(values) >= -1.0 - 1e-12
        assert max(values) == pytest.approx(1.0, abs=1e-3)
        for t in grid[::97]:
            assert ch.interference_db(t + 60.0 * m, b) == pytest.approx(ch.interference_db(t, b), abs=1e-9)


def test_contact_regimes():
    jittery = _contact(regime="jittery")
    assert (jittery.interference_amp_db, jittery.interference_sigma_db) == (5.0, 1.5)
    custom = _contact(interference_amp_db=2.0, interference_sigma_db=0.0)
    assert (custom.interference_amp_db, custom.interference_sigma_db) == (2.0, 0.0)
    with pytest.raises(ValueError):
        _contact(regime="stormy")


def test_contact_phases_are_a_seeded_shuffle():
    seen = set()
    for seed in range(40):
        phases = _contact(seed=seed).phases
        assert sorted(phases) == [0.0, 1.0 / 3.0, 2.0 / 3.0]
        assert _contact(seed=seed).phases == phases
        seen.add(phases)
    assert len(seen) == 6   # every permutation occurs


def test_contact_position_keyed_shadowing():
    """Critic C5/A6: keyed by (device, stop cell), fixed in time; the time
    variation then lives in I_b alone."""
    ch = _contact(shadow_keying="position")
    p = (10.0, 20.0, 0.0)
    x = ch.shadow_db(EPOCH, "dev-0", stop_pos=p)
    assert ch.shadow_db(EPOCH + 5000.0, "dev-0", stop_pos=p) == x          # frozen in time
    assert ch.shadow_db(EPOCH, "dev-0", stop_pos=(30.0, 5.0, 0.0)) == x    # same 37 m cell
    assert ch.shadow_db(EPOCH, "dev-0", stop_pos=(40.0, 20.0, 0.0)) != x   # next cell
    assert ch.shadow_db(EPOCH, "dev-0", stop_pos=(-1.0, 20.0, 0.0)) != x   # negative side
    assert ch.shadow_db(EPOCH, "dev-1", stop_pos=p) != x                   # another device
    assert ch.snr_db(EPOCH + 1.0, 0, 5.0, "dev-0", stop_pos=p) == (
        ch.pred_snr_db(0, 5.0) + x + ch.interference_db(EPOCH + 1.0, 0))
    with pytest.raises(ValueError):
        ch.shadow_db(EPOCH, "dev-0")
    cells = [ch.shadow_db(EPOCH, f"dev-{i % 50}", stop_pos=(37.0 * (i // 50), 0.0))
             for i in range(5000)]
    assert statistics.pstdev(cells) == pytest.approx(4.0, abs=0.2)


def test_contact_time_keyed_shadowing_moves_with_time():
    ch = _contact()
    assert ch.shadow_db(EPOCH, "dev-0") != ch.shadow_db(EPOCH + 30.0, "dev-0")
    # Time keying ignores the stop position.
    assert (ch.shadow_db(EPOCH + 3.0, "dev-0", stop_pos=(0.0, 0.0))
            == ch.shadow_db(EPOCH + 3.0, "dev-0", stop_pos=(90.0, 90.0)))


def test_contact_channel_has_no_horizon():
    ch = _contact()
    for t in (EPOCH + 1.0e7, 1.0e7, EPOCH + 1.0e7 + 0.37):
        assert math.isfinite(ch.snr_db(t, "narrow", 50.0, "dev-3"))


def test_contact_from_link_duck_types_the_contact_link():
    link = SimpleNamespace(mean_snr_db=_mean_fn, shadow_sigma_db=3.0,
                           names=("wide", "medium", "narrow"))
    ch = cm.ContactChannel.from_link(link, salt=5)
    assert ch.shadow_sigma_db == 3.0 and ch.bands == link.names
    assert ch.pred_snr_db("medium", 7.0) == _mean_fn("medium", 7.0)
    with pytest.raises(ValueError):
        cm.ContactChannel.from_link(link, salt=5, shadow_sigma_db=4.0)
    with pytest.raises(ValueError):
        cm.ContactChannel.from_link(link, salt=5, bands=("wide",))
    bare = cm.ContactChannel.from_link(SimpleNamespace(mean_snr_db=_mean_fn), salt=5)
    assert bare.shadow_sigma_db == cm.SHADOW_SIGMA_DB and bare.bands == cm.CONTACT_BANDS


def test_contact_channel_on_the_real_contact_link_gives_90_percent_at_the_edge():
    """U2 x U3: R(b) is the range with 90 % availability at the edge (design §1 D1),
    because the link's mean carries a 1.2816 sigma margin and this channel adds
    the shadowing with that sigma (interference off here)."""
    contact_link = pytest.importorskip("hermes.l1.contact_link")
    link = contact_link.ContactLink(anchor_planar_m=60.0)
    ch = cm.ContactChannel.from_link(link, salt=cm.ferry_salt(3, cm.SALT_CONTACT),
                                     interference_amp_db=0.0, interference_sigma_db=0.0)
    assert ch.bands == tuple(link.names) and ch.shadow_sigma_db == link.shadow_sigma_db
    for b in ch.bands:
        edge = link.range_planar_m(b)
        assert ch.pred_snr_db(b, edge) == link.mean_snr_db(b, edge)
        up = [ch.snr_db(EPOCH + 17.3 * i, b, edge, f"dev-{i % 7}") >= link.snr_floor_db
              for i in range(5000)]
        assert sum(up) / len(up) == pytest.approx(0.90, abs=0.02)


@pytest.mark.parametrize("kwargs,exc", [
    (dict(bands=("wide", "wide")), ValueError),
    (dict(bands=()), ValueError),
    (dict(shadow_keying="space"), ValueError),
    (dict(shadow_sigma_db=-1.0), ValueError),
    (dict(interference_period_s=0.0), ValueError),
    (dict(period_multipliers=(1.0, 2.0)), ValueError),
    (dict(period_multipliers=(1.0, 0.0, 1.0)), ValueError),
    (dict(noise_bin_s=0.0), ValueError),
    (dict(epoch_s=float("inf")), ValueError),
])
def test_contact_channel_rejects_bad_parameters(kwargs, exc):
    with pytest.raises(exc):
        _contact(**kwargs)


def test_contact_channel_rejects_bad_queries():
    with pytest.raises(TypeError):
        cm.ContactChannel(42, salt=1)
    ch = _contact()
    with pytest.raises(ValueError):
        ch.snr_db(EPOCH, "wide", -1.0, "dev-0")
    with pytest.raises(ValueError):
        ch.snr_db(float("nan"), "wide", 1.0, "dev-0")


def test_contact_describe_is_json_able():
    d = _contact(seed=4).describe()
    assert json.loads(json.dumps(d)) == d
    assert d["model"] == "contact" and d["shadow_keying"] == "time" and len(d["phases"]) == 3


# --------------------------------------------------------------------------- #
# BackhaulChannel
# --------------------------------------------------------------------------- #

def test_backhaul_period_is_n_missions_times_t_nom():
    assert cm.backhaul_period_s(4, 219.0) == 876.0
    assert cm.backhaul_period_s(1, 36.0) == 36.0
    for bad in ((0, 219.0), (4, 0.0), (4, float("nan"))):
        with pytest.raises(ValueError):
            cm.backhaul_period_s(*bad)


def test_backhaul_fixed_band_is_argmax_g():
    for seed in range(60):
        ch = _backhaul(seed=seed)
        assert len(ch.gains_db) == 3 and all(0.0 <= g < 3.0 for g in ch.gains_db)
        assert ch.fixed_band() == max(range(3), key=lambda c: ch.gains_db[c])
        # argmax g is the noise-free long-run best carrier: the time average
        # of the SNR over one period is base + g_c.
        quiet = _backhaul(seed=seed, sigma_db=0.0)
        n = 400
        means = [sum(quiet.snr_db(EPOCH + (i + 0.5) * quiet.period_s / n, c) for i in range(n)) / n
                 for c in range(3)]
        assert max(range(3), key=lambda c: means[c]) == ch.fixed_band()
        for c in range(3):
            assert means[c] == pytest.approx(quiet.pred_snr_db(c), abs=1e-9)


def test_backhaul_regimes_follow_the_legacy_values():
    clean, jittery = _backhaul(regime="clean"), _backhaul(regime="jittery")
    assert (clean.base_db, clean.amp_db, clean.sigma_db) == (12.0, 1.0, 0.4)
    assert (jittery.base_db, jittery.amp_db, jittery.sigma_db) == (6.0, 5.0, 1.5)
    assert clean.gains_db == jittery.gains_db and clean.phases == jittery.phases
    with pytest.raises(ValueError):
        _backhaul(regime="stormy")


def test_backhaul_snr_follows_the_design_formula():
    ch = _backhaul(seed=9, regime="jittery")
    assert ch.pred_snr_db(1) == ch.base_db + ch.gains_db[1]
    for i in range(60):
        t = EPOCH + 29.3 * i
        for c in range(3):
            wave = 5.0 * math.sin(2.0 * math.pi * ((t - EPOCH) / ch.period_s + ch.phases[c]))
            expected = 6.0 + ch.gains_db[c] + wave + 1.5 * cm.smooth_normal(
                ch.salt, "backhaul", c, t, 1.0, EPOCH)
            assert ch.snr_db(t, c) == expected
            assert ch.loss_probability(t, c) == cm.loss_from_snr(expected)
        assert ch.snr_all(t) == tuple(ch.snr_db(t, c) for c in range(3))


def test_backhaul_is_periodic_in_p_bh_without_noise():
    ch = _backhaul(seed=2, regime="jittery", sigma_db=0.0)
    for i in range(30):
        t = EPOCH + 37.7 * i
        for c in range(3):
            assert ch.snr_db(t + ch.period_s, c) == pytest.approx(ch.snr_db(t, c), abs=1e-9)


def test_backhaul_phases_and_gains_are_seeded():
    seen = set()
    for seed in range(40):
        ch = _backhaul(seed=seed)
        assert sorted(ch.phases) == [0.0, 1.0 / 3.0, 2.0 / 3.0]
        assert _backhaul(seed=seed).gains_db == ch.gains_db
        seen.add(ch.phases)
    assert len(seen) == 6
    assert _backhaul(seed=1).gains_db != _backhaul(seed=2).gains_db


def test_backhaul_is_paired_and_query_order_free():
    ts = [EPOCH + 3.7 * i for i in range(300)]
    one, two = _backhaul(seed=5, regime="jittery"), _backhaul(seed=5, regime="jittery")
    a = {(t, c): one.snr_db(t, c) for t in ts for c in range(3)}
    b = {(t, c): two.snr_db(t, c) for t in reversed(ts) for c in (2, 1, 0)}
    assert a == b
    other = _backhaul(seed=6, regime="jittery")
    assert all(other.snr_db(t, 0) != a[(t, 0)] for t in ts)


def test_backhaul_select_carrier():
    ch = _backhaul(seed=11, regime="jittery")
    ctrl = AdaptiveChannelController(channel_use_cost=(0.0, 0.0, 0.0), switch_cost=0.5)
    current = expected = -1
    for k in range(12):
        t = EPOCH + (k + 0.5) * 219.0
        assert ch.select_carrier(t, adaptive=False, current=current) == ch.fixed_band()
        expected = ctrl.select(ch.snr_all(t), expected)
        current = ch.select_carrier(t, adaptive=True, current=current)
        assert current == expected
    custom = AdaptiveChannelController(channel_use_cost=(0.0, 0.0, 0.0), switch_cost=100.0)
    assert ch.select_carrier(EPOCH + 5.0, adaptive=True, current=2, controller=custom) == 2


def test_backhaul_loss_magnitudes_match_critic_a4():
    """Critic A4: about 16 % per upload jittery at the fixed carrier (today: a
    flat 2 %), about 0.4 % clean; H3's controller keeps its loss low."""
    t_nom, n_missions = 219.0, 4
    period = cm.backhaul_period_s(n_missions, t_nom)
    uploads = [EPOCH + (k + 0.5) * t_nom for k in range(n_missions)]
    result = {}
    for regime in ("clean", "jittery"):
        fixed_avg, adaptive = [], []
        for seed in range(400):
            ch = _backhaul(seed=seed, regime=regime, period_s=period)
            b = ch.fixed_band()
            fixed_avg.append(sum(ch.loss_probability(EPOCH + (i + 0.5) * period / 100, b)
                                 for i in range(100)) / 100)
            current, losses = -1, []
            for t in uploads:
                current = ch.select_carrier(t, adaptive=True, current=current)
                losses.append(ch.loss_probability(t, current))
            adaptive.append(sum(losses) / len(losses))
        result[regime] = (statistics.fmean(fixed_avg), statistics.fmean(adaptive))
    assert 0.13 <= result["jittery"][0] <= 0.20, result
    assert result["clean"][0] < 0.008, result
    assert 0.005 <= result["jittery"][1] <= 0.04, result
    assert result["jittery"][1] < result["jittery"][0]
    assert result["clean"][1] < 0.008, result


@pytest.mark.parametrize("kwargs,exc", [
    (dict(period_s=0.0), ValueError),
    (dict(n_carriers=0), ValueError),
    (dict(sigma_db=-0.1), ValueError),
    (dict(noise_bin_s=-1.0), ValueError),
    (dict(salt=2.5), TypeError),
])
def test_backhaul_rejects_bad_parameters(kwargs, exc):
    args = dict(salt=1, period_s=100.0)
    args.update(kwargs)
    with pytest.raises(exc):
        cm.BackhaulChannel(**args)


def test_backhaul_rejects_bad_carriers():
    ch = _backhaul()
    for bad, exc in ((3, ValueError), (-1, ValueError), (True, TypeError), (0.0, TypeError)):
        with pytest.raises(exc):
            ch.snr_db(EPOCH, bad)


def test_backhaul_describe_is_json_able():
    d = _backhaul(seed=4, regime="jittery").describe()
    assert json.loads(json.dumps(d)) == d
    assert d["model"] == "backhaul" and d["fixed_band"] == _backhaul(seed=4).fixed_band()


# --------------------------------------------------------------------------- #
# Causal RF prior (critic B4)
# --------------------------------------------------------------------------- #

def test_rf_prior_default_is_the_supervisor_default():
    import inspect

    from hermes.mule.mule_main import MuleSupervisor

    default = inspect.signature(MuleSupervisor.__init__).parameters["rf_prior_snr_db"].default
    assert DEFAULT_RF_PRIOR_SNR_DB == default == 20.0


def test_rf_prior_is_20_db_before_the_first_upload():
    producer = RFPriorProducer()
    assert producer.prior_snr_db() == 20.0
    assert producer.prior_snr_db(carrier=1) == 20.0
    assert producer.store.snapshot() == ()


def test_rf_prior_follows_the_observed_uploads_causally():
    store = RFPriorStore()
    producer = RFPriorProducer(store)
    seen = {}
    observations = [(0, 7.5, EPOCH + 100.0), (0, 9.5, EPOCH + 320.0),
                    (2, 12.5, EPOCH + 540.0), (1, -3.0, EPOCH + 760.0)]
    for carrier, snr, t in observations:
        record = producer.observe_upload(carrier, snr, t)
        assert record == RFPrior(band=carrier, last_good_snr_db=snr, observed_at=t)
        seen[carrier] = snr
        # Only uploads so far count: the mean of each carrier's last observation.
        assert producer.prior_snr_db() == pytest.approx(sum(seen.values()) / len(seen))
        assert producer.prior_snr_db(carrier=carrier) == snr
        assert store.read(carrier) == record   # the scheduler's read API sees it
    assert producer.prior_snr_db(carrier=0) == 9.5


def test_rf_prior_store_read_api_is_unchanged():
    public = {n for n in dir(RFPriorStore)
              if not n.startswith("_") and callable(getattr(RFPriorStore, n))}
    assert public == {"snapshot", "read", "mean_snr_db"}
    assert RFPriorStore().mean_snr_db() == 0.0   # the store's own empty value is untouched


@pytest.mark.parametrize("args,exc", [
    ((0, float("nan"), 1.0), ValueError),
    ((0, 10.0, float("inf")), ValueError),
    ((-1, 10.0, 1.0), ValueError),
    ((True, 10.0, 1.0), TypeError),
    ((1.5, 10.0, 1.0), TypeError),
])
def test_rf_prior_rejects_bad_observations(args, exc):
    with pytest.raises(exc):
        RFPriorProducer().observe_upload(*args)

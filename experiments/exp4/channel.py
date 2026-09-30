"""EX-4.3 — RF channel environment for the L1 experiment (arm H3).

FeRRy Phase 3 (unit U2): this module is now a re-export shim. ``ChannelModel``,
``loss_from_snr``, ``BackhaulPlan`` and ``backhaul_plan`` moved verbatim to
:mod:`hermes.l1.channel_model`: ``hermes`` must not import ``experiments``
(finding A-01), and the ferry code that needs them lives in ``hermes``. The
names below are the same objects, so every existing import (the driver's
``--l1-channel`` path, the tests, the re-derivation of recorded traces) keeps
working and computes the same numbers; ``tests/golden/test_golden_channel.py``
pins them at afa9526. The model is maintained there, together with the
seconds-axis channels of Phase 3, which use other names (``ContactChannel``,
``BackhaulChannel``).

The names are re-exported, not wrapped. ``backhaul_plan`` looks up
``loss_from_snr``, ``best_average_band`` and ``AdaptiveChannelController`` in
:mod:`hermes.l1.channel_model`, so a test that patches one of them must patch
it there. Rebinding it on this module no longer reaches ``backhaul_plan``, as
it did before the move; nothing in the repo relied on that.

Models the mule->base-station backhaul as a set of RF channels whose
effective SNR varies over the mission sequence, and turns a channel choice
into a per-mission backhaul-loss probability. This is what lets adaptive
channel selection (L1, arm H3) *matter*: a good channel -> high SNR -> low
loss -> more rounds close.

Validity discipline (learned from the jittery remediation):

* **Bands cross over.** Each band peaks at a different time (distinct
  phases), so **no single fixed band is best throughout** — otherwise the
  static-best baseline (H2) would tie the adaptive controller (H3) and L1
  would have no honest value. Adaptation only helps when the best band
  changes, which is the realistic time-varying condition.
* **Fair baseline.** H2 uses ``best_average_band`` — the band a deployer
  picks from historical averages *without* real-time tracking (Exp 2's
  "Expected fixed"), NOT the retrospective per-instant oracle.
* **Same conditions.** H2 and H3 face the identical seeded SNR trace.
* **Clean ~ no L1 benefit.** Under clean links all bands sit high and
  stable, so fixed ~ adaptive and L1's effect is (correctly) negligible;
  the benefit, if any, appears under jittery.

The whole model is deterministic in the paired seed and runs in the driver
(in-process), producing a per-mission loss schedule the cluster applies —
so no cross-process channel coordination is needed.
"""

from __future__ import annotations

from hermes.l1.channel_model import (
    BackhaulPlan,
    ChannelModel,
    backhaul_plan,
    loss_from_snr,
)
# The model's own imports were importable from here before the move; keep them.
from hermes.l1.channel_utility import AdaptiveChannelController, best_average_band

__all__ = [
    "AdaptiveChannelController",
    "BackhaulPlan",
    "ChannelModel",
    "backhaul_plan",
    "best_average_band",
    "loss_from_snr",
]

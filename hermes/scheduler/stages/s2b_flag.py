"""Stage 2B — FL_READY flag / utility threshold.

Design §2.1::

    Thresholding incoming FL_READY_ADV against FL_Threshold.
    Note: S2B is *verified* on mule but *computed* on device —
    scheduler consumes the pre-computed utility.

The mule never recomputes utility from raw per-device features (that
would violate §7 principle 10 — "mule is a transport agent, never
inspects payloads"). It re-applies the threshold to the device's own
number so a miscalibrated device that claims ``utility=999`` for a
clearly-not-ready state is still rejected here if the threshold is
somehow honoured globally.
"""

from __future__ import annotations

from hermes.types import FLReadyAdv


# Default matches the AC-GAN pipeline's existing quality bar.
DEFAULT_FL_THRESHOLD: float = 0.60


def passes_fl_threshold(
    adv: FLReadyAdv,
    fl_threshold: float = DEFAULT_FL_THRESHOLD,
) -> bool:
    """Strict '>' per design §6.8::

        FL_OPEN(i) = (utility(i) > FL_Threshold) ∧ ...
    """
    return adv.utility > fl_threshold

"""L1 RF-channel selector (channel-only DDQN).

Design §2.6 + §7 principle 5: with the trajectory head gone, L1 reduces
to a single discrete-action DDQN picking one of the 3 RF bands
(3.32 / 3.34 / 3.90 GHz per slide 26). The scheduler sends only the
target waypoint DOWN; L1 chooses the channel and drives the radio.

Public surface:

* :class:`ChannelDDQN` — inference-only actor on the NUC.
* :class:`RFPriorStore` — read-only API for the selector (§2.1).
"""

from __future__ import annotations

from .channel_ddqn import CHANNEL_FREQS_GHZ, ChannelDDQN
from .rf_prior import RFPrior, RFPriorStore

__all__ = [
    "CHANNEL_FREQS_GHZ",
    "ChannelDDQN",
    "RFPrior",
    "RFPriorStore",
]

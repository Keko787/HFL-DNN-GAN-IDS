"""Stage 2A — On-contact readiness gate.

Design §2.1::

    Gate devices on contact (``FL_READY == True``, verified locally)

and §7 principle 10::

    Eligibility is computed locally on the edge device. The mule is a
    transport agent — it never inspects payloads.

S2A is the mule-side *verification* step: when an ``FL_READY_ADV`` lands,
we accept its self-declared state but we refuse to scoreboard anything
that arrived with a non-``FL_OPEN`` state or a stale timestamp.

No numerical scoring happens here — this is a binary admit/reject gate.
The *numeric* S2B check (utility threshold) runs next in
``s2b_flag``.
"""

from __future__ import annotations

from hermes.types import FLReadyAdv


def is_on_contact_ready(
    adv: FLReadyAdv,
    now: float,
    freshness_window_s: float = 5.0,
) -> bool:
    """Admit / reject a freshly-arrived ``FL_READY_ADV``.

    Rules:
    * The advert's own ``state`` must permit opening a session
      (``FLReadyAdv.is_eligible`` -> ``FLState.can_open_session``).
    * The advert must not be stale. A freshness window of a few seconds
      guards against replayed beacons from a previous mission round.
    * ``issued_at == 0.0`` is treated as "unset" and always passes the
      freshness check — used by unit tests that don't care about clocks.
    """
    if not adv.is_eligible():
        return False
    if adv.issued_at == 0.0:
        return True
    if freshness_window_s <= 0.0:
        return True
    return (now - adv.issued_at) <= freshness_window_s

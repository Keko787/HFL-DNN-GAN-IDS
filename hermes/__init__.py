"""HERMES — Layer 2/3 FL scheduler implementation.

Top-level package for the new programs introduced by the HERMES design:

* ``hermes.types``        — shared dataclasses crossing tier boundaries.
* ``hermes.transport``    — pluggable transports (loopback today, real later).
* ``hermes.cluster``      — HFLHostCluster (Phase 1).

See ``DeveloperDocs/HERMES_FL_Scheduler_Design.md`` and
``DeveloperDocs/HERMES_FL_Scheduler_Implementation_Plan.md`` for context.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

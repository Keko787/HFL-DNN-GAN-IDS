"""FLScheduler — L2 Scheduler on the mule NUC.

Public surface is intentionally small: callers outside the scheduler
package talk to :class:`FLScheduler` only. The stages under
``hermes.scheduler.stages`` are pure functions — exposed for testing, not
for direct composition.

Design refs:
* HERMES_FL_Scheduler_Design.md §2.1 FLScheduler responsibilities
* HERMES_FL_Scheduler_Design.md §5.1 FLScheduler loop
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 4
"""

from __future__ import annotations

from .fl_scheduler import FLScheduler, FLSchedulerError

__all__ = [
    "FLScheduler",
    "FLSchedulerError",
]

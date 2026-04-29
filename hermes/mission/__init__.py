"""Mission-scope programs on the mule-NUC and edge device.

Phase 2 deliverables:
* ``HFLHostMission`` — mule-side FL server (one per mule)
* ``ClientMission``  — device-side FL client (one per edge device)
* ``partial_fedavg`` — in-mission weight merger
* utility formulas    — Performance_score, diversity_adjusted, utility

Design ref: HERMES_FL_Scheduler_Design.md §5.3, §6.3
"""

from .partial_fedavg import PartialFedAvgError, partial_fedavg
from .utility import (
    performance_score,
    diversity_adjusted,
    utility,
    cosine_similarity,
)
from .host_mission import HFLHostMission, MissionSessionError
from .client_mission import ClientMission, LocalTrainFn, LocalTrainResult

__all__ = [
    "HFLHostMission",
    "MissionSessionError",
    "ClientMission",
    "LocalTrainFn",
    "LocalTrainResult",
    "partial_fedavg",
    "PartialFedAvgError",
    "performance_score",
    "diversity_adjusted",
    "utility",
    "cosine_similarity",
]

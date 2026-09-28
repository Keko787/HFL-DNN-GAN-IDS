"""The ``hermes_rl`` drone relay prototype, vendored at commit a8a453f.

Exports the environment API only; it needs numpy and nothing else. The
trainer (``train_dqn``, needs torch) and the visual demo (``drone_demo``,
needs pygame) are modules to run, not imports. See README.md here for
provenance and what changed.
"""

from .drone_env import (
    BaseStationConfig,
    DroneDataRelayEnv,
    EnvConfig,
    JobConfig,
    ScenarioRandomization,
    SensorConfig,
    overloaded_config,
)

__all__ = [
    "BaseStationConfig",
    "DroneDataRelayEnv",
    "EnvConfig",
    "JobConfig",
    "ScenarioRandomization",
    "SensorConfig",
    "overloaded_config",
]

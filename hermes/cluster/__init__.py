"""HFLHost-Cluster — cluster-scope FL coordinator (Tier 2).

Phase 1 of the HERMES build. Owns:

* the authoritative device registry,
* per-mule disjoint mission slicing,
* cross-mule FedAvg over partial aggregates,
* dock-side ingestion and dispatch.

θ_gen + synth-sample generation are *hosted* here too (slide 24 / design
§2.5), but the actual GAN code is reused unchanged from the existing
``App/TrainingApp/HFLHost`` pipeline — the cluster wires them in.
"""

from .device_registry import DeviceRegistry, RegistrySnapshot
from .cross_mule_fedavg import cross_mule_fedavg, FedAvgError
from .host_cluster import HFLHostCluster

__all__ = [
    "DeviceRegistry",
    "RegistrySnapshot",
    "cross_mule_fedavg",
    "FedAvgError",
    "HFLHostCluster",
]

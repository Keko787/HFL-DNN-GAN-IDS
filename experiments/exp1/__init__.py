"""Experiment 1 — Federated vs Centralized at fixed radio.

Sub-package layout:

* :mod:`experiments.exp1.protocol` — wire format (control frames in
  JSON, bulk frames as raw bytes); both directions.
* :mod:`experiments.exp1.topology` — server-side topology config
  (explicit JSON, explicit CLI, or discovery mode).
* :mod:`experiments.exp1.server` — server entry point. Runs the trial
  grid; one-sided timing.
* :mod:`experiments.exp1.client` — client entry point. Connects,
  registers, ships bytes when told.

The server + client pair is **testbed-agnostic**: change the topology
config and the same scripts run on a single Linux box (subprocess +
optional ``tc/netem`` on ``lo``), AERPAW (5 AVNs), Chameleon (5 KVMs),
or any other testbed that provides routable IPs.
"""

"""HERMES paper-experiment scaffolding.

Sibling to ``hermes/``. Phases 0–7 of the system plan ship the
*artefact* HERMES; this package ships the *measurements* that exercise
the artefact and produce paper figures.

See ``DeveloperDocs/HERMES_Experiments_Implementation_Plan.md`` for the
chunk-by-chunk plan. Sub-packages:

* ``experiments.runner`` — Chunk EX-0: shared trial-grid + CSV log +
  runner. Both Experiment 1 and Experiment 3 drive their trials through
  this harness.
* ``experiments.exp1`` — Federated vs Centralized at fixed radio
  (deferred — Chunks EX-1.1 .. EX-1.5).
* ``experiments.exp3`` — A1/A2/A3/A4 scheduling ablation (deferred —
  Chunks EX-3.1 .. EX-3.5).
* ``experiments.analysis`` — Jupyter notebooks for stats + figures
  (deferred — Chunks EX-1.5 / EX-3.5).
"""

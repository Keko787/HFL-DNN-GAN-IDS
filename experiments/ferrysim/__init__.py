"""FerrySim: the stack's own trial, run in process, where the pair score practises.

FeRRy Phase 5 (the user's decision 2 (a)): the real system (the Exp 4 driver,
the per-role JSON, the cluster, mule and device services) in one process, with
stand-ins for the devices' local training and their service loops, which do
not run (so the row's device-serve columns are harness artifacts: ``inprocess``,
the orchestrator's resolution R25). It lives here rather than in
``hermes/scheduler/selector/`` (build plan L496) because it needs both the
experiments' driver and ``hermes.l1``, and ``hermes`` never imports
``experiments`` (L524) while the scheduler never imports ``hermes.l1``.

Modules (unit U8a): ``inprocess`` (the in-process orchestrator, a copy of the
golden harness's), ``cells`` (the cells and the seed streams), ``episode``
(one episode and how it is read), ``reward`` (the derived reward, F·hand and
E3's), ``evaluate`` (held-out returns and worker processes) and ``headroom``
(the per-sortie oracle and the headroom report). Unit U8b adds the trainer,
the checkpoints, the report and the command line.

Importing the package loads none of them, so ``hermes`` paths never pay for
it; import the module you need.
"""

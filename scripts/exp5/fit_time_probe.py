"""Session-TTL pilot: the real model's local-fit time with N devices fitting at once.

The pilot plan (Experiment_4_Run_Guide.md section 2.6) sets --session-ttl-s to
at least 2x the 95th percentile of the real model's train_offline time at the
trial's concurrency. No event records that time, so this probe measures it
directly. It builds the canonical CICIoT2023 task for N devices as the driver
does (load_ciciot_task_canonical), and starts N worker processes for each of
--trials trials side by side (the launcher passes the number of trials it will
run at once at this N). Each worker builds the device's own callback,
experiments.exp4.model_task.make_local_train_fn, exactly what
hermes/processes/device.py builds on the real-model path, and times --fits
consecutive calls; a barrier releases all workers together. A device's first
fit in a trial includes TensorFlow's graph tracing, and so does each worker's
first call here, so it stays in the percentiles.

It reads the dataset and refuses a synthetic task. It writes the shards to a
temporary folder it deletes, and prints (and with --out writes) a JSON report.
Run it under the same thread settings as the sweep (scripts/exp5/launch.py
sets them), since they change fit time.

    python scripts/exp5/fit_time_probe.py --N 6 --fits 4 --trials 4 --out ttl_n6.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _worker(shard, seed, fits, arch, barrier, out):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from experiments.exp4.model_task import initial_theta, load_xy, make_local_train_fn

    X, y = load_xy(shard)
    kw = {} if arch is None else {"arch": arch}
    fn = make_local_train_fn(X, y, input_dim=X.shape[1], epochs=1, batch_size=64,
                             seed=seed, **kw)
    theta = initial_theta(X.shape[1], seed=0, **kw)
    barrier.wait()
    times = []
    for _ in range(fits):
        t = time.perf_counter()
        fn(theta, None)
        times.append(time.perf_counter() - t)
    out.put({"rows": int(len(y)), "fit_s": times})


def _quantile(values, p):
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--N", type=int, required=True, help="Devices per trial.")
    ap.add_argument("--fits", type=int, default=4, help="Fits per device (default 4, the missions).")
    ap.add_argument("--trials", type=int, default=1, help="Trials side by side (N x trials workers).")
    ap.add_argument("--seed", type=int, default=7, help="Task seed of the first trial.")
    ap.add_argument("--arch", default=None, help="--model-arch (default: the canonical model).")
    ap.add_argument("--out", type=Path, default=None, help="Also write the report here (JSON).")
    a = ap.parse_args(argv)

    from experiments.exp4.model_task import (
        default_ciciot_dir, load_ciciot_task_canonical, save_xy)

    if default_ciciot_dir() is None:
        sys.exit("CICIoT2023 not found: refusing to time a synthetic task")
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="fitprobe_") as tmp:
        shards = []
        for k in range(a.trials):
            task = load_ciciot_task_canonical(n_devices=a.N, seed=a.seed + k)
            if task.is_synthetic:
                sys.exit("the loader fell back to a synthetic task: refusing")
            for i, (X, y) in enumerate(task.device_shards):
                path = Path(tmp) / f"t{k}_d{i}.npz"
                save_xy(path, X, y)
                shards.append(str(path))
        barrier, out = ctx.Barrier(len(shards)), ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(s, j, a.fits, a.arch, barrier, out))
                 for j, s in enumerate(shards)]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        results = [out.get() for _ in procs]
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0

    every = sorted(t for r in results for t in r["fit_s"])
    later = sorted(t for r in results for t in r["fit_s"][1:])
    p95 = _quantile(every, 0.95)
    report = {
        "N": a.N, "trials_side_by_side": a.trials, "workers": len(procs), "fits_each": a.fits,
        "arch": a.arch or "ciciot", "rows_per_device": sorted({r["rows"] for r in results}),
        "fit_s": {"p50": round(_quantile(every, 0.5), 3), "p95": round(p95, 3),
                  "max": round(every[-1], 3),
                  "mean_after_first": round(statistics.mean(later), 3) if later else None},
        "ttl_floor_s": round(2 * p95, 2),
        "threads": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
            "OPENBLAS_NUM_THREADS")},
        "probe_wall_s": round(wall, 1),
    }
    text = json.dumps(report, indent=1)
    print(text)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

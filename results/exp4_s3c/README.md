# S3c pilot — checklist §5.0

Two configurations differing by **exactly one flag** (`--mission-window-adaptation`),
20 paired seeds, deadline enforcement on in both.

| File | What |
|---|---|
| `off.csv` / `on.csv` | per-trial rows, 20 each, all `status=ok` |
| `off_traces/` `on_traces/` | retained per-contact events + configs (`--keep-event-traces`) |
| `*.log` | runner output |

**Result and interpretation: [`../../DeveloperDocs/HERMES_PreRerun_Checklist.md`](../../DeveloperDocs/HERMES_PreRerun_Checklist.md) §5.0a.**
Short version: non-null but narrow and **transient** — S3c reaches a workable window *faster*
rather than reaching a better one. Does not survive multiplicity correction; mechanism verified
independently from the traces.

Reproduce:

```bash
bash experiments/exp4/run_s3c_pilot.sh
python -m experiments.exp4.analyze_s3c_pilot --dir results/exp4_s3c
```

> **At matrix scale, reconsider committing traces.** This pilot is 40 trials → 640 files / 2.3 MB,
> which is worth keeping in-tree as the evidence behind §5.0a. A 240-trial matrix would be ~3,400
> files; archive those alongside the CSVs instead of committing them, or keep only the arms a
> baseline will actually be scored against.

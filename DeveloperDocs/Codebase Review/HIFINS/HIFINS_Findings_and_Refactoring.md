# HIFINS — Findings and Refactoring Plan

**Status:** Review only. No project code has been modified.
**Scope:** `App/`, `Config/`, `Analysis/`, `AppSetup/`, `experiments/`, `FlightFramework/`
— 46.8 K LOC
**Companions:** [`../00_Critical_Problem_Areas.md`](../00_Critical_Problem_Areas.md) ·
[`../01_Refactoring_Strategy.md`](../01_Refactoring_Strategy.md) ·
[`HIFINS_Production_Code.md`](HIFINS_Production_Code.md) ·
[architecture](../../architecture%20documents/HIFins/HIFINS_Architecture.md)

---

## 0. Verdict

This is the half of the repository that can actually train a model, and it is the half
nobody can safely change.

`Config/ModelTrainingConfig/` is 21,278 LOC across 47 classes with **zero test
coverage**. The 500-line `fit` methods carry the September-2025 mode-collapse fixes in
some copies and not others, and no mechanism exists to tell which. Meanwhile 21 CLI
hyperparameters that the code advertises are read by nothing, so any sweep run through
the documented interface produced identical models under different recorded settings.

The root cause is a single pattern choice — **configuration-as-code**: each combination
of (model type × sub-model × training area) is a separate class with its own copy of the
training loop, rather than one parameterised trainer driven by a config object. Every
other finding here is downstream of that.

The good news is that the target shape is unambiguous, the seams to build against
already exist in `hermes/`, and roughly 9 K LOC can be deleted in phase 0 before any
risky work starts.

---

## 1. Findings

Cross-cutting entries are detailed in the main register; this document adds mechanism
and the plan.

### F-01 (P0) — 21 AC-GAN CLI hyperparameters are never read

**Mechanism.** `ArgumentConfigLoad.py:76-96` declares 21 `--AC_*` flags. Repository-wide
search finds each exactly once — at its own declaration. The trainer's optimizers are
built from literals at `ACGANCentralTrainingConfig.py:109-118`:

```python
lr_schedule_gen  = ExponentialDecay(initial_learning_rate=0.00012, decay_steps=10000,
                                    decay_rate=0.98, staircase=False)
lr_schedule_disc = ExponentialDecay(initial_learning_rate=0.00007, decay_steps=10000,
                                    decay_rate=0.98, staircase=False)
```

The D:G ratio is a **default argument of `fit`**:

```python
# ACGANCentralTrainingConfig.py:816
def fit(self, X_train=None, y_train=None, d_to_g_ratio=3):
```

and `TrainingClient.main` calls `client.fit()` with no arguments — so the effective ratio
is permanently 3, while `--AC_d_to_g_ratio` advertises a default of 1.

**Why it survived.** The plumbing cost. Threading one new value from argparse to the
trainer requires touching the parser, `hyperparameterLoading`'s 20-tuple, both
41-parameter dispatcher signatures, and the trainer's `__init__`. Nobody paid it 21
times. That is [F-04](#f-04) causing [F-01](#f-01) — fixing the plumbing is what makes
the hyperparameters cheap to wire.

**Plan → [R-14](../01_Refactoring_Strategy.md#r-14).** Variant A (defaults pinned to the
current literals) preserves behaviour exactly while making the flags live. Plus a
regression guard asserting every `--AC_*` flag is read somewhere, so this cannot recur.

---

### F-02 (P0) — Four advertised model types crash after minutes of work

```python
# modelCentralTrainingConfigLoad.py:70-77
elif model_type == "NIDS-IOT-Binary":             client = None
elif model_type == "NIDS-IOT-Multiclass":         client = None
elif model_type == "NIDS-IOT-Multiclass-Dynamic": client = None
# CANGAN: no branch at all — lines 158-166 are commented out
```

`client` is returned as `None` and `TrainingClient.main` calls `client.fit()` →
`AttributeError` — after `datasetLoadProcess` has already loaded and preprocessed
400 K rows and `modelCreateLoad` has built the Keras models.

The `CANGAN` value is offered on three flags (`--model_type`, `--dataset`,
`--dataset_processing`), each with a matching commented-out branch
(`datasetLoadProcess.py:77`, `:120`) — so the incompleteness is consistent, deliberate,
and undocumented.

**Plan → [R-15](../01_Refactoring_Strategy.md#r-15).** Validate
`(model_type, model_training, training_area)` against a factory registry at parse time.
[R-20](../01_Refactoring_Strategy.md#r-20) then makes `client = None` unrepresentable.

---

### F-03 (P1) — `--mode hermes` is a stub the tests certify as working

The client's hermes branch installs a training callback that raises
(`TrainingClient.py:192-197`); the host's constructs a cluster over a loopback link and
returns without serving (`HFLHost.py:155-189`). Both print success-shaped banners.

`tests/unit/test_mode_switch.py` asserts the **banner text**, which is why CI has never
noticed. The in-code comments say "awaiting Sprint 1.5 + Sprint 2 wiring"; both sprints
are recorded as closed in the README.

**Plan → [R-16](../01_Refactoring_Strategy.md#r-16)** (fail loudly now), then
[R-17](../01_Refactoring_Strategy.md#r-17) (make it real via `hermes_adapters/`).

---

### F-04 (P1) — Config-as-code, and the plumbing that enforces it {#f-04}

**Scale.** 47 classes / 45 files / 21,278 LOC in `Config/ModelTrainingConfig/`.

**Duplication census** (whole stack): `discriminator_loss` ×24, `generator_loss` ×17,
`evaluate_validation_disc` ×17, `evaluate_validation_NIDS` ×14, `setup_logger` ×12,
`probabilistic_fusion` ×10, `log_epoch_metrics` ×10, `gradient_penalty` ×7.

**Exact clone.** `CANGANCentralTrainingConfig.py` differs from
`ACGANCentralTrainingConfig.py` by four lines of `diff` output across 3,800 lines — a
class rename. It has no importer.

**Function sizes.** `ACGANCentralTrainingConfig.fit` 499 lines (complexity 34),
`.evaluate` 307, `.validation_disc` 133; `ACGANClientTrainingConfig.fit` 427;
`AC_DiscModelClientConfig.fit` 400; `ServerACDiscBothFitOnEndConfig.aggregate_fit` 311.

**Parameter plumbing.** `hyperparameterLoading` returns a 20-element positional tuple;
`modelCentralTrainingConfigLoad` and `modelFederatedTrainingConfigLoad` take 41
positional parameters each; `_run_fit_on_end_strategies` takes 42; 51 further functions
take ≥12. A single transposition anywhere in that chain is a silent wrong-value bug that
nothing can catch.

**The failure this already caused.** The September-2025 mode-collapse fixes (3:1 D:G
ratio, label-smoothing rebalance, generator-LR slowdown, discriminator health monitoring)
had to be hand-applied per copy. `ACGANCentralTrainingConfig.py` is the most-churned file
in the current tree (40 commits). Whether any copy was missed is currently
undeterminable, because **there are no tests**.

**Plan → [R-08](../01_Refactoring_Strategy.md#r-08) (characterization first),
[R-19](../01_Refactoring_Strategy.md#r-19) (typed configs),
[R-20](../01_Refactoring_Strategy.md#r-20) (registry factories),
[R-21](../01_Refactoring_Strategy.md#r-21) (one trainer family).**
Projected 21,278 → ~4,000 LOC.

---

### F-05 (P1) — Environment and deployment are not reproducible

| Artefact | Problem |
|---|---|
| `requirements_core.txt` | 273-package workstation `pip freeze`. `uuid==1.30` shadows the stdlib; `zmq==0.0.0` and `serial==0.0.97` are placeholder/wrong packages whose real counterparts are *also* pinned; `gps`, `distro-info`, `launchpadlib`, `wadllib`, `ssh-import-id` are apt packages. Differs from `requirements_edge.txt` by 45 of ~300 lines. |
| `docker-compose.yml` | Bind-mounts `C:/Users/kskos/PycharmProjects/FLVision/...` — one developer's machine, a *different project*. References `flwr-server` / `flwr-client` images no Dockerfile builds under those tags. |
| Missing root files | No `pyproject.toml`, `setup.py`, `pytest.ini`, `conftest.py`, `LICENSE`. README's layout block lists `pytest.ini`; README's footer links `LICENSE`. |
| `sys.path` hacks | 16 modules carry `sys.path.append(os.path.abspath('../../..'))`. |
| Dataset paths | `'../../../../datasets/CICIOT2023'` in six loaders; resolves only from `App/TrainingApp/Client/`, while the README documents running from the repository root and unzipping to `$HOME/datasets/`. |
| Server addresses | `192.168.129.{3,6,7,8}:8080` hardcoded in two places (`TrainingClient.py:124-133`, `ArgumentConfigLoad.py:211-220`). |

`hermes/` actually needs numpy. The published "lightweight core" pulls TensorFlow,
PyTorch, PyQt5, and Ansible.

**Plan → [R-04](../01_Refactoring_Strategy.md#r-04),
[R-06](../01_Refactoring_Strategy.md#r-06), [R-07](../01_Refactoring_Strategy.md#r-07).**

---

### F-06 (P2) — Dead code and vendored trees

| Path | LOC | Evidence it is dead |
|---|---|---|
| `FlightFramework/` | 5,853 | zero `import flight` outside itself, despite four documentation claims of reuse |
| `CANGANCentralTrainingConfig.py` | 1,900 | only reference is a commented-out import |
| `misc/Copy of ACGANCentralTrainingConfig.py` | ~600 | filename; also has a space in it |
| `ciciot2023DatasetLoad.py` | 271 | superseded by `…LoadV2`; hardcodes its own arguments |
| `iotbotnet2020DatasetLoad.py` | 315 | superseded by `…LoadV2` |
| **Total** | **~8,900** | |

`FlightFramework/quickstart/iotbotnetDatasetLoad.py` is additionally a verbatim copy of
the project's own 267-line `loadIOTBOTNET` with `/root/datasets/...` baked in — the same
function existing in three places.

Beyond dead weight, this is a licence-provenance question: a copied MIT third-party
project sits inside a repository with no `LICENSE` of its own.

**Plan → [R-05](../01_Refactoring_Strategy.md#r-05),
[R-02](../01_Refactoring_Strategy.md#r-02).**

---

### F-07 (P2) — Model-builder version sprawl

`discriminatorStruct.py` contains eight AC-GAN discriminator builders:
`build_AC_discriminator_V0`, `_ver_2`, `build_AC_discriminator`,
`build_CAN_AC_discriminator`, `_ver_last`, `_ver_3b`, `_ver_4`, `_v5`. `NIDS_Struct.py`
contains 13 NIDS builders. None carries a deprecation marker or a docstring naming the
current one; identifying the live builder requires tracing `modelCreateLoad`'s 375-line,
complexity-89 branch tree.

`_ver_last` followed by `_ver_3b`, `_ver_4` and `_v5` is a good summary of the problem.

**Plan → [R-22](../01_Refactoring_Strategy.md#r-22).**

---

### F-08 (P2) — Analysis monolith

`experiments/analysis/exp3.py:342` — `write_figures`, **863 lines, cyclomatic complexity
170**. The largest function in the repository by roughly 2×. `exp1.py`'s namesake is 215
lines / complexity 76.

Consequences: regenerating one figure re-runs all six; no branch is independently
testable (`test_write_figures_smoke` can only assert files appear); a change to the
β-sweep panel risks the ρ_contact bar chart.

Note the contrast with `experiments/runner/` in the same package — 567 LOC, small pure
functions, correct paired-seed derivation, honest resume semantics, well tested. The
harness is exemplary; only the figure layer is not.

**Plan → [R-26](../01_Refactoring_Strategy.md#r-26).**

---

### F-09 (P2) — Test posture

| Area | LOC | Tests |
|---|---|---|
| `Config/ModelTrainingConfig/` | 21,278 | **0** |
| `Config/SessionConfig/` | 1,659 | 0 (beyond a subprocess banner smoke) |
| `App/` | 1,938 | 0 |
| `Analysis/` | 2,185 | 0 |
| `experiments/` | 12,053 | 10 unit modules — good |
| `hermes/` | 14,147 | 72 modules — strong |

Local run: 512 passed, 8 failed, 1 skipped, against a README claim of "410 passed". Seven
failures are environmental with no skip guard; one is a genuine stale assertion
(`test_experiments_calibration.py:36` still asserts `status == "placeholder"` after the
shipped TOML was promoted to `"verified"`).

**Plan → [R-08](../01_Refactoring_Strategy.md#r-08),
[R-09](../01_Refactoring_Strategy.md#r-09).**

---

## 2. Ordered work plan for this stack

| # | Item | Fixes | Effort | Risk | Gate |
|---|---|---|---|---|---|
| 1 | Delete ~8.9 K LOC of dead trees | F-06 | 0.5 d | Very low | tag before deletion |
| 2 | Repair + split requirements; add `LICENSE` | F-05 | 1 d | Very low | clean-venv install test |
| 3 | `pyproject.toml` + `pytest.ini`; drop 16 `sys.path` hacks | F-05 | 2 d | Low | `pip install -e .` |
| 4 | `hifins/paths.py` repo-anchored dataset resolution | F-05 | 1 d | Low | runs identically from both CWDs |
| 5 | **Characterization harness for every live trainer** | F-04, F-09 | 4 d | Low | **gates items 7–9** |
| 6 | Fail fast on unimplemented model types | F-02 | 0.5 d | ⚠ behaviour | approval |
| 7 | Typed config dataclasses replace the 20-tuple and 41-arg calls | F-04 | 5 d | Medium | goldens green |
| 8 | Registry factories replace the branch trees | F-04, F-02 | 3 d | Medium | goldens green |
| 9 | One parameterised trainer family | F-04 | 15 d | **High** | goldens green, per-trainer |
| 10 | Wire the AC-GAN hyperparameters (Variant A) | F-01 | 1 d | ⚠ behaviour | approval + goldens |
| 11 | `hermes_adapters/` — real `GeneratorHost` + `LocalTrainFn` | F-03 | 8 d | Medium | e2e real-model run |
| 12 | Consolidate model builders; archive superseded versions | F-07 | 2 d | Low | goldens green |
| 13 | Split `write_figures` | F-08 | 3 d | Low | figures byte-identical |

Item 5 is the gate. Items 1–4 are pure cleanup and can ship immediately.

---

## 3. What not to change

| Keep | Why |
|---|---|
| `experiments/runner/` | Correct paired-seed derivation, honest resume, small pure functions. Reference-quality. |
| `Config/modelStructures/` builder signatures | Pure Keras factories with no side effects. Prune versions ([R-22](../01_Refactoring_Strategy.md#r-22)); don't restructure. |
| The `preprocess_*` split by dataset-processing mode | The dispatch is verbose but the boundary is right. |
| v2 dataset loaders' argument handling | They genuinely honour their parameters. Only the v1 copies are broken. |
| Published figure *output* | Validated by published results. Decompose the code ([R-26](../01_Refactoring_Strategy.md#r-26)); never re-derive the numbers. |
| The AC-GAN training *algorithm* | Mode-collapse tuning is a research result. Consolidation must preserve it byte-for-byte, which is what the goldens are for. |

---

## 4. Migration pattern for the trainer consolidation

Item 9 is the risky one. Run it strictly per-trainer, never in bulk:

```
for each live trainer class T:
  1. Golden-snapshot T on the synthetic fixture (weights hash, loss trace, log lines).
  2. Express T as a config instance of the new parameterised family.
  3. Run both. Assert byte-equality of the snapshot.
  4. Point T's factory entry at the new path. Suite green.
  5. Delete T's file. Suite green.
  6. Commit. One trainer per commit, one trainer per PR.
```

If step 3 fails, the difference is either a real behavioural divergence (fix the new
path) or an intentional difference that was never documented (document it, then decide).
Either way the answer is knowable before anything is deleted — which is the entire point
of doing item 5 first.

**Reference implementations:** [`HIFINS_Production_Code.md`](HIFINS_Production_Code.md).

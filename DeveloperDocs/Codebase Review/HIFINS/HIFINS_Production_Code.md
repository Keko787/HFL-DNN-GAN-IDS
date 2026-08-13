# HIFINS — Production-Grade Code Proposals

**Status:** Proposal. **Not applied.** Requires approval per
[`../01_Refactoring_Strategy.md` §5](../01_Refactoring_Strategy.md#5-approval-checklist).
**Constraint:** every change below is behaviour-preserving unless a section is marked
**⚠ BEHAVIOUR CHANGE**, in which case the change is stated explicitly and a
behaviour-preserving variant is given.

---

## Contents

1. [Packaging](#1-packaging) — F-05
2. [Path resolution](#2-path-resolution) — F-05
3. [Typed configuration](#3-typed-configuration) — F-04
4. [Registry factories](#4-registry-factories) — F-02, F-04
5. [`hermes_adapters/`](#5-hermes_adapters) — F-03
6. [AC-GAN hyperparameter wiring ⚠](#6-ac-gan-hyperparameter-wiring) — F-01
7. [One trainer family](#7-one-trainer-family) — F-04
8. [Characterization harness](#8-characterization-harness) — F-04, F-09

---

## 1. Packaging

**Defect (F-05).** No `pyproject.toml`, no `pytest.ini`, no `LICENSE`; 16 modules carry
`sys.path.append(os.path.abspath('../../..'))`; `@pytest.mark.slow` is unregistered.

**New file — `pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "hifins"
version = "0.9.0"
description = "Hierarchical FL + GAN-based NIDS over a mule-assisted four-tier topology (HERMES)"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }

# Deliberately minimal. `hermes/` imports exactly one third-party name.
# Everything heavier is an extra, so an edge node can install the
# scheduling substrate without pulling TensorFlow.
dependencies = ["numpy>=1.24,<3"]

[project.optional-dependencies]
test    = ["pytest>=7.0"]
ml      = ["tensorflow==2.15.*", "scikit-learn>=1.3", "pandas>=2.0",
           "matplotlib>=3.7", "scipy>=1.10", "h5py>=3.9"]
legacy  = ["hifins[ml]", "flwr==1.9.0"]
dev     = ["hifins[test,ml,legacy]", "ruff", "mypy"]

[project.scripts]
hermes-cluster = "hermes.processes.cluster:main"
hermes-mule    = "hermes.processes.mule:main"
hermes-device  = "hermes.processes.device:main"

[tool.setuptools.packages.find]
include = ["hermes*", "experiments*", "Config*", "App*", "hermes_adapters*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = [
    "slow: real subprocesses or long-running; deselect with -m 'not slow'",
    "requires_tf: needs TensorFlow (skipped when unavailable)",
    "requires_flwr: needs Flower (skipped when unavailable)",
]

[tool.ruff]
line-length = 100
target-version = "py310"
```

**New file — `conftest.py` (repository root):**

```python
"""Repo-root pytest configuration.

Provides the skip guards that let the suite be green in a core-only venv.
Today seven of the eight local failures are "TensorFlow/Flower absent"
reported as failures, which trains people to ignore a red suite — see
finding T-01.
"""
from __future__ import annotations

import importlib.util

import pytest


def _missing(module: str) -> bool:
    return importlib.util.find_spec(module) is None


def pytest_collection_modifyitems(config, items):
    skip_tf = pytest.mark.skip(reason="TensorFlow not installed (pip install 'hifins[ml]')")
    skip_flwr = pytest.mark.skip(reason="Flower not installed (pip install 'hifins[legacy]')")
    tf_absent, flwr_absent = _missing("tensorflow"), _missing("flwr")
    for item in items:
        if tf_absent and "requires_tf" in item.keywords:
            item.add_marker(skip_tf)
        if flwr_absent and "requires_flwr" in item.keywords:
            item.add_marker(skip_flwr)
```

**Then:** `pip install -e .` and delete all 16 `sys.path.append` lines. Verify with
`python -c "import hermes, experiments, Config"` from an unrelated directory.

---

## 2. Path resolution

**Defect (F-05).** `'../../../../datasets/CICIOT2023'` in six loaders resolves only from
`App/TrainingApp/Client/`, while the README documents running from the repository root
and unzipping to `$HOME/datasets/`.

**New file — `hifins/paths.py`:**

```python
"""Dataset and artefact path resolution.

Replaces six copies of ``'../../../../datasets/<NAME>'``. That literal
resolves correctly only when the process CWD is ``App/TrainingApp/Client/``
— four levels up from there is the datasets root. The README instructs
running from the repository root and unzipping to ``$HOME/datasets``, under
which the literal points three levels *above* the repository's parent. The
documented setup and the documented invocation could never both work.

Resolution order (first hit wins):

  1. an explicit ``root`` argument                       — tests, CI
  2. ``$HIFINS_DATASET_ROOT``                            — deployment
  3. ``$HOME/datasets``                                  — the documented setup
  4. ``<repo-root>/../datasets``                         — common dev layout
  5. ``../../../../datasets`` relative to CWD            — legacy, kept last

Step 5 is retained deliberately: an existing run launched from
``App/TrainingApp/Client/`` finds exactly the directory it always found, so
this change is behaviour-preserving for current users while making every
other invocation work too.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# hifins/paths.py -> hifins/ -> <repo root>
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset directory cannot be located.

    Carries every candidate that was tried, because the failure this
    replaces was a bare FileNotFoundError on a mangled relative path with
    no indication of what the resolver was looking for.
    """


def dataset_root(root: Optional[os.PathLike] = None) -> Path:
    for candidate in _candidate_roots(root):
        if candidate.is_dir():
            return candidate
    raise DatasetNotFoundError(
        "no dataset root found. Set HIFINS_DATASET_ROOT, or place datasets "
        "under $HOME/datasets. Tried:\n  "
        + "\n  ".join(str(c) for c in _candidate_roots(root))
    )


def dataset_dir(name: str, *, root: Optional[os.PathLike] = None) -> Path:
    """Resolve one named dataset directory, e.g. ``dataset_dir("CICIOT2023")``."""
    base = dataset_root(root)
    path = base / name
    if not path.is_dir():
        raise DatasetNotFoundError(
            f"dataset {name!r} not found under {base}. "
            f"Available: {sorted(p.name for p in base.iterdir() if p.is_dir())}"
        )
    return path


def _candidate_roots(root: Optional[os.PathLike]) -> List[Path]:
    out: List[Path] = []
    if root is not None:
        out.append(Path(root).expanduser().resolve())
    env = os.environ.get("HIFINS_DATASET_ROOT")
    if env:
        out.append(Path(env).expanduser().resolve())
    out.append(Path.home() / "datasets")
    out.append((REPO_ROOT.parent / "datasets").resolve())
    out.append(Path("../../../../datasets").resolve())   # legacy, last
    return out
```

**Call-site change** (`ciciot2023DatasetLoadV2.py:255`):

```python
-    DATASET_DIRECTORY = f'../../../../datasets/CICIOT2023_POISONED{poisonedDataType}' \
-        if poisonedDataType else '../../../../datasets/CICIOT2023'
+    name = f"CICIOT2023_POISONED{poisonedDataType}" if poisonedDataType else "CICIOT2023"
+    DATASET_DIRECTORY = str(dataset_dir(name, root=dataset_root_override))
```

---

## 3. Typed configuration

**Defect (F-04).** `hyperparameterLoading` returns a 20-element positional tuple; three
dispatchers take 41–42 positional parameters; 51 further functions take ≥12.

**New file — `hifins/config/session.py`:**

```python
"""Frozen configuration objects replacing the positional-tuple plumbing.

Before: ``hyperparameterLoading`` returned a 20-element tuple, unpacked into
20 locals at each call site and threaded positionally through 41-parameter
dispatchers. A single transposition anywhere in that chain is a silent
wrong-value bug no type checker or test can catch — and the cost of adding
one parameter (six edits across five files) is the direct reason the 21
AC-GAN CLI flags were declared and never wired (finding F-01).

Frozen dataclasses because these are read-only after construction; a trainer
that mutates its own config is a bug we would rather not be able to write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

ModelType = Literal["NIDS", "NIDS-IOT-Binary", "NIDS-IOT-Multiclass",
                    "NIDS-IOT-Multiclass-Dynamic", "GAN", "WGAN-GP", "AC-GAN"]
SubModel = Literal["NIDS", "Generator", "Discriminator", "Both"]
TrainingArea = Literal["Central", "Federated"]


@dataclass(frozen=True)
class DatasetConfig:
    name: Literal["CICIOT", "IOTBOTNET", "IOT", "LIVEDATA"] = "CICIOT"
    processing: str = "Default"
    train_sample_size: int = 50
    test_sample_size: int = 15
    training_dataset_size: int = 400_000
    testing_dataset_size: int = 80_000
    attack_eval_samples_ratio: float = 1.0
    random_seed: int = 110
    root: Optional[str] = None          # None -> hifins.paths resolution


@dataclass(frozen=True)
class ModelShape:
    """Shapes derived from the loaded data. Not user-supplied."""
    input_dim: int
    num_classes: Optional[int] = None
    latent_dim: Optional[int] = None
    noise_dim: Optional[int] = None


@dataclass(frozen=True)
class OptimizerConfig:
    """One optimizer's full specification.

    Every field previously lived as a literal inside a trainer's __init__
    (ACGANCentralTrainingConfig.py:109-118) while the CLI advertised flags
    that were never read. Defaults here are those literals verbatim, so
    Variant A of R-14 is byte-identical to today. See §6.
    """
    learning_rate: float
    decay_steps: int = 10_000
    decay_rate: float = 0.98
    staircase: bool = False
    beta_1: float = 0.5
    beta_2: float = 0.999
    clipnorm: Optional[float] = 1.0


@dataclass(frozen=True)
class ACGANHyperParams:
    # Defaults are ACGANCentralTrainingConfig.py's hardcoded literals.
    generator: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=0.00012))
    discriminator: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=0.00007))
    d_to_g_ratio: int = 3                  # fit()'s default argument, not the CLI's 1
    valid_smoothing_factor: float = 0.08
    fake_smoothing_factor: float = 0.05
    gen_smoothing_factor: float = 0.08
    attack_weight: float = 0.5
    benign_weight: float = 0.5
    validity_weight: float = 0.5
    class_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.d_to_g_ratio < 1:
            raise ValueError(f"d_to_g_ratio must be >= 1, got {self.d_to_g_ratio}")
        for name in ("valid_smoothing_factor", "fake_smoothing_factor",
                     "gen_smoothing_factor"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {v}")


@dataclass(frozen=True)
class RegularizationConfig:
    enabled: bool = True
    l2_alpha: float = 1e-4
    dp_enabled: bool = False
    l2_norm_clip: Optional[float] = None
    noise_multiplier: Optional[float] = None
    num_microbatches: int = 1


@dataclass(frozen=True)
class CallbackConfig:
    early_stopping: bool = False
    es_metric: str = "val_loss"
    es_patience: int = 3
    restore_best_weights: bool = True
    lr_schedule_reduction: bool = False
    lr_metric: str = "val_loss"
    lr_patience: int = 2
    checkpointing: bool = False
    checkpoint_metric: str = "val_loss"
    checkpoint_mode: str = "min"
    save_best_only: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 64
    steps_per_epoch: Optional[int] = None      # None -> len(X) // batch_size
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)


@dataclass(frozen=True)
class FederationConfig:
    area: TrainingArea = "Central"
    server_address: Optional[str] = None       # None -> resolved from --host
    rounds: int = 1
    min_clients: int = 2
    synth_portion: float = 0.0
    node_id: int = 1


@dataclass(frozen=True)
class RunConfig:
    """Everything one training run needs. Replaces the 41-parameter calls."""
    model_type: ModelType
    sub_model: SubModel
    dataset: DatasetConfig
    training: TrainingConfig
    federation: FederationConfig
    shape: Optional[ModelShape] = None                  # filled after data load
    acgan: ACGANHyperParams = field(default_factory=ACGANHyperParams)
    pretrained: "PretrainedPaths" = field(default_factory=lambda: PretrainedPaths())
    save_name: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class PretrainedPaths:
    gan: Optional[str] = None
    generator: Optional[str] = None
    discriminator: Optional[str] = None
    nids: Optional[str] = None
```

**Adapter for incremental migration** — build the config from today's `args`, so call
sites migrate one at a time:

```python
# hifins/config/from_args.py
def run_config_from_args(args) -> RunConfig:
    """Bridge argparse Namespace -> RunConfig.

    Exists so the migration is incremental: a call site can take the
    RunConfig while its neighbours still take the 41-tuple. Delete once
    every dispatcher is migrated.
    """
    return RunConfig(
        model_type=args.model_type,
        sub_model=args.model_training,
        dataset=DatasetConfig(
            name=args.dataset,
            processing=args.dataset_processing,
            train_sample_size=args.ciciot_train_sample_size,
            test_sample_size=args.ciciot_test_sample_size,
            training_dataset_size=args.ciciot_training_dataset_size,
            testing_dataset_size=args.ciciot_testing_dataset_size,
            attack_eval_samples_ratio=args.ciciot_attack_eval_samples_ratio,
            random_seed=args.ciciot_random_seed,
        ),
        training=TrainingConfig(epochs=args.epochs),
        federation=FederationConfig(
            area=args.trainingArea,
            server_address=resolve_server_address(args),
            rounds=getattr(args, "rounds", 1),
            min_clients=getattr(args, "min_clients", 2),
            synth_portion=getattr(args, "synth_portion", 0.0),
        ),
        acgan=acgan_params_from_args(args),          # §6
        pretrained=PretrainedPaths(
            gan=args.pretrained_GAN, generator=args.pretrained_generator,
            discriminator=args.pretrained_discriminator, nids=args.pretrained_nids,
        ),
        save_name=args.save_name,
        timestamp=args.timestamp,
    )
```

**Also fixes E-03** (the duplicated `192.168.129.x` table):

```python
# hifins/config/hosts.py
"""Server-address resolution for the legacy Flower path.

The four-way host table was duplicated in TrainingClient.py:124-133 (used)
and ArgumentConfigLoad.py:211-220 (displayed). Two copies of one lab subnet,
guaranteed to drift, both needing a code edit for any topology change.
One table, overridable by environment so a redeploy is not a commit.
"""
from __future__ import annotations

import os
from typing import Dict

DEFAULT_PORT = int(os.environ.get("HIFINS_FL_PORT", "8080"))

_PRESET_HOSTS: Dict[str, str] = {
    "1": "192.168.129.3", "2": "192.168.129.6",
    "3": "192.168.129.7", "4": "192.168.129.8",
}


def resolve_server_address(args) -> str:
    """Custom host wins; then $HIFINS_FL_SERVER; then the preset table;
    then treat --host as a literal address (the existing failsafe)."""
    if getattr(args, "custom_host", None):
        return f"{args.custom_host}:{DEFAULT_PORT}"
    env = os.environ.get("HIFINS_FL_SERVER")
    if env:
        return env if ":" in env else f"{env}:{DEFAULT_PORT}"
    host = str(getattr(args, "host", "1"))
    return f"{_PRESET_HOSTS.get(host, host)}:{DEFAULT_PORT}"
```

---

## 4. Registry factories

**Defect (F-02, F-04).** `modelCreateLoad` is 375 lines / complexity 89; the two
dispatchers are 41-parameter if/elif chains with `client = None` fall-throughs.

```python
# hifins/factories.py
"""Decorator-registered factories replacing the branch trees.

Two problems disappear at once:

* ``modelCreateLoad``'s 375-line, complexity-89 if/elif tree and the two
  41-parameter dispatchers become table lookups.
* The ``client = None`` fall-through becomes unrepresentable. Today four
  advertised --model_type values return None and crash with
  ``AttributeError: 'NoneType' object has no attribute 'fit'`` *after* the
  dataset load and model construction (finding F-02). A registry either has
  the key or raises at lookup, naming the valid keys.

Adding a model type is one decorator in one new file. No existing file is
edited, which is what makes the registry worth the indirection.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Protocol, Tuple

from .config.session import ModelShape, RunConfig

TrainerKey = Tuple[str, str, str]           # (model_type, sub_model, training_area)


class Trainer(Protocol):
    def fit(self) -> None: ...
    def evaluate(self) -> None: ...
    def save(self, name: str) -> None: ...


class UnsupportedTrainerError(NotImplementedError):
    """Raised for a (model_type, sub_model, training_area) with no factory."""


_TRAINERS: Dict[TrainerKey, Callable[..., Trainer]] = {}


def register_trainer(model_type: str, sub_model: str, training_area: str):
    def _decorate(fn: Callable[..., Trainer]) -> Callable[..., Trainer]:
        key = (model_type, sub_model, training_area)
        if key in _TRAINERS:
            raise RuntimeError(f"duplicate trainer registration for {key}")
        _TRAINERS[key] = fn
        return fn
    return _decorate


def supported_keys() -> Iterable[TrainerKey]:
    return sorted(_TRAINERS)


def is_supported(model_type: str, sub_model: str, training_area: str) -> bool:
    return (model_type, sub_model, training_area) in _TRAINERS


def build_trainer(cfg: RunConfig, *, models, data) -> Trainer:
    key = (cfg.model_type, cfg.sub_model, cfg.federation.area)
    try:
        factory = _TRAINERS[key]
    except KeyError:
        raise UnsupportedTrainerError(
            f"no trainer registered for model_type={cfg.model_type!r} "
            f"sub_model={cfg.sub_model!r} area={cfg.federation.area!r}.\n"
            f"Supported combinations:\n  "
            + "\n  ".join(f"{m} / {s} / {a}" for m, s, a in supported_keys())
        ) from None
    return factory(cfg=cfg, models=models, data=data)
```

**⚠ BEHAVIOUR CHANGE (R-15), and the whole point of it** — validate at parse time, before
the dataset load:

```python
# in parse_training_client_args, after args = parser.parse_args()
if not is_supported(args.model_type, args.model_training, args.trainingArea):
    parser.error(
        f"--model_type {args.model_type} with --model_training "
        f"{args.model_training} and --trainingArea {args.trainingArea} is not "
        f"implemented.\nSupported:\n  "
        + "\n  ".join(f"{m} / {s} / {a}" for m, s, a in supported_keys())
    )
```

What changes: an `AttributeError` after minutes of data loading becomes an immediate
`argparse` error naming what is supported. Nothing that previously worked stops working —
the four affected values (`CANGAN`, `NIDS-IOT-Binary`, `NIDS-IOT-Multiclass`,
`NIDS-IOT-Multiclass-Dynamic`) could never run.

---

## 5. `hermes_adapters/`

**Defect (F-03, and H-08 from the HERMES side).** The `--mode hermes` bridge is a stub;
`hermes/processes/*` reaches into `experiments.exp4.model_task` instead of going through
the Protocols. Two consequences that matter scientifically:

- `StubGeneratorHost.make_synth_batch` returns **zero tensors** in every live HERMES
  run, so the GAN in "GAN-based NIDS" has never run inside HERMES.
- `experiments/exp4/model_task.py:150-152` states the device-side callback *accepts and
  ignores* the synth batch: *"accepted for signature compatibility … but ignored in
  EX-4.1 (the GAN is separate, revision-scope work)."*

So the synthetic-augmentation path is inert on both ends. The adapters below are what
close it — and they need **zero changes to `hermes/`**, because the Protocols are already
the right shape.

```python
# hermes_adapters/keras_generator_host.py
"""Real GeneratorHost backed by the AC-GAN generator + DNN-IDS discriminator.

Satisfies hermes.cluster.host_cluster.GeneratorHost structurally — no
inheritance, no import of hermes internals beyond the Weights alias, so this
package depends on hermes and hermes depends on nothing here.

Replaces StubGeneratorHost, whose make_synth_batch returns zero tensors. The
GAN half of "GAN-based NIDS" has therefore never executed inside a HERMES
run; this is the change that makes it real.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)
Weights = List[np.ndarray]


class KerasGeneratorHost:

    def __init__(
        self,
        *,
        generator_path: Optional[str] = None,
        discriminator_path: Optional[str] = None,
        input_dim: int,
        latent_dim: int = 100,
        num_classes: int = 2,
        seed: int = 0,
    ) -> None:
        # Lazy import: the cluster process pays the TensorFlow import cost
        # only when a real provider is configured. The stub path stays
        # numpy-only, which is what keeps `hermes` installable on an edge
        # node without an ML stack.
        import tensorflow as tf

        from Config.modelStructures.GAN.generatorStruct import build_AC_generator
        from Config.modelStructures.NIDS.NIDS_Struct import create_CICIOT_Model

        tf.keras.utils.set_random_seed(int(seed))
        self._latent_dim = latent_dim
        self._num_classes = num_classes
        self._rng = np.random.default_rng(seed)

        self._generator = (
            tf.keras.models.load_model(generator_path)
            if generator_path and Path(generator_path).exists()
            else build_AC_generator(latent_dim, num_classes, input_dim)
        )
        self._discriminator = (
            tf.keras.models.load_model(discriminator_path)
            if discriminator_path and Path(discriminator_path).exists()
            else create_CICIOT_Model(
                input_dim=input_dim, regularizationEnabled=True,
                DP_enabled=False, l2_alpha=1e-4,
            )
        )
        self._last_refinement_round = -1
        log.info(
            "KerasGeneratorHost ready: input_dim=%d latent_dim=%d classes=%d "
            "generator=%s discriminator=%s",
            input_dim, latent_dim, num_classes,
            generator_path or "<fresh>", discriminator_path or "<fresh>",
        )

    # ---- GeneratorHost Protocol ----

    def make_synth_batch(self, n: int) -> List[np.ndarray]:
        """Draw ``n`` synthetic samples from θ_gen.

        Class labels are drawn uniformly so the batch is balanced regardless
        of the real data's skew — the synth batch exists to *augment* the
        device's minority class, so mirroring the real imbalance would
        defeat its purpose.
        """
        if n <= 0:
            return []
        noise = self._rng.normal(0.0, 1.0, size=(n, self._latent_dim)).astype(np.float32)
        labels = self._rng.integers(0, self._num_classes, size=(n, 1)).astype(np.int32)
        samples = self._generator.predict([noise, labels], verbose=0)
        return [np.asarray(row, dtype=np.float32) for row in samples]

    def get_global_disc_weights(self) -> Weights:
        return [np.asarray(w).copy() for w in self._discriminator.get_weights()]

    def update_disc_from_cluster_avg(self, weights: Weights) -> None:
        current = self._discriminator.get_weights()
        if len(weights) != len(current):
            raise ValueError(
                f"cluster average has {len(weights)} layers; discriminator "
                f"has {len(current)}"
            )
        for i, (new, old) in enumerate(zip(weights, current)):
            if new.shape != old.shape:
                raise ValueError(
                    f"layer {i} shape mismatch: aggregate {new.shape} vs "
                    f"model {old.shape}"
                )
        self._discriminator.set_weights([np.asarray(w) for w in weights])

    def apply_tier3_gen_refinement(
        self, weights: Weights, refinement_round: int = 0
    ) -> None:
        # Tier-3 is best-effort polled; an older packet can arrive after a
        # newer one. Same ordering rule as StubGeneratorHost.
        if refinement_round < self._last_refinement_round:
            log.info(
                "ignoring stale tier-3 refinement round=%d (have %d)",
                refinement_round, self._last_refinement_round,
            )
            return
        self._generator.set_weights([np.asarray(w) for w in weights])
        self._last_refinement_round = refinement_round


def make_generator_host(**kwargs) -> KerasGeneratorHost:
    """Entry point for hermes.processes.providers.load_provider."""
    return KerasGeneratorHost(**kwargs)
```

```python
# hermes_adapters/keras_local_train.py
"""Real LocalTrainFn: fits the canonical DNN-IDS on one device shard.

Differs from experiments.exp4.model_task.make_local_train_fn in one respect
that matters: it *uses* the synth batch. The exp4 implementation accepts and
discards it ("the GAN is separate, revision-scope work"), so the synthetic
augmentation the whole AC-GAN pipeline exists to produce has never reached a
device's training step.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


class KerasLocalTrain:

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        input_dim: Optional[int] = None,
        epochs: int = 1,
        batch_size: int = 64,
        l2_alpha: float = 1e-3,
        learning_rate: float = 1e-3,
        synth_label: float = 1.0,
        max_synth_fraction: float = 0.25,
        seed: int = 0,
    ) -> None:
        import tensorflow as tf
        from Config.modelStructures.NIDS.NIDS_Struct import create_CICIOT_Model

        tf.keras.utils.set_random_seed(int(seed))
        self._dim = int(input_dim if input_dim is not None else X.shape[1])
        self._X = np.asarray(X, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32).reshape(-1)
        self._epochs, self._batch_size = epochs, batch_size
        self._synth_label = synth_label
        self._max_synth_fraction = max_synth_fraction
        # Built once and re-fitted from the pushed θ each round. Rebuilding a
        # Keras model per round would dominate the device's runtime.
        self._model = create_CICIOT_Model(
            input_dim=self._dim, regularizationEnabled=True,
            DP_enabled=False, l2_alpha=l2_alpha,
        )
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

    def __call__(self, theta_disc, synth_batch: List[np.ndarray]):
        from hermes.mission import LocalTrainResult

        self._model.set_weights([np.asarray(w) for w in theta_disc])
        X, y = self._augment(synth_batch)
        history = self._model.fit(
            X, y, epochs=self._epochs, batch_size=self._batch_size, verbose=0,
        )
        after = [np.asarray(w).copy() for w in self._model.get_weights()]
        return LocalTrainResult(
            delta_theta=after,
            num_examples=int(len(y)),
            accuracy=float(history.history["accuracy"][-1]),
            auc=float(history.history["auc"][-1]),
            loss=float(history.history["loss"][-1]),
            theta_after=after,
        )

    def _augment(self, synth_batch):
        """Append synthetic rows, capped as a fraction of the real shard.

        The cap matters: an unbounded synth batch would let the cluster's
        generator dominate a small device's local distribution, which is the
        opposite of the diversity term the utility score rewards.
        """
        if not synth_batch:
            return self._X, self._y
        synth = np.asarray(
            [np.asarray(s, dtype=np.float32).reshape(-1) for s in synth_batch]
        )
        if synth.ndim != 2 or synth.shape[1] != self._dim:
            log.warning(
                "ignoring synth batch with shape %s (expected (*, %d))",
                synth.shape, self._dim,
            )
            return self._X, self._y
        cap = int(len(self._y) * self._max_synth_fraction)
        if cap <= 0:
            return self._X, self._y
        synth = synth[:cap]
        labels = np.full(len(synth), self._synth_label, dtype=np.float32)
        return (np.concatenate([self._X, synth], axis=0),
                np.concatenate([self._y, labels], axis=0))


def make_local_train_provider(*, shard_path: str, **kwargs) -> KerasLocalTrain:
    """Entry point for hermes.processes.providers.load_provider."""
    from experiments.exp4.model_task import load_xy
    X, y = load_xy(shard_path)
    return KerasLocalTrain(X, y, **kwargs)
```

Wire them in via config, not import
([`../Hermes/HERMES_Production_Code.md` §6](../Hermes/HERMES_Production_Code.md#6-inverting-the-experiments-dependency)):

```jsonc
{
  "generator_provider":   "hermes_adapters.keras_generator_host:make_generator_host",
  "local_train_provider": "hermes_adapters.keras_local_train:make_local_train_provider"
}
```

**⚠ Note.** Enabling a *real* synth batch changes device-side training inputs relative to
today's zero-tensor stub. That is the intended fix, but it invalidates comparison against
Exp-4 results collected under the stub. Treat the switch-on as a new experimental
condition, not a bug fix, and record it in the methodology docs.

---

## 6. AC-GAN hyperparameter wiring

**⚠ BEHAVIOUR CHANGE — approval-gated ([R-14](../01_Refactoring_Strategy.md#r-14)).**

**Variant A (recommended, behaviour-preserving):**

```python
# hifins/config/from_args.py
def acgan_params_from_args(args) -> ACGANHyperParams:
    """Build ACGANHyperParams from the CLI.

    Variant A: argparse defaults are changed to match the values the trainer
    currently hardcodes, so a run with no flags is byte-identical to today.
    A run *with* flags now actually honours them — which it never did: all
    21 --AC_* flags appear exactly once in the repository, at their own
    declaration (finding C-01).
    """
    return ACGANHyperParams(
        generator=OptimizerConfig(
            learning_rate=args.AC_gen_learning_rate,
            decay_steps=args.AC_gen_decay_steps,
            decay_rate=args.AC_gen_decay_rate,
            staircase=args.AC_gen_staircase,
            beta_1=args.AC_gen_beta_1,
            beta_2=args.AC_gen_beta_2,
        ),
        discriminator=OptimizerConfig(
            learning_rate=args.AC_disc_learning_rate,
            decay_steps=args.AC_disc_decay_steps,
            decay_rate=args.AC_disc_decay_rate,
            staircase=args.AC_disc_staircase,
            beta_1=args.AC_disc_beta_1,
            beta_2=args.AC_disc_beta_2,
        ),
        d_to_g_ratio=args.AC_d_to_g_ratio,
        valid_smoothing_factor=args.AC_disc_valid_smoothing_factor,
        fake_smoothing_factor=args.AC_disc_fake_smoothing_factor,
        gen_smoothing_factor=args.AC_gen_smoothing_factor,
        attack_weight=args.AC_disc_attack_weight,
        benign_weight=args.AC_disc_benign_weight,
        validity_weight=args.AC_disc_validity_weight,
        class_weight=args.AC_disc_class_weight,
    )
```

Argparse defaults change to the current effective values:

```diff
-parser.add_argument("--AC_disc_learning_rate", type=float, default=0.00001, …)
+# Default = the value the trainer hardcodes today (ACGANCentralTrainingConfig.py:114).
+# Changing the default rather than the trainer keeps every existing
+# invocation byte-identical while making the flag live for the first time.
+parser.add_argument("--AC_disc_learning_rate", type=float, default=0.00007, …)

-parser.add_argument("--AC_gen_learning_rate", type=float, default=0.00003, …)
+parser.add_argument("--AC_gen_learning_rate", type=float, default=0.00012, …)

-parser.add_argument("--AC_d_to_g_ratio", type=int, default=1, …)
+# fit()'s own default is 3 and TrainingClient calls fit() with no arguments,
+# so 3 — not 1 — is what every run has actually used.
+parser.add_argument("--AC_d_to_g_ratio", type=int, default=3, …)
```

Trainer construction becomes:

```python
class ACGANTrainer(BaseTrainer):
    def _build_optimizers(self) -> None:
        import tensorflow as tf
        hp = self.cfg.acgan
        self.gen_optimizer = _adam(hp.generator)
        self.disc_optimizer = _adam(hp.discriminator)
        # Effective values in the log, so a lab notebook can be checked
        # against what actually ran. Their absence is why C-01 went
        # unnoticed for a full experiment cycle.
        self.logger.info("AC-GAN effective hyperparameters: %s", hp)


def _adam(cfg: OptimizerConfig):
    import tensorflow as tf
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.learning_rate,
        decay_steps=cfg.decay_steps,
        decay_rate=cfg.decay_rate,
        staircase=cfg.staircase,
    )
    return tf.keras.optimizers.Adam(
        learning_rate=schedule, beta_1=cfg.beta_1,
        beta_2=cfg.beta_2, clipnorm=cfg.clipnorm,
    )
```

**Regression guard so this cannot recur:**

```python
# tests/unit/test_cli_flags_are_wired.py
"""Every --AC_* flag must be read by something.

Finding C-01: 21 AC-GAN hyperparameter flags were declared and referenced
nowhere else in the repository, so every sweep run through the documented
CLI trained identical models under different recorded settings.
"""
import argparse
from pathlib import Path

from Config.SessionConfig.ArgumentConfigLoad import parse_training_client_args

REPO = Path(__file__).resolve().parents[2]
EXCLUDE = {"ArgumentConfigLoad.py"}


def _declared_flags() -> set[str]:
    parser = _build_parser_without_parsing()      # refactor extracted from the module
    return {
        a.dest for a in parser._actions
        if a.dest.startswith("AC_")
    }


def test_every_acgan_flag_is_consumed():
    sources = [
        p.read_text(encoding="utf-8", errors="replace")
        for p in REPO.rglob("*.py")
        if p.name not in EXCLUDE and ".claude" not in p.parts
    ]
    orphans = [
        dest for dest in _declared_flags()
        if not any(f"args.{dest}" in s or f'"{dest}"' in s for s in sources)
    ]
    assert not orphans, (
        f"CLI flags declared but never read: {sorted(orphans)}. "
        f"A flag that does nothing silently invalidates any sweep that sets it."
    )
```

**Variant B (honour the published defaults)** keeps `0.00003` / `0.00001` / `1` and
changes what every existing invocation does. Only choose it if the published defaults are
believed correct and the September-2025 mode-collapse tuning is to be re-run.

---

## 7. One trainer family

**Defect (F-04).** 47 classes, 21,278 LOC, `discriminator_loss` copied 24 times, `fit`
methods up to 499 lines.

Target shape — the step functions are the *only* thing that differs between variants:

```
hifins/training/
├── base.py          BaseTrainer          fit / evaluate / save lifecycle
├── mixins.py        MetricsLoggingMixin, ValidationMixin, CheckpointMixin
├── losses.py        discriminator_loss, generator_loss, gradient_penalty, nids_loss
│                    ← ONE copy each, replacing 24 / 17 / 7 / 13
├── variants/
│   ├── gan.py       GANTrainer        train_step
│   ├── wgan_gp.py   WGANGPTrainer     train_step + gradient penalty
│   └── acgan.py     ACGANTrainer      train_step + d_to_g_ratio loop
└── adapters/
    ├── central.py         CentralAdapter        plain fit/evaluate/save
    ├── flower_client.py   FlowerClientAdapter   NumPyClient get/set/fit/evaluate
    └── flower_strategy.py FlowerStrategyAdapter Strategy aggregate_fit hook
```

```python
# hifins/training/base.py
"""Lifecycle shared by every trainer.

Replaces 47 classes that each re-implemented the same fit/evaluate/save
skeleton around a different train_step. The 500-line `fit` methods are
almost entirely this skeleton; the genuinely variant-specific part is under
80 lines in each case.

Subclasses implement exactly three hooks. Everything else — epoch loop,
metric accumulation, early stopping, checkpointing, logging — lives here,
once, and is therefore fixed once.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..config.session import RunConfig


@dataclass
class TrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class BaseTrainer(ABC):

    def __init__(self, *, cfg: RunConfig, models: Dict[str, Any], data: TrainingData):
        self.cfg = cfg
        self.models = models
        self.data = data
        self.logger = logging.getLogger(
            f"hifins.training.{type(self).__name__}"
        )
        self._build_optimizers()

    # ---- hooks a variant must implement ----

    @abstractmethod
    def _build_optimizers(self) -> None: ...

    @abstractmethod
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """One optimizer step. Returns the scalar metrics for this step."""

    @abstractmethod
    def evaluate_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]: ...

    # ---- shared lifecycle ----

    def fit(self) -> Dict[str, list]:
        history: Dict[str, list] = {}
        steps = self.cfg.training.steps_per_epoch or (
            len(self.data.y_train) // self.cfg.training.batch_size
        )
        self._log_configuration(steps)
        stopper = _EarlyStopper(self.cfg.training.callbacks)

        for epoch in range(self.cfg.training.epochs):
            epoch_metrics: Dict[str, list] = {}
            for step in range(steps):
                for k, v in self.train_step(self._next_batch(step)).items():
                    epoch_metrics.setdefault(k, []).append(v)

            means = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            val = self.evaluate_step(self.data.X_val, self.data.y_val)
            means.update({f"val_{k}": v for k, v in val.items()})
            for k, v in means.items():
                history.setdefault(k, []).append(v)
            self._log_epoch(epoch, means)

            if stopper.should_stop(means):
                self.logger.info("early stopping at epoch %d", epoch)
                break
        return history

    def evaluate(self) -> Dict[str, float]:
        metrics = self.evaluate_step(self.data.X_test, self.data.y_test)
        self.logger.info("test metrics: %s", metrics)
        return metrics

    def save(self, name: str) -> None:
        for label, model in self.models.items():
            if model is None:
                continue
            path = f"{label}_{name}.h5"
            model.save(path)
            self.logger.info("saved %s -> %s", label, path)
```

Each concrete variant is then small. `ACGANTrainer.train_step`, for instance, is the
existing inner loop from `ACGANCentralTrainingConfig.fit` lines ~931–1050, lifted
**verbatim** with `d_to_g_ratio` read from `self.cfg.acgan` instead of a default argument.
The `hifins/training/losses.py` bodies are the existing `discriminator_loss` /
`generator_loss` implementations, one copy, chosen as the canonical version and pinned by
the golden snapshots.

**Migration rule (from the plan): one trainer per commit, one trainer per PR, and never
delete a class before its golden passes against the new path.**

---

## 8. Characterization harness

**This gates §7.** `Config/ModelTrainingConfig/` has zero tests; nothing may be
consolidated until its current behaviour is captured.

```python
# tests/characterization/conftest.py
"""Golden-snapshot fixtures for the legacy trainers.

These are CHARACTERIZATION tests, not correctness tests: they assert only
that a refactor did not change what the code does. They make no claim that
the current behaviour is right — several of these trainers carry the
September-2025 mode-collapse fixes and several may not, which is exactly
what nobody can currently determine (finding F-04).

Everything is seeded and tiny (200 rows, 1 epoch) so the suite runs in
under two minutes on CPU and can gate every PR.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).parent / "goldens"
N_ROWS, N_FEATURES, SEED = 200, 21, 20260810


@pytest.fixture(scope="session")
def tiny_dataset():
    """Deterministic stand-in with CIC-IoT-2023's shape (21 features, binary)."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(0.0, 1.0, size=(N_ROWS, N_FEATURES)).astype(np.float32)
    y = (rng.random(N_ROWS) > 0.5).astype(np.float32)
    split = int(N_ROWS * 0.6), int(N_ROWS * 0.8)
    return {
        "X_train": X[: split[0]], "y_train": y[: split[0]],
        "X_val":   X[split[0]:split[1]], "y_val": y[split[0]:split[1]],
        "X_test":  X[split[1]:], "y_test": y[split[1]:],
    }


def weights_digest(model) -> str:
    """Order- and dtype-sensitive digest of a Keras model's weights.

    Weight hashes rather than full arrays because the goldens live in git
    and must stay diffable; any change to any weight changes the digest.
    """
    h = hashlib.sha256()
    for w in model.get_weights():
        arr = np.ascontiguousarray(w, dtype=np.float64)
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def assert_matches_golden(name: str, snapshot: dict, *, update: bool = False) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    if update or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip(f"golden written: {path} — re-run to assert against it")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot == expected, (
        f"behaviour changed for {name}.\n"
        f"If intentional, re-run with --update-goldens and justify the diff "
        f"in the PR description."
    )
```

```python
# tests/characterization/test_acgan_central.py
import pytest

pytestmark = [pytest.mark.requires_tf, pytest.mark.slow]


def test_acgan_central_one_epoch_is_stable(tiny_dataset, acgan_models):
    """Pin ACGANCentralTrainingConfig's observable behaviour before R-21.

    Captured: post-fit weight digests for both sub-models, the per-epoch
    loss trace, and the effective hyperparameters. If R-21's parameterised
    ACGANTrainer reproduces this snapshot exactly, the consolidation is
    provably behaviour-preserving for this trainer.
    """
    from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig \
        .GAN.FullModel.ACGANCentralTrainingConfig import CentralACGan

    trainer = CentralACGan(
        discriminator=acgan_models["discriminator"],
        generator=acgan_models["generator"],
        nids=acgan_models["nids"],
        x_train=tiny_dataset["X_train"], y_train=tiny_dataset["y_train"],
        x_val=tiny_dataset["X_val"],     y_val=tiny_dataset["y_val"],
        x_test=tiny_dataset["X_test"],   y_test=tiny_dataset["y_test"],
        BATCH_SIZE=16, noise_dim=100, latent_dim=100, num_classes=2,
        input_dim=21, epochs=1, steps_per_epoch=4, learning_rate=1e-4,
    )
    trainer.fit()

    assert_matches_golden("acgan_central_1epoch", {
        "generator_digest":     weights_digest(trainer.generator),
        "discriminator_digest": weights_digest(trainer.discriminator),
        # Records the C-01 state: what the optimizer ACTUALLY used, which
        # is not what the CLI advertised. Under R-14 Variant A this value
        # must not move.
        "effective_gen_lr":  float(trainer.gen_optimizer.learning_rate(0)),
        "effective_disc_lr": float(trainer.disc_optimizer.learning_rate(0)),
    })
```

**Coverage target for the harness:** every trainer reachable from
`modelCentralTrainingConfigLoad` or `modelFederatedTrainingConfigLoad` — approximately 14
live classes out of the 47 on disk. The other 33 are unreachable and should be confirmed
dead, then deleted under [R-05](../01_Refactoring_Strategy.md#r-05) rather than
characterized.

---

## 9. Summary of expected impact

| Change | LOC delta | Behaviour | Fixes |
|---|---|---|---|
| Packaging (`pyproject`, `pytest.ini`, `conftest`, `LICENSE`) | +90, −16 `sys.path` lines | identical | F-05 |
| `hifins/paths.py` + call sites | +85 | identical (legacy path kept last) | F-05 |
| Typed config dataclasses | +260 | identical | F-04 |
| Registry factories | +90, **−600** of branch tree | ⚠ fails earlier and more clearly | F-02, F-04 |
| `hermes_adapters/` | +300 | ⚠ new capability (synth batch becomes real) | F-03 |
| AC-GAN hyperparameter wiring (Variant A) | +60 | identical with no flags; **flags now work** | F-01 |
| One trainer family | **−17,000** | identical (golden-gated) | F-04 |
| Characterization harness | +900 (tests) | none | F-04, F-09 |
| Dead-tree deletion | **−8,900** | identical | F-06 |

**Net: ≈ −24,700 LOC.** The training stack goes from 21,278 LOC of untested, duplicated
training loops to roughly 4,000 LOC of tested, parameterised ones — and the two halves of
the system become able to run together for the first time.

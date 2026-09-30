"""Canonical snapshots of runtime values, and the comparison the pins use.

FeRRy Phase 3, unit UG: the legacy behaviour of afa9526 is recorded as JSON
(``tests/golden/data``) and every later unit must reproduce it (Freeze Rule 1).
Two properties matter more than compactness:

* **Exact.** Floats are stored as ``"f:" + repr(x)``, so a pin compares the
  exact binary value (``repr`` round-trips, and keeps ``-0.0``, ``nan`` and
  ``inf``). Numpy arrays are stored as shape, dtype and a SHA-256 of their
  bytes.
* **Additive-field tolerant (critic A3).** Phase 3 adds default-valued fields
  to messages, lines and configs (``solicit_id``, ``in_reply_to``,
  ``uplink_drop``, ``snr_db``, ``band``, ``bytes_sent``, ...). A dataclass is
  stored with the field set it had when captured, and :func:`diff` projects the
  current object onto that set: a new field is ignored, a removed or renamed
  one is a mismatch. The same holds for :func:`record` (event payloads, CSV
  rows). Plain dicts, lists and scalars must match exactly.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

#: The commit whose behaviour the fixtures pin.
BASE_COMMIT = "afa952682c1e8a30160a390397f7f369a898b584"

TYPE_KEY = "__type__"
_MISSING = object()


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #

def f(x: float) -> str:
    """A float as its exact repr, tagged so it never collides with a string."""
    return "f:" + repr(float(x))


def array_digest(a: np.ndarray) -> Dict[str, Any]:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes())
    return {"__nd__": list(a.shape), "dtype": str(a.dtype), "sha": h.hexdigest()[:20]}


def canon(value: Any) -> Any:
    """JSON-able canonical form of ``value`` (see the module docstring)."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return f"e:{type(value).__name__}.{value.name}"
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return f(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, np.ndarray):
        return array_digest(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: Dict[str, Any] = {TYPE_KEY: type(value).__name__}
        for fld in dataclasses.fields(value):
            out[fld.name] = canon(getattr(value, fld.name))
        return out
    if isinstance(value, Mapping):
        return {str(canon_key(k)): canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canon(v) for v in value), key=_sort_key)
    raise TypeError(f"no canonical form for {type(value).__name__}: {value!r}")


def canon_key(k: Any) -> str:
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__name__}.{k.name}"
    if isinstance(k, float):
        return f(k)
    return str(k)


def record(kind: str, mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """A mapping compared like a dataclass: later keys may be added."""
    out: Dict[str, Any] = {TYPE_KEY: kind}
    for k, v in mapping.items():
        out[str(k)] = canon(v)
    return out


def _sort_key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def digest(value: Any, n: int = 16) -> str:
    """Short SHA-256 of an already-canonical value."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]


# --------------------------------------------------------------------------- #
# Projection and comparison
# --------------------------------------------------------------------------- #

def type_schemas(value: Any, out: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """Field set of every dataclass/record type found anywhere in ``value``."""
    if out is None:
        out = {}
    if isinstance(value, dict):
        if TYPE_KEY in value:
            keys = out.setdefault(value[TYPE_KEY], [])
            for k in value:
                if k not in keys:
                    keys.append(k)
        for v in value.values():
            type_schemas(v, out)
    elif isinstance(value, list):
        for v in value:
            type_schemas(v, out)
    return out


def project(current: Any, schemas: Mapping[str, List[str]]) -> Any:
    """``current`` with every record cut down to the golden field set of its type.

    Used for order-free comparisons, where elements are sorted by their JSON
    text: projecting first (at any depth) keeps a later default-valued field
    from changing that text. A golden field missing from the current record
    stays missing, so the comparison that follows reports it.
    """
    if isinstance(current, dict):
        keys = schemas.get(current.get(TYPE_KEY)) if TYPE_KEY in current else None
        if keys is not None:
            return {k: project(current[k], schemas) for k in keys if k in current}
        return {k: project(v, schemas) for k, v in current.items()}
    if isinstance(current, list):
        return [project(v, schemas) for v in current]
    return current


def diff(golden: Any, current: Any, path: str = "$", limit: int = 25) -> List[str]:
    """Mismatches between a golden and a current canonical value.

    Dataclass/record dicts (carrying ``__type__``) are compared on the golden's
    keys only; everything else must match exactly.
    """
    out: List[str] = []
    _diff(golden, current, path, out, limit)
    return out


def _short(v: Any) -> str:
    s = json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
    return s if len(s) <= 160 else s[:157] + "..."


def _diff(g: Any, c: Any, path: str, out: List[str], limit: int) -> None:
    if len(out) >= limit:
        return
    if c is _MISSING:
        out.append(f"{path}: missing in current (golden {_short(g)})")
        return
    if isinstance(g, dict):
        if not isinstance(c, dict):
            out.append(f"{path}: golden is a mapping, current is {_short(c)}")
            return
        if TYPE_KEY in g:
            if c.get(TYPE_KEY) != g[TYPE_KEY]:
                out.append(f"{path}: type {g[TYPE_KEY]} became {c.get(TYPE_KEY)}")
                return
            for k in g:
                if k == TYPE_KEY:
                    continue
                _diff(g[k], c.get(k, _MISSING), f"{path}.{k}", out, limit)
            return
        gk, ck = set(g), set(c)
        if gk != ck:
            out.append(
                f"{path}: keys differ: missing={sorted(gk - ck)[:8]} extra={sorted(ck - gk)[:8]}"
            )
        for k in g:
            if k in c:
                _diff(g[k], c[k], f"{path}[{k!r}]", out, limit)
        return
    if isinstance(g, list):
        if not isinstance(c, list):
            out.append(f"{path}: golden is a list, current is {_short(c)}")
            return
        if len(g) != len(c):
            out.append(f"{path}: length {len(g)} became {len(c)}")
        for i, (gi, ci) in enumerate(zip(g, c)):
            _diff(gi, ci, f"{path}[{i}]", out, limit)
        return
    if g != c or type(g) is not type(c):
        out.append(f"{path}: golden {_short(g)} != current {_short(c)}")


def assert_same(golden: Any, current: Any, what: str) -> None:
    problems = diff(golden, current)
    if problems:
        raise AssertionError(
            f"{what}: legacy behaviour pinned at afa9526 changed "
            f"({len(problems)} mismatch(es) shown):\n  " + "\n  ".join(problems)
        )


def order_free(golden_items: List[Any], current_items: List[Any]) -> tuple:
    """Both lists as sorted JSON texts, the current one projected first.

    Every record in the current elements, at any depth, is cut down to the
    field set its type has in the golden elements, so additive fields do not
    reorder or break the comparison (critic A3).
    """
    schemas = type_schemas(golden_items)
    return (sorted(_sort_key(g) for g in golden_items),
            sorted(_sort_key(project(c, schemas)) for c in current_items))


def mask_floats(value: Any, lo: float, hi: float, token: str = "f:<ts>") -> Any:
    """Replace every canonical float in ``[lo, hi]`` by ``token``.

    Stamps from a clock that real threads advance in a nondeterministic order
    are masked this way before an order-free comparison.
    """
    if isinstance(value, str) and value.startswith("f:"):
        try:
            x = float(value[2:])
        except ValueError:
            return value
        if not math.isnan(x) and lo <= x <= hi:
            return token
        return value
    if isinstance(value, dict):
        return {k: mask_floats(v, lo, hi, token) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_floats(v, lo, hi, token) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Fixture files
# --------------------------------------------------------------------------- #

def fixture_path(name: str) -> Path:
    return DATA_DIR / f"{name}.json"


def load(name: str) -> Dict[str, Any]:
    path = fixture_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"golden fixture {path} is missing; it is captured at afa9526 by "
            f"tests/golden/make_goldens.py and must never be regenerated from "
            f"changed code"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump(name: str, meta: Mapping[str, Any], cases: Mapping[str, Any]) -> Path:
    """Write a fixture: one case per line, so a diff points at the case."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = fixture_path(name)
    lines = ["{", '"_meta": ' + json.dumps(meta, sort_keys=True) + ",", '"cases": {']
    items = list(cases.items())
    for i, (key, val) in enumerate(items):
        sep = "," if i < len(items) - 1 else ""
        lines.append(
            json.dumps(key) + ": "
            + json.dumps(val, separators=(",", ":"), ensure_ascii=True) + sep
        )
    lines.append("}")
    lines.append("}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def meta(unit: str, **extra: Any) -> Dict[str, Any]:
    out = {"base_commit": BASE_COMMIT, "unit": unit}
    out.update(extra)
    return out


def iter_cases(fixture: Mapping[str, Any]) -> Iterable[str]:
    return list(fixture["cases"].keys())

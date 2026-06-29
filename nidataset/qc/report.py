"""
Report data model for the nidataset quality-control module.

Every check produces a :class:`CheckResult` (``ok`` / ``warning`` / ``error`` +
message + observed value). Results are grouped into reports that know their own
worst status and serialize cleanly to plain ``dict`` / JSON, so the same objects
power both the Python API and the ``niqc`` CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Status levels, ordered from best to worst. Kept as plain strings so reports
# stay JSON-serializable without custom encoders.
OK = "ok"
WARNING = "warning"
ERROR = "error"

_SEVERITY = {OK: 0, WARNING: 1, ERROR: 2}


def worst_status(statuses: List[str]) -> str:
    """Return the most severe status in *statuses* (``ok`` if empty)."""
    return max(statuses, key=lambda s: _SEVERITY[s], default=OK)


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays and tuples into JSON-friendly Python types."""
    # Imported lazily so report.py has no hard numpy dependency at import time.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is always present in practice
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass
class CheckResult:
    """
    Outcome of a single QC check.

    :param name:    Stable identifier of the check (e.g. ``"affine_nonsingular"``).
    :param status:  One of :data:`OK`, :data:`WARNING`, :data:`ERROR`.
    :param message: Human-readable explanation of the outcome.
    :param value:   Observed value that drove the status (JSON-serializable).
    """

    name: str
    status: str
    message: str
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable ``dict`` of this result."""
        d = asdict(self)
        d["value"] = _to_jsonable(self.value)
        return d


@dataclass
class Report:
    """
    Collection of :class:`CheckResult` for one target (volume, pair or triple).

    :param target:  Identifier of what was checked (path, or ``"img|mask"``).
    :param kind:    ``"volume"``, ``"pair"`` or ``"triple"``.
    :param results: Individual check outcomes.
    :param meta:    Free-form context (shape, dtype, orientation, ...).
    """

    target: str
    kind: str
    results: List[CheckResult] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, message: str, value: Any = None) -> CheckResult:
        """Append a :class:`CheckResult` and return it."""
        result = CheckResult(name=name, status=status, message=message, value=value)
        self.results.append(result)
        return result

    @property
    def status(self) -> str:
        """Worst status across all contained checks."""
        return worst_status([r.status for r in self.results])

    def counts(self) -> Dict[str, int]:
        """Return ``{ok, warning, error}`` tallies across checks."""
        out = {OK: 0, WARNING: 0, ERROR: 0}
        for r in self.results:
            out[r.status] += 1
        return out

    def issues(self) -> List[CheckResult]:
        """Return only the warning/error results (the ones worth reading)."""
        return [r for r in self.results if r.status != OK]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable ``dict`` of the whole report."""
        return {
            "target": self.target,
            "kind": self.kind,
            "status": self.status,
            "counts": self.counts(),
            "meta": _to_jsonable(self.meta),
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class DatasetReport:
    """
    Aggregate report over many per-item :class:`Report` objects.

    :param root:          Folder / CSV the dataset was read from.
    :param items:         Per-item reports, in deterministic (sorted) order.
    :param distributions: Cross-dataset summaries (orientations, spacings, ...).
    """

    root: str
    items: List[Report] = field(default_factory=list)
    distributions: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Worst status across all items."""
        return worst_status([it.status for it in self.items])

    def counts(self) -> Dict[str, int]:
        """Per-item status tally (one count per item, by its worst status)."""
        out = {OK: 0, WARNING: 0, ERROR: 0}
        for it in self.items:
            out[it.status] += 1
        return out

    def worst_first(self) -> List[Report]:
        """Items sorted worst status first, ties keep input (sorted) order."""
        return sorted(self.items, key=lambda it: -_SEVERITY[it.status])

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable ``dict`` of the dataset report."""
        return {
            "root": self.root,
            "status": self.status,
            "counts": self.counts(),
            "n_items": len(self.items),
            "distributions": _to_jsonable(self.distributions),
            "items": [it.to_dict() for it in self.items],
        }


def to_json(report: Any, path: Optional[str] = None, indent: int = 2) -> str:
    """
    Serialize a :class:`Report` / :class:`DatasetReport` (or its ``dict``) to JSON.

    :param report: Object exposing ``to_dict()`` or a plain ``dict``.
    :param path:   If given, also write the JSON to this file.
    :param indent: JSON indentation.

    :returns: The JSON string.
    """
    data = report.to_dict() if hasattr(report, "to_dict") else report
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text

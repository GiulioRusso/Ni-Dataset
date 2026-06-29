"""
Configuration for the nidataset quality-control module.

All thresholds and rules live in :class:`QCConfig` with generic, domain-neutral
defaults (no CT/MR-specific assumptions). Override them in Python::

    QCConfig(expected_orientation="RAS", affine_atol=1e-3)

or load them from a ``.json`` (stdlib) / ``.yaml`` (requires ``pyyaml``) file::

    QCConfig.load("qc.yaml")

See ``qc.example.yaml`` for a commented template and ``qc.example.json`` for a
zero-dependency equivalent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class QCConfig:
    """
    Thresholds and rules for QC checks. Every value is user-overridable.

    Geometry
    --------
    :param expected_orientation: Expected ``aff2axcodes`` orientation as a 3-char
        string (e.g. ``"RAS"``). ``None`` disables the check (orientation is still
        always reported).
    :param affine_atol:          Absolute tolerance when comparing affines
        (singularity, sform/qform agreement, pair/triple alignment).
    :param isotropy_tol:         Allowed spacing anisotropy as ``max/min - 1``.
        ``0.05`` means up to 5% difference between the largest and smallest voxel
        edge is still considered isotropic.
    :param spacing_range:        Plausible per-axis voxel size ``(min, max)`` in mm.

    Data
    ----
    :param intensity_range:      Plausible intensity ``(min, max)``; a volume whose
        min/max falls outside triggers a warning. ``None`` disables it.
    :param warn_float64:         Warn when data dtype is ``float64`` (float32 usually
        suffices and halves size).
    :param empty_slice_bg_value: Voxels ``<=`` this value count as background when
        deciding whether a slice is "empty".
    :param empty_slice_min_fg_fraction: A slice is "empty" if the fraction of
        foreground voxels (``> empty_slice_bg_value``) is below this.
    :param max_empty_slice_fraction: Warn if, on any axis, more than this fraction
        of slices are empty (padding / corruption signal).

    Pair / triple
    -------------
    :param allowed_labels:       Label set a mask/annotation may contain. ``None``
        means "binary" (``{0, 1}``).
    :param max_annotation_outside_fraction: Error if more than this fraction of
        annotation voxels falls outside the mask (brain).
    :param min_component_size:   Connected components smaller than this (voxels) are
        flagged as possibly spurious.
    :param max_components:       Warn if the annotation has more than this many
        connected components (fragmentation signal).
    :param warn_bbox_touches_border: Warn if the annotation bounding box touches the
        volume border (possible clipping).

    Dataset
    -------
    :param max_workers:          Thread workers for dataset scans (output stays
        deterministic regardless). ``1`` disables parallelism.
    """

    # Geometry
    expected_orientation: Optional[str] = None
    affine_atol: float = 1e-4
    isotropy_tol: float = 0.05
    spacing_range: Tuple[float, float] = (0.1, 10.0)

    # Data
    intensity_range: Optional[Tuple[float, float]] = None
    warn_float64: bool = True
    empty_slice_bg_value: float = 0.0
    empty_slice_min_fg_fraction: float = 1e-4
    max_empty_slice_fraction: float = 0.5

    # Pair / triple
    allowed_labels: Optional[Sequence[float]] = None
    max_annotation_outside_fraction: float = 0.0
    min_component_size: int = 1
    max_components: int = 50
    warn_bbox_touches_border: bool = True

    # Dataset
    max_workers: int = 4

    def __post_init__(self) -> None:
        if self.expected_orientation is not None:
            self.expected_orientation = self.expected_orientation.upper()
            if len(self.expected_orientation) != 3:
                raise ValueError(
                    f"expected_orientation must be 3 characters (e.g. 'RAS'), "
                    f"got {self.expected_orientation!r}"
                )
        # Normalize tuples that may arrive as lists from JSON/YAML.
        if self.spacing_range is not None:
            self.spacing_range = tuple(self.spacing_range)  # type: ignore[assignment]
        if self.intensity_range is not None:
            self.intensity_range = tuple(self.intensity_range)  # type: ignore[assignment]
        if self.allowed_labels is not None:
            self.allowed_labels = tuple(self.allowed_labels)

    @property
    def label_set(self) -> Tuple[float, ...]:
        """Resolved set of allowed labels (defaults to binary ``(0, 1)``)."""
        return tuple(self.allowed_labels) if self.allowed_labels is not None else (0.0, 1.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QCConfig":
        """
        Build a :class:`QCConfig` from a plain ``dict``, ignoring unknown keys.

        :raises ValueError: If a key is unknown (typo protection).
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown QCConfig keys: {sorted(unknown)}. Valid keys: {sorted(known)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "QCConfig":
        """
        Load config from ``.json`` (stdlib) or ``.yaml`` / ``.yml`` (needs ``pyyaml``).

        :raises FileNotFoundError: If *path* does not exist.
        :raises ImportError:       If a YAML file is given but ``pyyaml`` is missing.
        :raises ValueError:        On unknown keys or unsupported extension.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: '{path}'")
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # ponytail: yaml is optional, JSON works dep-free
                raise ImportError(
                    "YAML config requires the optional 'pyyaml' package. "
                    "Install it (`pip install pyyaml`) or use a .json config instead."
                ) from exc
            data = yaml.safe_load(text) or {}
        elif ext == ".json":
            data = json.loads(text) if text.strip() else {}
        else:
            raise ValueError(f"Unsupported config extension '{ext}'. Use .json, .yaml or .yml.")
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
        return cls.from_dict(data)

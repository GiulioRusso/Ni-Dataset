"""
Pipeline config schema.

A pipeline file (``.yaml`` / ``.yml`` with PyYAML, or ``.json``) looks like::

    dataset:                      # named input paths, relative to the config file
      images: data/images
      masks: data/masks
    output: out                   # run folder, relative to the config file
    steps:                        # executed in order
      - op: extract_slices        # see nidataset.pipeline.OPS
        views: [axial, coronal]   # optional: run once per view
        nii_folder: "{images}"    # every other key is a kwarg of the op
        saving_mode: view

String kwargs may use ``{<dataset key>}``, ``{output}`` and ``{view}`` placeholders.
Each step writes to ``{output}/<name>`` (``name`` defaults to ``op``) unless it
sets ``output_path`` itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .._helpers import load_mapping

VIEWS = ("axial", "coronal", "sagittal")
_STEP_KEYS = ("op", "name", "views")


class PipelineConfigError(ValueError):
    """Raised when a pipeline config is invalid. Lists every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid pipeline config:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass
class Step:
    op: str
    name: str
    views: list[str] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    dataset: dict[str, str]
    output: str
    steps: list[Step]
    base_dir: str = "."  # literal relative paths in step kwargs resolve against this
    raw: dict[str, Any] = field(default_factory=dict)  # as loaded, for hashing / manifest

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str = ".") -> PipelineConfig:
        """
        Parse and structurally validate *data*. Paths are resolved against *base_dir*.

        Op names, kwargs and file existence are checked later by the runner
        (it owns the op registry).

        :raises PipelineConfigError: Listing every structural problem.
        """
        problems: list[str] = []
        unknown = set(data) - {"dataset", "output", "steps"}
        if unknown:
            problems.append(f"unknown top-level keys {sorted(unknown)} (expected dataset, output, steps)")

        dataset = data.get("dataset") or {}
        if not isinstance(dataset, dict) or not all(isinstance(v, str) for v in dataset.values()):
            problems.append("'dataset' must map names to path strings")
            dataset = {}
        reserved = {"output", "view"} & set(dataset)
        if reserved:
            problems.append(f"dataset keys {sorted(reserved)} are reserved placeholders; rename them")

        output = data.get("output")
        if not isinstance(output, str) or not output:
            problems.append("'output' must be a non-empty path string")
            output = ""

        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            problems.append("'steps' must be a non-empty list")
            raw_steps = []

        steps: list[Step] = []
        for i, raw in enumerate(raw_steps, 1):
            if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
                problems.append(f"step {i}: must be a mapping with an 'op' string")
                continue
            name = raw.get("name", raw["op"])
            views = raw.get("views")
            if views is not None:
                if isinstance(views, str):
                    views = [views]
                bad = [v for v in views if v not in VIEWS] if isinstance(views, list) else [views]
                if bad:
                    problems.append(f"step {i} ({name}): invalid views {bad}; choose from {list(VIEWS)}")
            kwargs = {k: v for k, v in raw.items() if k not in _STEP_KEYS}
            steps.append(Step(op=raw["op"], name=str(name), views=views, kwargs=kwargs))

        names = [s.name for s in steps]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            problems.append(f"duplicate step names {dupes}; give repeated ops a unique 'name'")

        if problems:
            raise PipelineConfigError(problems)

        def resolve(p: str) -> str:
            return os.path.abspath(os.path.join(base_dir, os.path.expanduser(p)))

        return cls(
            dataset={k: resolve(v) for k, v in dataset.items()},
            output=resolve(output),
            steps=steps,
            base_dir=os.path.abspath(base_dir),
            raw=data,
        )

    @classmethod
    def load(cls, path: str) -> PipelineConfig:
        """Load a ``.yaml`` / ``.yml`` / ``.json`` pipeline; relative paths resolve against its folder."""
        return cls.from_dict(load_mapping(path), base_dir=os.path.dirname(os.path.abspath(path)))

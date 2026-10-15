"""
Validate and execute a pipeline config, writing ``run_manifest.json``.

Everything that can be checked without running (op names, kwargs against the
function signature, placeholders, input paths, external tools) is checked up
front, so a typo in step 5 fails before step 1 touches the disk.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .._helpers import ensure_dir
from .._version import __version__
from .registry import OPS, REQUIRES
from .schema import PipelineConfig, PipelineConfigError

MANIFEST_NAME = "run_manifest.json"

_FSL_HINT = (
    "Install FSL (https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) and make sure $FSLDIR/bin is on "
    "PATH; launch from a terminal that sources FSL's profile (IDEs often miss it)."
)


class PipelineError(RuntimeError):
    """Raised when a step fails at run time. The manifest is still written."""


@dataclass
class Call:
    step: str
    op: str
    view: str | None
    func: Callable[..., Any]
    kwargs: dict[str, Any]


def _is_path_param(name: str) -> bool:
    return name != "output_path" and (name.endswith("_path") or name.endswith("_folder"))


def _inside(path: str, folder: str) -> bool:
    return os.path.commonpath([path, folder]) == folder


def plan(cfg: PipelineConfig) -> list[Call]:
    """
    Expand steps into concrete calls (one per view) and validate all of them.

    :raises PipelineConfigError: Listing every problem across all steps.
    """
    problems: list[str] = []
    calls: list[Call] = []
    for step in cfg.steps:
        where = f"step '{step.name}'"
        func = OPS.get(step.op)
        if func is None:
            problems.append(f"{where}: unknown op '{step.op}'; available: {sorted(OPS)}")
            continue
        missing_tools = [t for t in REQUIRES.get(step.op, ()) if shutil.which(t) is None]
        if missing_tools:
            problems.append(f"{where}: op '{step.op}' needs {missing_tools} on PATH (not found). {_FSL_HINT}")

        sig = inspect.signature(func)
        uses_view = "view" in sig.parameters or any(isinstance(v, str) and "{view}" in v for v in step.kwargs.values())
        if step.views and not uses_view:
            problems.append(f"{where}: 'views' given but op '{step.op}' takes no view and no kwarg uses {{view}}")
            continue

        for view in step.views or [None]:
            ctx = {**cfg.dataset, "output": cfg.output}
            if view:
                ctx["view"] = view
            kwargs: dict[str, Any] = {}
            for key, value in step.kwargs.items():
                if isinstance(value, str):
                    try:
                        value = value.format_map(ctx)
                    except KeyError as exc:
                        problems.append(
                            f"{where}: unknown placeholder {{{exc.args[0]}}} in '{key}'; available: {sorted(ctx)}"
                        )
                        continue
                    if _is_path_param(key):
                        value = os.path.abspath(os.path.join(cfg.base_dir, os.path.expanduser(value)))
                elif isinstance(value, list):
                    value = tuple(value)  # YAML/JSON lists -> the tuples the API documents
                kwargs[key] = value
            if view and "view" in sig.parameters:
                kwargs.setdefault("view", view)
            kwargs.setdefault("output_path", os.path.join(cfg.output, step.name))

            try:
                sig.bind(**kwargs)
            except TypeError as exc:
                problems.append(f"{where} ({step.op}): {exc}; accepted kwargs: {list(sig.parameters)}")
                continue

            for key, value in kwargs.items():
                # Inputs under `output` are produced by earlier steps, so they can't exist yet.
                if _is_path_param(key) and isinstance(value, str) and not os.path.exists(value):
                    if not _inside(value, cfg.output):
                        problems.append(f"{where}: {key} '{value}' does not exist")
            calls.append(Call(step.name, step.op, view, func, kwargs))

    if problems:
        raise PipelineConfigError(problems)
    return calls


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _count_files(path: str) -> int:
    return sum(len(files) for _, _, files in os.walk(path)) if os.path.isdir(path) else int(os.path.isfile(path))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_pipeline(
    config: str | os.PathLike | dict | PipelineConfig,
    base_dir: str | None = None,
    only: list[str] | None = None,
) -> dict:
    """
    Validate then run a pipeline; write ``<output>/run_manifest.json``.

    :param config:   Path to a ``.yaml`` / ``.yml`` / ``.json`` pipeline file, a dict with the
        same structure, or a :class:`PipelineConfig`.
    :param base_dir: For dict configs, the folder relative paths resolve against (default: cwd).
        File configs always resolve against their own folder.
    :param only:     Step names to run (config order is kept); the others are skipped.
        Use it to rerun one stage, e.g. ``only=["register"]``. Inputs under ``output``
        produced by skipped steps must already exist on disk.

    :returns: The manifest dict (also written to disk).

    :raises PipelineConfigError: If the config is invalid (nothing has run).
    :raises PipelineError:       If a step fails (the manifest records which).

    Example
    -------
    >>> import nidataset as nid
    >>> manifest = nid.run_pipeline("pipeline.yaml")
    >>> [(s["step"], s["seconds"]) for s in manifest["steps"]]
    """
    source = None
    if isinstance(config, PipelineConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = PipelineConfig.from_dict(config, base_dir=base_dir or os.getcwd())
    else:
        source = os.path.abspath(os.fspath(config))
        cfg = PipelineConfig.load(source)

    if only:
        names = [s.name for s in cfg.steps]
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise PipelineConfigError([f"unknown step names {unknown} in 'only'; available: {names}"])
        cfg = dataclasses.replace(cfg, steps=[s for s in cfg.steps if s.name in only])

    calls = plan(cfg)
    ensure_dir(cfg.output)
    manifest_path = os.path.join(cfg.output, MANIFEST_NAME)
    manifest: dict[str, Any] = {
        "nidataset_version": __version__,
        "python": sys.version.split()[0],
        "config": source,
        "config_sha256": hashlib.sha256(json.dumps(cfg.raw, sort_keys=True, default=str).encode()).hexdigest(),
        "output": cfg.output,
        "only": list(only) if only else None,
        "started": _now(),
        "finished": None,
        "status": "running",
        "steps": [],
    }

    def write() -> None:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    for call in calls:
        entry: dict[str, Any] = {
            "step": call.step,
            "op": call.op,
            "function": f"{call.func.__module__}.{call.func.__name__}",
            "view": call.view,
            "kwargs": _jsonable(call.kwargs),
        }
        manifest["steps"].append(entry)
        start = time.perf_counter()
        try:
            result = call.func(**call.kwargs)
        except Exception as exc:
            entry.update(status="failed", seconds=round(time.perf_counter() - start, 3), error=repr(exc))
            manifest.update(status="failed", finished=_now())
            write()
            view = f", view={call.view}" if call.view else ""
            raise PipelineError(f"step '{call.step}' ({call.op}{view}) failed: {exc}") from exc
        out = call.kwargs["output_path"]
        entry.update(
            status="ok",
            seconds=round(time.perf_counter() - start, 3),
            result=_jsonable(result),
            output_path=out,
            n_files=_count_files(out),
        )

    manifest.update(status="ok", finished=_now())
    write()
    return manifest

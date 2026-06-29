"""
``niqc`` — command-line entry point for nidataset quality control.

Wraps the same functions as the Python API. Auto-detects whether PATH is a single
NIfTI file, a folder, or a ``.csv`` manifest of pairs/triples, and prints a
coloured, human-readable report with a final summary.

Exit codes
----------
* ``0`` — completed; no ``error`` results, **or** errors present but ``--strict``
  not given.
* ``1`` — ``--strict`` given and at least one ``error`` result was found.
* ``2`` — usage error or a failure while running (bad path, unreadable config).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

from .config import QCConfig
from .report import Report, DatasetReport, to_json, OK, WARNING, ERROR

# Status -> (symbol, ANSI colour).
_SYMBOL = {OK: ("✓", "\033[32m"), WARNING: ("⚠", "\033[33m"), ERROR: ("✗", "\033[31m")}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _use_color(stream) -> bool:
    """Colour only when writing to a TTY and NO_COLOR is unset."""
    return stream.isatty() and os.environ.get("NO_COLOR") is None


def _paint(status: str, text: str, color: bool) -> str:
    symbol, ansi = _SYMBOL[status]
    if color:
        return f"{ansi}{symbol} {text}{_RESET}"
    return f"{symbol} {text}"


def _print_report(report: Report, color: bool, verbose: bool, stream) -> None:
    """Print one volume/pair/triple report (issues by default, all if verbose)."""
    header = _paint(report.status, f"{report.kind}: {report.target}", color)
    print(header, file=stream)
    shown = report.results if verbose else report.issues()
    for r in shown:
        print(f"    {_paint(r.status, f'{r.name}: {r.message}', color)}", file=stream)
    if not shown:
        print(f"    {_paint(OK, 'all checks passed', color)}", file=stream)


def _print_dataset(ds: DatasetReport, color: bool, verbose: bool, stream) -> None:
    """Print a dataset report: worst items first, then distributions."""
    for item in ds.worst_first():
        _print_report(item, color, verbose, stream)
    dist = ds.distributions
    print(f"\n{_BOLD if color else ''}Distributions{_RESET if color else ''}", file=stream)
    for key in ("orientation", "dtype", "shape", "spacing"):
        if dist.get(key):
            print(f"  {key}: {dist[key]}", file=stream)
    outliers = dist.get("outliers", {})
    for key, paths in outliers.items():
        if paths:
            print(f"  {key} outliers: {paths}", file=stream)


def _summary(counts: Dict[str, int], color: bool, stream) -> None:
    parts = [_paint(s, f"{counts.get(s, 0)} {s}", color) for s in (OK, WARNING, ERROR)]
    print(f"\n{_BOLD if color else ''}Summary:{_RESET if color else ''} " + "  ".join(parts),
          file=stream)


def _specs_from_reports(items: List[Report]) -> List[Dict[str, Optional[str]]]:
    """Build thumbnail specs from per-item report metadata."""
    specs: List[Dict[str, Optional[str]]] = []
    for it in items:
        if it.kind == "volume":
            specs.append({"image": it.target})
        else:
            specs.append({"image": it.meta.get("image"),
                          "mask": it.meta.get("mask"),
                          "annotation": it.meta.get("annotation")})
    return specs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="niqc",
        description="Quality control for NIfTI datasets (geometry, data, image/mask/annotation coherence).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  niqc scan.nii.gz                       # single volume\n"
            "  niqc scans/ --strict                   # folder, fail CI on any error\n"
            "  niqc triples.csv --json report.json    # CSV manifest -> JSON report\n"
            "  niqc --pair ct.nii.gz brain.nii.gz     # image<->mask coherence\n"
            "  niqc --triple ct.nii.gz brain.nii.gz lesion.nii.gz --thumbnails qc/\n"
            "  niqc scans/ --config qc.yaml           # custom thresholds\n"
        ),
    )
    p.add_argument("path", nargs="?", help="NIfTI file, folder, or .csv manifest of pairs/triples.")
    p.add_argument("--pair", nargs=2, metavar=("IMG", "MASK"), help="Explicit image/mask pair.")
    p.add_argument("--triple", nargs=3, metavar=("IMG", "MASK", "ANN"),
                   help="Explicit image/mask/annotation triple.")
    p.add_argument("--config", metavar="FILE", help="QCConfig from a .json or .yaml file.")
    p.add_argument("--json", nargs="?", const="-", metavar="FILE",
                   help="Emit the structured report as JSON (to FILE, or stdout if omitted).")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any check is an error (for CI / pre-commit).")
    p.add_argument("--thumbnails", metavar="DIR", help="Write a PNG thumbnail grid into DIR.")
    p.add_argument("--verbose", "-v", action="store_true", help="Show every check, not just issues.")
    p.add_argument("--no-color", action="store_true", help="Disable coloured output.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ``niqc`` CLI. Returns the process exit code."""
    # Imported here so check functions don't pull the CLI at module import.
    from .checks import check_volume
    from .pairs import check_pair, check_triple
    from .dataset import check_dataset

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.path and not args.pair and not args.triple:
        parser.error("provide a PATH, or --pair IMG MASK, or --triple IMG MASK ANN.")
    if sum(bool(x) for x in (args.path, args.pair, args.triple)) > 1:
        parser.error("PATH, --pair and --triple are mutually exclusive.")

    try:
        config = QCConfig.load(args.config) if args.config else QCConfig()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    color = not args.no_color and _use_color(sys.stdout)

    try:
        if args.pair:
            result = check_pair(args.pair[0], args.pair[1], config)
        elif args.triple:
            result = check_triple(args.triple[0], args.triple[1], args.triple[2], config)
        elif os.path.isdir(args.path) or args.path.lower().endswith(".csv"):
            result = check_dataset(args.path, config)
        else:
            result = check_volume(args.path, config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    is_dataset = isinstance(result, DatasetReport)

    # JSON output.
    if args.json is not None:
        if args.json == "-":
            print(to_json(result))
        else:
            to_json(result, args.json)
            print(f"JSON report written to {args.json}", file=sys.stderr)
    else:
        if is_dataset:
            _print_dataset(result, color, args.verbose, sys.stdout)
        else:
            _print_report(result, color, args.verbose, sys.stdout)
        _summary(result.counts(), color, sys.stdout)

    # Thumbnails.
    if args.thumbnails:
        from .thumbnails import thumbnail_grid
        items = result.items if is_dataset else [result]
        specs = _specs_from_reports(items)
        out_png = os.path.join(args.thumbnails, "qc_thumbnails.png")
        try:
            thumbnail_grid(specs, out_png)
            print(f"thumbnails written to {out_png}", file=sys.stderr)
        except ValueError as exc:
            print(f"thumbnail warning: {exc}", file=sys.stderr)

    # Exit code.
    if args.strict and result.status == ERROR:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

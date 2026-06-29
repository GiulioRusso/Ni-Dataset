---
title: niqc CLI
parent: Quality Control (qc)
nav_order: 7
---

# `niqc` — command-line interface

Installed as a `pyproject.toml` entry point. Wraps the same functions as the
Python API and auto-detects whether the target is a single file, a folder, or a
CSV manifest of pairs/triples.

```bash
niqc PATH [options]
```

## Arguments & options

| Option | Description |
|--------|-------------|
| `PATH` | A NIfTI file, a folder, or a `.csv` manifest. Auto-detected. |
| `--pair IMG MASK` | Explicit image/mask pair (runs [`check_pair`](check_pair.md)). |
| `--triple IMG MASK ANN` | Explicit image/mask/annotation triple (runs [`check_triple`](check_triple.md)). |
| `--config FILE` | Load [`QCConfig`](qcconfig.md) from a `.json` or `.yaml` file. |
| `--json [FILE]` | Emit the structured report as JSON (to `FILE`, or stdout if omitted). |
| `--strict` | Exit non-zero if any check is an `error` (for CI / pre-commit). |
| `--thumbnails DIR` | Write a PNG thumbnail grid (`qc_thumbnails.png`) into `DIR`. |
| `--verbose`, `-v` | Show every check, not just issues. |
| `--no-color` | Disable coloured output. |

`PATH`, `--pair` and `--triple` are mutually exclusive.

## Output

Human-readable and coloured: `✓` ok, `⚠` warning, `✗` error, with a final summary
line. For datasets, items are printed worst-first followed by the cross-dataset
distributions. Colour auto-disables when output is not a TTY or `NO_COLOR` is set.

## Exit codes

| Code | Condition |
|------|-----------|
| `0` | Completed; no `error`, **or** errors present but `--strict` not given. |
| `1` | `--strict` given and at least one `error` result was found. |
| `2` | Usage error, bad path, or unreadable config. |

This makes `niqc --strict` a clean CI / pre-commit gate before a training run:
warnings inform without failing, only genuine corruption blocks the pipeline.

## Examples

```bash
# Single volume, coloured report
niqc scan.nii.gz

# Folder, fail CI on any error
niqc scans/ --strict

# CSV manifest of triples -> JSON report for a dataset card / pipeline
niqc triples.csv --json report.json

# Explicit image <-> mask coherence
niqc --pair ct.nii.gz brain.nii.gz

# Triple + visual overlay thumbnails
niqc --triple ct.nii.gz brain.nii.gz lesion.nii.gz --thumbnails qc/

# Custom thresholds
niqc scans/ --config qc.yaml
```

### Pre-commit / CI snippet

```yaml
# fail the pipeline before training if the dataset has geometric errors
- name: Dataset QC
  run: niqc data/train/ --strict --json qc_report.json
```

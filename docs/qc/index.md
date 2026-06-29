---
title: Quality Control (qc)
nav_order: 4
has_children: true
---

# 🩺 Quality Control (`nidataset.qc`)

The `qc` sub-module validates the **geometric coherence** of NIfTI datasets for
detection / segmentation. It answers a question no viewer does:

> *Is this dataset geometrically coherent and trustworthy, or is something silently
> poisoning training without raising any error?*

It hunts the **silent** bugs — the ones that never crash, only give wrong results:
unexpected orientation (LAS vs RAS), a mask shifted a few voxels from its image,
empty annotations, anisotropic spacing, all-black border slices, non-portable
`int64` data, NaN/inf.

The module is exposed two ways, both first-class:

- **Python API** — `nid.qc.<function>(...)`, each returning an inspectable,
  JSON-serializable report.
- **CLI** — the `niqc` command (installed as a `pyproject.toml` entry point).

## Functions

| Function | Purpose |
|----------|---------|
| [`check_volume`](check_volume.md)   | All single-volume checks: geometry, data, file. |
| [`check_pair`](check_pair.md)       | Image ↔ mask coherence (shape, world-space alignment, labels). |
| [`check_triple`](check_triple.md)   | Image ↔ mask ↔ annotation coherence (+ containment, components). |
| [`check_dataset`](check_dataset.md) | Per-item checks + cross-dataset distributions over a folder/CSV. |
| [`thumbnail_grid`](thumbnail_grid.md) | PNG grid of central orthogonal slices with mask/annotation overlay. |
| [`QCConfig`](qcconfig.md)           | User-overridable thresholds for every check. |
| [`niqc` CLI](cli.md)                | Command-line wrapper with `--strict`, `--json`, `--thumbnails`, `--config`. |

## Quick start

```python
import nidataset as nid

rep = nid.qc.check_volume("scan.nii.gz")
print(rep.status)                      # 'ok' | 'warning' | 'error'
print([r.name for r in rep.issues()])  # only the problems
nid.qc.to_json(rep, "report.json")

# Image/mask/annotation coherence (the high-value path)
rep = nid.qc.check_triple("ct.nii.gz", "brain.nii.gz", "lesion.nii.gz")
```

```bash
niqc scans/ --strict --json report.json
```

## Report objects

| Object | Description |
|--------|-------------|
| `CheckResult(name, status, message, value)` | One check outcome. `status` ∈ `ok`/`warning`/`error`. |
| `Report(target, kind, results, meta)`       | One volume/pair/triple. `status` = worst result; `issues()` = warnings+errors; `meta` carries shape/dtype/orientation/spacing. |
| `DatasetReport(root, items, distributions)` | Many items + cross-dataset summaries. `worst_first()` ranks items. |
| `to_json(report, path=None)`                | Serialize any report to a JSON string (and optional file). |

## Severity & exit codes

`ok` = passed, `warning` = suspicious but possibly intentional, `error` = almost
certainly corrupts training. `niqc --strict` returns exit code **1** only when an
`error` is present (warnings never fail the build), **2** on usage/IO errors,
**0** otherwise. Full rationale and default thresholds are in
[`QC_DESIGN.md`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/docs/QC_DESIGN.md).

## Geometric convention

All geometry is in **nibabel** space: voxel indices `(i, j, k)` and a 4×4 affine
mapping voxel → world (RAS+). Orientation comes from `nibabel.aff2axcodes`.
SimpleITK's reversed `(z, y, x)` / LPS convention is **never** mixed in.

---
title: check_volume
parent: Quality Control (qc)
nav_order: 1
---

# `check_volume`

Run all single-volume QC checks on one NIfTI file: geometry, data and file
integrity.

```python
check_volume(
    nii_path: str,
    config: Optional[QCConfig] = None,
) -> Report
```

## Overview

Loads a NIfTI volume and runs every single-volume check, returning a `Report`
whose `status` is the worst outcome. The geometry checks are the core value: a
volume can open fine in any viewer yet be silently wrong (unexpected orientation,
singular affine, anisotropic spacing, disagreeing sform/qform).

## Checks performed

| Group | Check name | Status on failure | What it catches |
|-------|------------|-------------------|-----------------|
| File | `readable` | error | Corrupt/unreadable file. |
| File | `extension` | warning | Content without a `.nii`/`.nii.gz` extension. |
| File | `dimensionality` | warning (4D) / error (other) | 4D timeseries or unexpected ndim; 3D is `ok`. |
| Geometry | `affine_finite` | error | Missing affine or NaN/inf in it. |
| Geometry | `affine_nonsingular` | error | `|det|` of the direction matrix ≤ `affine_atol`. |
| Geometry | `orientation` | error | `aff2axcodes` differs from `expected_orientation`. |
| Geometry | `spacing_positive` | error | Non-positive / non-finite voxel sizes. |
| Geometry | `spacing_isotropy` | warning | `max/min - 1` spacing ratio above `isotropy_tol`. |
| Geometry | `spacing_range` | warning | Voxel size outside `spacing_range`. |
| Geometry | `sform_qform_agree` | warning | Stored sform and qform disagree beyond `affine_atol`. |
| Data | `finite_values` | error | NaN or inf voxels. |
| Data | `dtype` | warning | `int64`/`uint64` (non-portable) or `float64`. |
| Data | `constant_volume` | error | All-zero / constant volume. |
| Data | `intensity_range` | warning | min/max outside `intensity_range` (if set). |
| Data | `empty_slices` | warning | Empty-slice fraction on an axis above `max_empty_slice_fraction`. |

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `nii_path` | `str` | *required* | Path to the input `.nii` / `.nii.gz` file. |
| `config` | `QCConfig` | `QCConfig()` | Thresholds; see [`QCConfig`](qcconfig.md). |

## Returns

`Report` (`kind="volume"`). `report.status` is the worst check; `report.meta`
carries `shape`, `ndim`, `dtype`, `orientation`, `spacing`, `intensity` and
`empty_slices`.

## Exceptions

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | File does not exist. |
| `ValueError` | File is not a valid `.nii`/`.nii.gz`. |

(An unreadable-but-present file is reported as a `readable` **error**, not raised.)

## Examples

```python
from nidataset.qc import check_volume, QCConfig

# Expect RAS orientation
rep = check_volume("scan.nii.gz", QCConfig(expected_orientation="RAS"))
print(rep.status)                       # 'ok' | 'warning' | 'error'
for r in rep.issues():
    print(r.name, r.status, r.message)

print(rep.meta["orientation"], rep.meta["spacing"], rep.meta["dtype"])
```

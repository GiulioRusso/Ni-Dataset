---
title: rotate_volume
parent: Quick Operations (nii)
nav_order: 1
---

# `rotate_volume`

Rotate a NIfTI volume by 90-degree steps in a voxel plane, losslessly, updating the affine to preserve world space.

```python
rotate_volume(
    nii_path: str,
    output_path: str,
    k: int = 1,
    axes: tuple = (0, 1),
    debug: bool = False
) -> str
```

## Overview

Rotation is performed by relabeling voxels (`numpy.rot90`), so it is **lossless** —
no interpolation, no intensity change. The affine is recomputed from the exact
voxel remapping, so the rotated volume keeps its correct world coordinates and
orientation metadata. Only 90-degree multiples are supported; for arbitrary angles
use a resampling-based tool.

The output is saved as `<PREFIX>_rot<k>.nii.gz`.

## Parameters

| Name          | Type    | Default    | Description                                                        |
|---------------|---------|------------|--------------------------------------------------------------------|
| `nii_path`    | `str`   | *required* | Path to the input `.nii.gz` file.                                 |
| `output_path` | `str`   | *required* | Directory where the rotated volume will be saved.                 |
| `k`           | `int`   | `1`        | Number of 90-degree rotations (may be negative).                  |
| `axes`        | `tuple` | `(0, 1)`   | The two distinct voxel axes (from `0`, `1`, `2`) defining the plane. |
| `debug`       | `bool`  | `False`    | If `True`, logs the output path and parameters.                   |

## Returns

`str` – Path to the saved rotated file.

## Output File

```
<PREFIX>_rot<k>.nii.gz
```

**Example**: Input `scan.nii.gz` with `k=1` → Output `scan_rot1.nii.gz`

## Exceptions

| Exception           | Condition                                                    |
|---------------------|-------------------------------------------------------------|
| `FileNotFoundError` | Input file does not exist                                   |
| `ValueError`        | Input file is not a valid `.nii.gz` / `.nii`                |
| `ValueError`        | `axes` are not two distinct values from `(0, 1, 2)`         |

## Usage Notes

- **Lossless**: voxels are relabeled, not interpolated — intensities are unchanged.
- **`k=4`** (or any multiple of 4) returns the original volume and affine.
- **Affine-correct**: world coordinates of every voxel are preserved.

## Examples

### Rotate 90° in the axial plane

```python
from nidataset.spatial import rotate_volume

rotate_volume("scan.nii.gz", "out/", k=1, axes=(0, 1))
# Output: out/scan_rot1.nii.gz
```

### Rotate 180° in a different plane

```python
rotate_volume("scan.nii.gz", "out/", k=2, axes=(1, 2))
# Output: out/scan_rot2.nii.gz
```

## Typical Workflow

```python
from nidataset.spatial import rotate_volume

# Standardize acquisition orientation before slicing
out = rotate_volume(
    nii_path="data/raw/scan.nii.gz",
    output_path="data/oriented/",
    k=1,
    axes=(0, 1),
    debug=True,
)
print(f"Rotated volume: {out}")
```

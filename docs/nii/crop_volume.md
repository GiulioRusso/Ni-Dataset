---
title: crop_volume
parent: Quick Operations (nii)
nav_order: 5
---

# `crop_volume`

Crop a NIfTI volume to an explicit voxel box, shifting the affine so kept voxels retain their world coordinates.

```python
crop_volume(
    nii_path: str,
    output_path: str,
    bbox: tuple,
    debug: bool = False
) -> str
```

## Overview

Crops the volume to a half-open voxel box `bbox = (x0, x1, y0, y1, z0, z1)`, using
the same semantics as numpy slicing (`data[x0:x1, y0:y1, z0:z1]`). The affine
origin is translated so the retained region stays in the same physical position.

The output is saved as `<PREFIX>_cropped.nii.gz`.

## Parameters

| Name          | Type    | Default    | Description                                                            |
|---------------|---------|------------|------------------------------------------------------------------------|
| `nii_path`    | `str`   | *required* | Path to the input `.nii.gz` file.                                     |
| `output_path` | `str`   | *required* | Directory where the cropped volume is saved.                          |
| `bbox`        | `tuple` | *required* | Half-open voxel bounds `(x0, x1, y0, y1, z0, z1)`.                     |
| `debug`       | `bool`  | `False`    | If `True`, logs the output path and box.                              |

## Returns

`str` – Path to the saved cropped file.

## Output File

```
<PREFIX>_cropped.nii.gz
```

## Exceptions

| Exception           | Condition                                                    |
|---------------------|--------------------------------------------------------------|
| `FileNotFoundError` | Input file does not exist                                   |
| `ValueError`        | Input file is not a valid `.nii.gz` / `.nii`                |
| `ValueError`        | A bound is out of range (must satisfy `0 <= lo < hi <= size`) |

## Usage Notes

- **Half-open bounds**: `x0` is inclusive, `x1` is exclusive — like numpy slicing.
- **Affine-correct**: world coordinates of every retained voxel are preserved.
- For automatic border trimming, use [`crop_to_content`](crop_to_content.md) instead.

## Examples

### Crop to an explicit box

```python
from nidataset.spatial import crop_volume

crop_volume("scan.nii.gz", "out/", bbox=(10, 100, 10, 100, 5, 60))
# Output: out/scan_cropped.nii.gz  (shape 90 x 90 x 55)
```

## Typical Workflow

```python
from nidataset.spatial import crop_volume

# Remove a known scanner border before downstream processing
out = crop_volume(
    nii_path="data/scan.nii.gz",
    output_path="data/cropped/",
    bbox=(20, 236, 20, 236, 0, 128),
    debug=True,
)
print(f"Cropped volume: {out}")
```

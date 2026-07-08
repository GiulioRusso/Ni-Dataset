---
title: crop_to_content
parent: Quick Operations (nii)
nav_order: 6
---

# `crop_to_content`

Crop a NIfTI volume to the minimal box enclosing its foreground (the minimum enclosing rectangle), trimming empty borders.

```python
crop_to_content(
    nii_path: str,
    output_path: str,
    threshold: Optional[float] = None,
    margin: int = 0,
    debug: bool = False
) -> str
```

## Overview

Finds the smallest voxel box that contains all foreground voxels and crops to it,
removing blank borders. Foreground is `data != 0` when `threshold` is `None`, or
`data > threshold` otherwise. An optional `margin` keeps padding voxels around the
box. The affine is shifted so the retained region keeps its world coordinates.

The output is saved as `<PREFIX>_content.nii.gz`.

## Parameters

| Name          | Type    | Default    | Description                                                        |
|---------------|---------|------------|--------------------------------------------------------------------|
| `nii_path`    | `str`   | *required* | Path to the input `.nii.gz` file.                                 |
| `output_path` | `str`   | *required* | Directory where the cropped volume is saved.                      |
| `threshold`   | `float` | `None`     | Foreground cutoff. `None` keeps all non-zero voxels.              |
| `margin`      | `int`   | `0`        | Voxels of padding kept around the box (clamped to bounds).        |
| `debug`       | `bool`  | `False`    | If `True`, logs the output path and box.                          |

## Returns

`str` – Path to the saved cropped file.

## Output File

```
<PREFIX>_content.nii.gz
```

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Input file does not exist                              |
| `ValueError`        | Input file is not a valid `.nii.gz` / `.nii`           |
| `ValueError`        | No voxel passes the foreground test (empty volume)     |

## Usage Notes

- **Threshold choice**: for CT where background is a large negative value, pass an
  explicit `threshold` (e.g. `-500`) rather than relying on the non-zero default.
- **Margin**: useful to avoid clipping objects that touch the foreground box edge.
- **Affine-correct**: world coordinates of every retained voxel are preserved.

## Examples

### Trim empty borders

```python
from nidataset.spatial import crop_to_content

crop_to_content("scan.nii.gz", "out/")
# Output: out/scan_content.nii.gz
```

### Keep a 2-voxel margin

```python
crop_to_content("scan.nii.gz", "out/", threshold=0, margin=2)
```

## Typical Workflow

```python
from nidataset.spatial import crop_to_content

# Shrink volumes to their content before building an ML dataset
out = crop_to_content(
    nii_path="data/scan.nii.gz",
    output_path="data/content/",
    margin=4,
    debug=True,
)
print(f"Content-cropped volume: {out}")
```

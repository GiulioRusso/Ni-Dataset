---
title: rotate_volume_dataset
parent: Quick Operations (nii)
nav_order: 2
---

# `rotate_volume_dataset`

Rotate every NIfTI file in a folder by 90-degree steps, losslessly.

```python
rotate_volume_dataset(
    nii_folder: str,
    output_path: str,
    k: int = 1,
    axes: tuple = (0, 1),
    debug: bool = False
) -> List[str]
```

## Overview

Batch version of [`rotate_volume`](rotate_volume.md): applies the same rotation to
every `.nii.gz` file in `nii_folder`. Each output keeps a correct affine. Files
that fail to process are skipped with a warning.

## Parameters

| Name          | Type    | Default    | Description                                                        |
|---------------|---------|------------|--------------------------------------------------------------------|
| `nii_folder`  | `str`   | *required* | Folder containing `.nii.gz` files.                                |
| `output_path` | `str`   | *required* | Output directory for rotated files.                               |
| `k`           | `int`   | `1`        | Number of 90-degree rotations (may be negative).                  |
| `axes`        | `tuple` | `(0, 1)`   | The two distinct voxel axes defining the rotation plane.          |
| `debug`       | `bool`  | `False`    | If `True`, logs details for each file.                            |

## Returns

`List[str]` – List of output file paths (one per successfully processed file).

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Error Handling**: Files that fail to process are skipped with a warning.
- **Progress Display**: Shows a tqdm progress bar during processing.

## Examples

```python
from nidataset.spatial import rotate_volume_dataset

paths = rotate_volume_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/rotated/",
    k=1,
    axes=(0, 1),
)
print(f"Rotated {len(paths)} volumes")
```

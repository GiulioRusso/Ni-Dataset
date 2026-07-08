---
title: flip_volume_dataset
parent: Quick Operations (nii)
nav_order: 4
---

# `flip_volume_dataset`

Mirror every NIfTI file in a folder along a voxel axis.

```python
flip_volume_dataset(
    nii_folder: str,
    output_path: str,
    axis: int = 0,
    debug: bool = False
) -> List[str]
```

## Overview

Batch version of [`flip_volume`](flip_volume.md): applies the same flip to every
`.nii.gz` file in `nii_folder`, each with a correct affine. Files that fail to
process are skipped with a warning.

## Parameters

| Name          | Type   | Default    | Description                                 |
|---------------|--------|------------|---------------------------------------------|
| `nii_folder`  | `str`  | *required* | Folder containing `.nii.gz` files.         |
| `output_path` | `str`  | *required* | Output directory for flipped files.        |
| `axis`        | `int`  | `0`        | Voxel axis to flip (`0`, `1`, or `2`).     |
| `debug`       | `bool` | `False`    | If `True`, logs details for each file.     |

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
from nidataset.spatial import flip_volume_dataset

paths = flip_volume_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/flipped/",
    axis=0,
)
print(f"Flipped {len(paths)} volumes")
```

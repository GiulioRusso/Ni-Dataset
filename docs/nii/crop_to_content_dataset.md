---
title: crop_to_content_dataset
parent: Quick Operations (nii)
nav_order: 7
---

# `crop_to_content_dataset`

Crop every NIfTI file in a folder to its minimal foreground box.

```python
crop_to_content_dataset(
    nii_folder: str,
    output_path: str,
    threshold: Optional[float] = None,
    margin: int = 0,
    debug: bool = False
) -> List[str]
```

## Overview

Batch version of [`crop_to_content`](crop_to_content.md): trims blank borders from
every `.nii.gz` file in `nii_folder`, each with a correct affine. Files that fail
to process (including empty volumes) are skipped with a warning.

## Parameters

| Name          | Type    | Default    | Description                                                   |
|---------------|---------|------------|---------------------------------------------------------------|
| `nii_folder`  | `str`   | *required* | Folder containing `.nii.gz` files.                           |
| `output_path` | `str`   | *required* | Output directory for cropped files.                          |
| `threshold`   | `float` | `None`     | Foreground cutoff. `None` keeps all non-zero voxels.         |
| `margin`      | `int`   | `0`        | Voxels of padding kept around each box.                      |
| `debug`       | `bool`  | `False`    | If `True`, logs details for each file.                       |

## Returns

`List[str]` – List of output file paths (one per successfully processed file).

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Per-file boxes**: each volume is cropped to its own foreground extent, so
  output shapes may differ between files.
- **Error Handling**: files that fail (e.g. empty volumes) are skipped with a warning.
- **Progress Display**: shows a tqdm progress bar during processing.

## Examples

```python
from nidataset.spatial import crop_to_content_dataset

paths = crop_to_content_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/content/",
    margin=2,
)
print(f"Cropped {len(paths)} volumes")
```

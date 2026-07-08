---
title: rescale_intensity_dataset
parent: Quick Operations (nii)
nav_order: 9
---

# `rescale_intensity_dataset`

Linearly rescale the intensities of every NIfTI file in a folder.

```python
rescale_intensity_dataset(
    nii_folder: str,
    output_path: str,
    out_min: float = 0.0,
    out_max: float = 1.0,
    in_min: Optional[float] = None,
    in_max: Optional[float] = None,
    debug: bool = False
) -> List[str]
```

## Overview

Batch version of [`rescale_intensity`](rescale_intensity.md): applies the same
linear rescaling to every `.nii.gz` file in `nii_folder`. Files that fail to
process are skipped with a warning.

## Parameters

| Name          | Type    | Default    | Description                                             |
|---------------|---------|------------|---------------------------------------------------------|
| `nii_folder`  | `str`   | *required* | Folder containing `.nii.gz` files.                     |
| `output_path` | `str`   | *required* | Output directory for rescaled files.                   |
| `out_min`     | `float` | `0.0`      | Lower bound of the output range.                       |
| `out_max`     | `float` | `1.0`      | Upper bound of the output range.                       |
| `in_min`      | `float` | `None`     | Lower bound of the input range (default: per-file min). |
| `in_max`      | `float` | `None`     | Upper bound of the input range (default: per-file max). |
| `debug`       | `bool`  | `False`    | If `True`, logs details for each file.                 |

## Returns

`List[str]` – List of output file paths (one per successfully processed file).

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Per-file input range**: with `in_min`/`in_max` left as `None`, each file is
  scaled by its own min/max. Set them explicitly for a shared, consistent window.
- **Error Handling**: files that fail to process are skipped with a warning.
- **Progress Display**: shows a tqdm progress bar during processing.

## Examples

```python
from nidataset.transforms import rescale_intensity_dataset

paths = rescale_intensity_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/rescaled/",
    out_min=0,
    out_max=1,
    in_min=-1000,
    in_max=1000,
)
print(f"Rescaled {len(paths)} volumes")
```

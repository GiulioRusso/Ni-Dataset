---
title: windowing_dataset
parent: Package Functions
nav_order: 46
---

# `windowing_dataset`

Apply CT windowing to all NIfTI files in a folder.

```python
windowing_dataset(
    nii_folder: str,
    output_path: str,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    preset: Optional[str] = None,
    normalize: bool = True,
    debug: bool = False
) -> List[str]
```

## Overview

This function batch-processes all NIfTI files in a directory by applying the same CT windowing parameters to each using `windowing`. It is designed for standardizing tissue visualization across an entire dataset.

## Parameters

| Name            | Type    | Default    | Description                                                                          |
|-----------------|---------|------------|--------------------------------------------------------------------------------------|
| `nii_folder`    | `str`   | *required* | Folder containing `.nii.gz` files.                                                  |
| `output_path`   | `str`   | *required* | Output directory for windowed files.                                                |
| `window_center` | `float` | `None`     | Center of the window in Hounsfield units.                                           |
| `window_width`  | `float` | `None`     | Width of the window.                                                                |
| `preset`        | `str`   | `None`     | Named window preset (see `windowing`).                                              |
| `normalize`     | `bool`  | `True`     | If `True`, scales windowed values to [0, 1].                                        |
| `debug`         | `bool`  | `False`    | If `True`, logs details for each file.                                              |

## Returns

`List[str]` – List of output file paths.

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Error Handling**: Files that fail to process are skipped with a warning
- **Progress Display**: Shows a tqdm progress bar during processing

## Examples

### Apply Brain Window to Dataset
Process all CT scans with brain windowing:

```python
from nidataset.transforms import windowing_dataset

paths = windowing_dataset(
    nii_folder="dataset/ct_scans/",
    output_path="dataset/brain_window/",
    preset="brain"
)
print(f"Processed {len(paths)} files")
```

### Custom Window for Dataset

```python
paths = windowing_dataset(
    nii_folder="dataset/ct_scans/",
    output_path="dataset/custom_window/",
    window_center=40,
    window_width=80,
    normalize=True
)
```

## Typical Workflow

```python
from nidataset.transforms import windowing_dataset

# 1. Apply windowing to all CT scans
paths = windowing_dataset(
    nii_folder="data/raw_cts/",
    output_path="data/windowed/",
    preset="brain",
    normalize=True
)

# 2. Verify
print(f"Windowed {len(paths)} volumes")
```

---
title: resample_to_reference_dataset
parent: Package Functions
nav_order: 38
---

# `resample_to_reference_dataset`

Resample all NIfTI files in a folder to match the spatial grid of a reference volume.

```python
resample_to_reference_dataset(
    nii_folder: str,
    reference_path: str,
    output_path: str,
    interpolation: str = "linear",
    debug: bool = False
) -> List[str]
```

## Overview

This function batch-processes all NIfTI files in a directory by resampling each to match the spatial properties (origin, spacing, direction, size) of a single reference volume using `resample_to_reference`.

## Parameters

| Name             | Type   | Default    | Description                                                                          |
|------------------|--------|------------|--------------------------------------------------------------------------------------|
| `nii_folder`     | `str`  | *required* | Folder containing `.nii.gz` files.                                                  |
| `reference_path` | `str`  | *required* | Path to the reference `.nii.gz` file defining the target space.                     |
| `output_path`    | `str`  | *required* | Output directory for resampled files.                                               |
| `interpolation`  | `str`  | `"linear"` | Interpolation method: `"linear"`, `"nearest"`, or `"bspline"`.                      |
| `debug`          | `bool` | `False`    | If `True`, logs details for each file.                                              |

## Returns

`List[str]` – List of output file paths.

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Error Handling**: Files that fail to process are skipped with a warning
- **Progress Display**: Shows a tqdm progress bar during processing
- **Same Reference**: All files are resampled to the same reference grid

## Examples

### Resample Dataset to Template
Bring all scans into template space:

```python
from nidataset.transforms import resample_to_reference_dataset

paths = resample_to_reference_dataset(
    nii_folder="dataset/scans/",
    reference_path="templates/mni_template.nii.gz",
    output_path="dataset/template_space/"
)
print(f"Resampled {len(paths)} files")
```

### Resample Masks with Nearest Neighbor

```python
paths = resample_to_reference_dataset(
    nii_folder="dataset/masks/",
    reference_path="templates/mni_template.nii.gz",
    output_path="dataset/masks_template_space/",
    interpolation="nearest"
)
```

## Typical Workflow

```python
from nidataset.transforms import resample_to_reference_dataset

# 1. Resample all scans to a common reference
paths = resample_to_reference_dataset(
    nii_folder="data/multi_site/",
    reference_path="data/reference.nii.gz",
    output_path="data/unified/",
    interpolation="linear"
)

# 2. All volumes now have identical dimensions and spacing
print(f"Unified {len(paths)} volumes to common space")
```

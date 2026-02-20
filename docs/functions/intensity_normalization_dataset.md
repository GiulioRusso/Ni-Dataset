---
title: intensity_normalization_dataset
parent: Package Functions
nav_order: 24
---

# `intensity_normalization_dataset`

Apply intensity normalization to all NIfTI files in a folder.

```python
intensity_normalization_dataset(
    nii_folder: str,
    output_path: str,
    method: str = "zscore",
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    reference_path: Optional[str] = None,
    debug: bool = False
) -> List[str]
```

## Overview

This function batch-processes all NIfTI files in a directory by applying the same intensity normalization method to each using `intensity_normalization`. It is designed for standardizing an entire dataset in a single call.

## Parameters

| Name               | Type             | Default        | Description                                                                    |
|--------------------|------------------|----------------|--------------------------------------------------------------------------------|
| `nii_folder`       | `str`            | *required*     | Folder containing `.nii.gz` files.                                            |
| `output_path`      | `str`            | *required*     | Output directory for normalized files.                                        |
| `method`           | `str`            | `"zscore"`     | Normalization method (see `intensity_normalization`).                          |
| `percentile_range` | `Tuple[float, float]` | `(1.0, 99.0)` | Percentile range for `"percentile"` method.                              |
| `reference_path`   | `str`            | `None`         | Reference NIfTI path for `"histogram"` matching.                              |
| `debug`            | `bool`           | `False`        | If `True`, logs details for each file.                                        |

## Returns

`List[str]` – List of output file paths.

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Error Handling**: Files that fail to process are skipped with a warning
- **Progress Display**: Shows a tqdm progress bar during processing
- **Same Reference**: When using `"histogram"` matching, all files are matched to the same reference

## Examples

### Normalize Entire Dataset
Apply min-max normalization to all scans:

```python
from nidataset.transforms import intensity_normalization_dataset

paths = intensity_normalization_dataset(
    nii_folder="dataset/raw/",
    output_path="dataset/normalized/",
    method="minmax"
)
print(f"Normalized {len(paths)} files")
```

### Histogram Harmonization
Match all scans to a reference:

```python
paths = intensity_normalization_dataset(
    nii_folder="multi_site_data/",
    output_path="harmonized/",
    method="histogram",
    reference_path="templates/reference_scan.nii.gz"
)
```

## Typical Workflow

```python
from nidataset.transforms import intensity_normalization_dataset

# 1. Normalize the entire dataset
paths = intensity_normalization_dataset(
    nii_folder="data/scans/",
    output_path="data/normalized/",
    method="percentile",
    percentile_range=(1.0, 99.0)
)

# 2. Verify output
print(f"Processed {len(paths)} volumes")
for p in paths[:3]:
    print(f"  {p}")
```

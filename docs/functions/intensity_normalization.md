---
title: intensity_normalization
parent: Package Functions
nav_order: 23
---

# `intensity_normalization`

Normalize the intensity values of a 3D NIfTI volume using z-score, min-max, percentile clipping, or histogram matching.

```python
intensity_normalization(
    nii_path: str,
    output_path: str,
    method: str = "zscore",
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    reference_path: Optional[str] = None,
    debug: bool = False
) -> str
```

## Overview

This function standardizes intensity values in a NIfTI volume, which is a critical preprocessing step for machine learning and multi-site analysis. Different normalization methods address different use cases:

- **Z-score**: Centers data around zero with unit variance. Best for general-purpose normalization.
- **Min-max**: Scales values to [0, 1]. Good for consistent input range to neural networks.
- **Percentile**: Clips to a percentile range then applies min-max. Robust to outliers.
- **Histogram**: Matches the intensity histogram to a reference volume. Best for multi-site harmonization.

The output is saved as `<PREFIX>_normalized.nii.gz`.

## Normalization Methods

| Method        | Formula                                        | Output Range | Best For                       |
|---------------|------------------------------------------------|--------------|--------------------------------|
| `zscore`      | `(x - mean) / std`                             | unbounded    | General normalization          |
| `minmax`      | `(x - min) / (max - min)`                      | [0, 1]       | Neural network inputs          |
| `percentile`  | Clip to percentile range, then min-max          | [0, 1]       | Robust to intensity outliers   |
| `histogram`   | Match histogram to reference via quantile mapping | varies     | Multi-site harmonization       |

## Parameters

| Name               | Type             | Default      | Description                                                                      |
|--------------------|------------------|--------------|----------------------------------------------------------------------------------|
| `nii_path`         | `str`            | *required*   | Path to the input `.nii.gz` file.                                               |
| `output_path`      | `str`            | *required*   | Directory where the normalized volume will be saved.                            |
| `method`           | `str`            | `"zscore"`   | Normalization method: `"zscore"`, `"minmax"`, `"percentile"`, or `"histogram"`. |
| `percentile_range` | `Tuple[float, float]` | `(1.0, 99.0)` | Low and high percentile values for the `"percentile"` method.               |
| `reference_path`   | `str`            | `None`       | Reference NIfTI path required for `"histogram"` matching.                       |
| `debug`            | `bool`           | `False`      | If `True`, logs the output path and method used.                                |

## Returns

`str` – Path to the saved normalized file.

## Output File

The normalized volume is saved as:
```
<PREFIX>_normalized.nii.gz
```

**Example**: Input `scan_001.nii.gz` → Output `scan_001_normalized.nii.gz`

## Exceptions

| Exception    | Condition                                                        |
|--------------|------------------------------------------------------------------|
| `ValueError` | Unknown normalization method                                     |
| `ValueError` | `reference_path` not provided for `"histogram"` method           |
| `FileNotFoundError` | Input or reference file does not exist                  |

## Usage Notes

- **Affine Preservation**: The spatial transformation matrix and header are preserved
- **Data Type**: Output is saved as `float32` regardless of input type
- **Zero Std**: If the volume has zero standard deviation, z-score normalization is skipped with a warning
- **Percentile Method**: First clips to the specified percentile range, then applies min-max normalization

## Examples

### Z-Score Normalization
Standard zero-mean, unit-variance normalization:

```python
from nidataset.transforms import intensity_normalization

intensity_normalization(
    nii_path="scans/patient_001.nii.gz",
    output_path="normalized/",
    method="zscore"
)
```

### Min-Max Normalization
Scale to [0, 1] for neural network input:

```python
intensity_normalization(
    nii_path="scans/patient_001.nii.gz",
    output_path="normalized/",
    method="minmax"
)
```

### Robust Percentile Normalization
Clip outliers before scaling:

```python
intensity_normalization(
    nii_path="scans/patient_001.nii.gz",
    output_path="normalized/",
    method="percentile",
    percentile_range=(2.0, 98.0)
)
```

### Histogram Matching
Harmonize a scan to match a reference volume:

```python
intensity_normalization(
    nii_path="site_b/scan.nii.gz",
    output_path="harmonized/",
    method="histogram",
    reference_path="site_a/template_scan.nii.gz"
)
```

## Typical Workflow

```python
from nidataset.transforms import intensity_normalization

# 1. Choose normalization method based on your use case
method = "percentile"  # Robust to outliers

# 2. Normalize the volume
out_path = intensity_normalization(
    nii_path="data/raw_scan.nii.gz",
    output_path="data/preprocessed/",
    method=method,
    percentile_range=(1.0, 99.0),
    debug=True
)

# 3. Use the normalized volume for training or analysis
print(f"Normalized volume: {out_path}")
```

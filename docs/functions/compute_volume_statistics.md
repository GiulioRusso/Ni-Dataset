---
title: compute_volume_statistics
parent: Package Functions
nav_order: 35
---

# `compute_volume_statistics`

Compute detailed intensity statistics for a single NIfTI volume, including distribution metrics, percentiles, and signal-to-noise ratio.

```python
compute_volume_statistics(
    nii_path: str,
    mask_path: Optional[str] = None,
    debug: bool = False
) -> Dict[str, float]
```

## Overview

This function extracts comprehensive intensity statistics from a NIfTI volume. Statistics can optionally be restricted to a region of interest defined by a mask. The computed metrics include central tendency, spread, percentiles, distribution shape, and signal quality measures.

This is useful for:
- Quality control and outlier detection
- Understanding intensity distributions before normalization
- Comparing acquisitions from different scanners
- Monitoring preprocessing effects

## Computed Statistics

| Statistic       | Description                                                   |
|-----------------|---------------------------------------------------------------|
| `mean`          | Mean intensity value                                          |
| `std`           | Standard deviation of intensities                             |
| `min`           | Minimum intensity value                                       |
| `max`           | Maximum intensity value                                       |
| `median`        | Median intensity value                                        |
| `percentile_1`  | 1st percentile                                                |
| `percentile_5`  | 5th percentile                                                |
| `percentile_25` | 25th percentile (Q1)                                          |
| `percentile_75` | 75th percentile (Q3)                                          |
| `percentile_95` | 95th percentile                                               |
| `percentile_99` | 99th percentile                                               |
| `nonzero_count` | Number of non-zero voxels                                     |
| `total_voxels`  | Total number of voxels analyzed                               |
| `skewness`      | Distribution skewness (0 = symmetric)                         |
| `kurtosis`      | Excess kurtosis (0 = normal distribution)                     |
| `snr`           | Signal-to-noise ratio (mean / std of non-zero voxels)         |

## Parameters

| Name        | Type   | Default    | Description                                                                                  |
|-------------|--------|------------|----------------------------------------------------------------------------------------------|
| `nii_path`  | `str`  | *required* | Path to the input `.nii.gz` file.                                                           |
| `mask_path` | `str`  | `None`     | Optional mask NIfTI file. If provided, statistics are computed only within the masked region.|
| `debug`     | `bool` | `False`    | If `True`, logs all computed statistics.                                                     |

## Returns

`Dict[str, float]` – Dictionary mapping statistic names to their computed values.

## Exceptions

| Exception           | Condition                              |
|---------------------|----------------------------------------|
| `FileNotFoundError` | Input file does not exist              |
| `ValueError`        | File is not a valid `.nii.gz` format   |

## Usage Notes

- **Mask Support**: When a mask is provided, only voxels where the mask is non-zero are included
- **SNR Calculation**: SNR is computed as mean/std of non-zero voxels; returns 0.0 if std is zero
- **Skewness/Kurtosis**: Returns 0.0 when standard deviation is zero

## Examples

### Basic Usage
Get statistics for a single volume:

```python
from nidataset.analysis import compute_volume_statistics

stats = compute_volume_statistics("scan.nii.gz")
print(f"Mean: {stats['mean']:.2f}")
print(f"Std:  {stats['std']:.2f}")
print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
print(f"SNR:  {stats['snr']:.2f}")
```

### With Brain Mask
Restrict statistics to brain tissue:

```python
stats = compute_volume_statistics(
    nii_path="ct_scan.nii.gz",
    mask_path="brain_mask.nii.gz",
    debug=True
)
print(f"Brain mean intensity: {stats['mean']:.2f}")
print(f"Brain voxels: {stats['nonzero_count']}")
```

### Distribution Analysis
Examine the intensity distribution:

```python
stats = compute_volume_statistics("scan.nii.gz")

print(f"Percentile range (1-99): [{stats['percentile_1']:.1f}, {stats['percentile_99']:.1f}]")
print(f"IQR: [{stats['percentile_25']:.1f}, {stats['percentile_75']:.1f}]")
print(f"Skewness: {stats['skewness']:.3f}")
print(f"Kurtosis: {stats['kurtosis']:.3f}")

if stats['skewness'] > 1:
    print("Distribution is right-skewed")
elif stats['skewness'] < -1:
    print("Distribution is left-skewed")
else:
    print("Distribution is approximately symmetric")
```

## Typical Workflow

```python
from nidataset.analysis import compute_volume_statistics

# 1. Compute statistics
stats = compute_volume_statistics("data/patient_scan.nii.gz")

# 2. Check data quality
print(f"Intensity range: [{stats['min']:.1f}, {stats['max']:.1f}]")
print(f"SNR: {stats['snr']:.2f}")
print(f"Non-zero fraction: {stats['nonzero_count'] / stats['total_voxels']:.2%}")

# 3. Determine normalization parameters
print(f"Suggested percentile clip: [{stats['percentile_1']:.1f}, {stats['percentile_99']:.1f}]")
```

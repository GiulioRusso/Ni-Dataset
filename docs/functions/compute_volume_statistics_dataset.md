---
title: compute_volume_statistics_dataset
parent: Package Functions
nav_order: 36
---

# `compute_volume_statistics_dataset`

Compute intensity statistics for all NIfTI files in a folder and save results as a CSV.

```python
compute_volume_statistics_dataset(
    nii_folder: str,
    output_path: str,
    mask_folder: Optional[str] = None,
    debug: bool = False
) -> str
```

## Overview

This function iterates over all NIfTI files in a directory and computes detailed intensity statistics for each using `compute_volume_statistics`. Results are saved in a CSV file `volume_statistics.csv` with one row per file and columns for each statistic.

This is useful for:
- Dataset-wide quality control
- Identifying scans with unusual intensity distributions
- Planning normalization strategies across a dataset
- Documenting dataset properties for publications

## Parameters

| Name          | Type   | Default    | Description                                                                                  |
|---------------|--------|------------|----------------------------------------------------------------------------------------------|
| `nii_folder`  | `str`  | *required* | Folder containing `.nii.gz` files.                                                          |
| `output_path` | `str`  | *required* | Directory where the output CSV will be saved.                                               |
| `mask_folder` | `str`  | `None`     | Optional folder with matching mask files. Masks are matched by filename.                    |
| `debug`       | `bool` | `False`    | If `True`, logs detailed statistics for each file.                                          |

## Returns

`str` – Path to the saved `volume_statistics.csv` file.

## Output File

### CSV Structure
The function creates `volume_statistics.csv` with columns:

| Column           | Description                                 |
|------------------|---------------------------------------------|
| `FILENAME`       | Name of the NIfTI file                      |
| `MEAN`           | Mean intensity                              |
| `STD`            | Standard deviation                          |
| `MIN`            | Minimum intensity                           |
| `MAX`            | Maximum intensity                           |
| `MEDIAN`         | Median intensity                            |
| `PERCENTILE_*`   | Percentile values (1, 5, 25, 75, 95, 99)   |
| `NONZERO_COUNT`  | Number of non-zero voxels                   |
| `TOTAL_VOXELS`   | Total voxels analyzed                       |
| `SKEWNESS`       | Distribution skewness                       |
| `KURTOSIS`       | Excess kurtosis                             |
| `SNR`            | Signal-to-noise ratio                       |

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Mask Matching**: Masks are matched by identical filename in the mask folder
- **Error Handling**: Files that fail to process are skipped with a warning
- **Progress Display**: Shows a tqdm progress bar during processing

## Examples

### Basic Usage
Compute statistics for all scans:

```python
from nidataset.analysis import compute_volume_statistics_dataset

csv_path = compute_volume_statistics_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/stats/"
)
print(f"Statistics saved to: {csv_path}")
```

### With Masks
Restrict statistics to masked regions:

```python
csv_path = compute_volume_statistics_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/stats/",
    mask_folder="dataset/brain_masks/"
)
```

### Analyze Results
Load and analyze the statistics:

```python
import pandas as pd
from nidataset.analysis import compute_volume_statistics_dataset

compute_volume_statistics_dataset("scans/", "output/")

df = pd.read_csv("output/volume_statistics.csv")
print(f"Mean SNR across dataset: {df['SNR'].mean():.2f}")
print(f"Intensity range: [{df['MIN'].min():.1f}, {df['MAX'].max():.1f}]")

# Find outliers
low_snr = df[df['SNR'] < df['SNR'].quantile(0.05)]
print(f"\nLow-SNR scans: {len(low_snr)}")
```

## Typical Workflow

```python
from nidataset.analysis import compute_volume_statistics_dataset
import pandas as pd

# 1. Compute statistics for the entire dataset
csv_path = compute_volume_statistics_dataset(
    nii_folder="data/raw_scans/",
    output_path="data/analysis/"
)

# 2. Load results
df = pd.read_csv(csv_path)

# 3. Dataset summary
print(f"Total scans: {len(df)}")
print(f"Mean intensity: {df['MEAN'].mean():.2f} +/- {df['MEAN'].std():.2f}")
print(f"Mean SNR: {df['SNR'].mean():.2f}")

# 4. Quality filtering
good_scans = df[df['SNR'] > 5.0]['FILENAME'].tolist()
print(f"\nScans passing SNR threshold: {len(good_scans)}/{len(df)}")
```

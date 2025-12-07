---
title: crop_and_pad_dataset
parent: Package Functions
nav_order: 23
---

# `crop_and_pad_dataset`

Batch process all NIfTI scans in a folder by cropping to non-zero regions and padding/cropping to a uniform target shape.

```python
crop_and_pad_dataset(
    nii_folder: str,
    output_path: str,
    target_shape: tuple = (128, 128, 128),
    save_stats: bool = False
) -> None
```

## Overview

This function processes all `.nii.gz` files in a directory by applying the `crop_and_pad` operation to each volume. Each scan is:
1. Cropped to its non-zero bounding box (removes empty space)
2. Padded or cropped to match the specified `target_shape`
3. Saved with the suffix `_cropped_padded.nii.gz`

The function is designed for batch preprocessing of medical imaging datasets, particularly CTA scans, ensuring all volumes have consistent dimensions for downstream analysis or machine learning tasks.

## Parameters

| Name           | Type    | Default           | Description                                                                                          |
|----------------|---------|-------------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`   | `str`   | *required*        | Path to the directory containing CTA volumes in `.nii.gz` format.                                   |
| `output_path`  | `str`   | *required*        | Directory where processed scans will be saved. Created automatically if it doesn't exist.           |
| `target_shape` | `tuple` | `(128, 128, 128)` | Desired output shape `(X, Y, Z)` applied uniformly to all scans.                                    |
| `save_stats`   | `bool`  | `False`           | If `True`, saves a `crop_pad_stats.csv` file with original and final shapes for each processed scan.|

## Returns

`None` – The function processes files in-place and saves outputs to disk.

## Output Files

### Processed Scans
Each input file `<PREFIX>.nii.gz` produces an output file:
```
<PREFIX>_cropped_padded.nii.gz
```

### Statistics File (Optional)
When `save_stats=True`, a CSV file `crop_pad_stats.csv` is created in `output_path` with columns:
- **FILENAME**: Original input filename
- **ORIGINAL_SHAPE**: Shape before processing `(X, Y, Z)`
- **FINAL_SHAPE**: Shape after processing (should match `target_shape`)

## Exceptions

| Exception            | Condition                                                    |
|----------------------|--------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed; other formats are ignored
- **3D Volumes Required**: Input scans must be 3D NIfTI images
- **Progress Display**: Processing progress is shown with a tqdm progress bar
- **Error Handling**: Files that fail to process are skipped with error messages, allowing the batch to continue
- **Output Directory**: Automatically created if it doesn't exist

## Examples

### Basic Usage
Process all scans in a folder to a standard 128×128×128 shape:

```python
from nidataset.volume import crop_and_pad_dataset

crop_and_pad_dataset(
    nii_folder="dataset/CTA_raw/",
    output_path="output/CTA_cropped_padded/",
    target_shape=(128, 128, 128)
)
```

### With Statistics Tracking
Enable statistics logging to track shape changes:

```python
crop_and_pad_dataset(
    nii_folder="dataset/CTA_raw/",
    output_path="output/CTA_cropped_padded/",
    target_shape=(128, 128, 128),
    save_stats=True
)
# Creates: output/CTA_cropped_padded/crop_pad_stats.csv
```

### Custom Target Shape
Use a different target shape for your specific needs:

```python
crop_and_pad_dataset(
    nii_folder="data/scans/",
    output_path="data/preprocessed/",
    target_shape=(256, 256, 256),
    save_stats=True
)
```

## Typical Workflow

```python
from nidataset.volume import crop_and_pad_dataset

# 1. Prepare your dataset folder
dataset_folder = "raw_data/cta_scans/"

# 2. Define output location
output_folder = "preprocessed_data/uniform_scans/"

# 3. Run batch processing with statistics
crop_and_pad_dataset(
    nii_folder=dataset_folder,
    output_path=output_folder,
    target_shape=(128, 128, 128),
    save_stats=True
)

# 4. Review statistics
import pandas as pd
stats = pd.read_csv("preprocessed_data/uniform_scans/crop_pad_stats.csv")
print(stats.head())
```

---
title: compare_volumes_dataset
parent: Package Functions
nav_order: 34
---

# `compare_volumes_dataset`

Compute comparison metrics between matching NIfTI files in two folders and save results as a CSV.

```python
compare_volumes_dataset(
    nii_folder_a: str,
    nii_folder_b: str,
    output_path: str,
    metrics: Optional[List[str]] = None,
    debug: bool = False
) -> str
```

## Overview

This function iterates over two directories of NIfTI files, matches them by filename, and computes pairwise similarity metrics using `compare_volumes`. Results are saved in a CSV file `volume_comparison.csv` with one row per matched pair.

This is useful for:
- Batch evaluation of segmentation predictions against ground truth
- Comparing registration results across an entire dataset
- Quantifying differences between two processing pipelines

## Parameters

| Name           | Type        | Default    | Description                                                                               |
|----------------|-------------|------------|-------------------------------------------------------------------------------------------|
| `nii_folder_a` | `str`       | *required* | First folder containing `.nii.gz` files.                                                 |
| `nii_folder_b` | `str`       | *required* | Second folder containing `.nii.gz` files.                                                |
| `output_path`  | `str`       | *required* | Directory where the output CSV will be saved.                                            |
| `metrics`      | `List[str]` | `None`     | List of metric names to compute. If `None`, computes all available metrics.              |
| `debug`        | `bool`      | `False`    | If `True`, logs detailed metric results for each pair.                                   |

## Returns

`str` – Path to the saved `volume_comparison.csv` file.

## Output File

### CSV Structure
The function creates `volume_comparison.csv` with columns:

| Column       | Description                          |
|--------------|--------------------------------------|
| `FILENAME`   | Name of the matched NIfTI file       |
| `DICE`       | Dice similarity coefficient          |
| `HAUSDORFF`  | Hausdorff distance in voxels         |
| `MSE`        | Mean Squared Error                   |
| ...          | One column per requested metric      |

## Exceptions

| Exception           | Condition                                                    |
|---------------------|--------------------------------------------------------------|
| `FileNotFoundError` | No matching filenames found between the two folders          |

## Usage Notes

- **Filename Matching**: Files are matched by exact filename between the two folders
- **Unmatched Files**: Files that exist in only one folder are ignored
- **Error Handling**: Files that fail to process are skipped with a warning
- **Progress Display**: Shows a tqdm progress bar during processing

## Examples

### Segmentation Evaluation
Compare predicted segmentations against ground truth:

```python
from nidataset.analysis import compare_volumes_dataset

csv_path = compare_volumes_dataset(
    nii_folder_a="predictions/",
    nii_folder_b="ground_truth/",
    output_path="evaluation/",
    metrics=["dice", "jaccard", "hausdorff"]
)
print(f"Results saved to: {csv_path}")
```

### Analyze Results
Load and analyze the comparison CSV:

```python
import pandas as pd
from nidataset.analysis import compare_volumes_dataset

compare_volumes_dataset("pred/", "gt/", "eval/", metrics=["dice", "mse"])

df = pd.read_csv("eval/volume_comparison.csv")
print(f"Mean Dice: {df['DICE'].mean():.4f}")
print(f"Std Dice:  {df['DICE'].std():.4f}")
print(f"Worst case: {df.loc[df['DICE'].idxmin(), 'FILENAME']}")
```

## Typical Workflow

```python
from nidataset.analysis import compare_volumes_dataset
import pandas as pd

# 1. Run batch comparison
csv_path = compare_volumes_dataset(
    nii_folder_a="model_output/",
    nii_folder_b="annotations/",
    output_path="results/",
    metrics=["dice", "jaccard", "hausdorff", "volume_diff"]
)

# 2. Load and summarize
df = pd.read_csv(csv_path)
print(df.describe())

# 3. Identify problematic cases
poor_cases = df[df["DICE"] < 0.7]
print(f"\n{len(poor_cases)} cases with Dice < 0.7:")
print(poor_cases[["FILENAME", "DICE"]])
```

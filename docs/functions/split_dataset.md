---
title: split_dataset
parent: Package Functions
nav_order: 37
---

# `split_dataset`

Split a NIfTI dataset folder into train, validation, and test subsets with reproducible randomization.

```python
split_dataset(
    nii_folder: str,
    output_path: str,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    copy_files: bool = False,
    debug: bool = False
) -> Dict[str, List[str]]
```

## Overview

This function randomly assigns NIfTI files from a dataset folder into train, validation, and test subsets. It creates a JSON manifest (`split.json`) listing the files in each subset, and optionally copies files into separate subdirectories.

This is useful for:
- Preparing datasets for machine learning experiments
- Ensuring reproducible train/val/test splits
- Maintaining consistent splits across team members

## Parameters

| Name         | Type                    | Default             | Description                                                                            |
|--------------|-------------------------|---------------------|----------------------------------------------------------------------------------------|
| `nii_folder` | `str`                   | *required*          | Folder containing `.nii.gz` files to split.                                           |
| `output_path`| `str`                   | *required*          | Directory where split artifacts are saved.                                            |
| `ratios`     | `Tuple[float, ...]`     | `(0.7, 0.15, 0.15)` | Train, val, test fractions. Must sum to 1.0.                                          |
| `seed`       | `int`                   | `42`                | Random seed for reproducibility.                                                      |
| `copy_files` | `bool`                  | `False`             | If `True`, copies files into `train/`, `val/`, `test/` subdirectories.                |
| `debug`      | `bool`                  | `False`             | If `True`, logs split counts.                                                         |

## Returns

`Dict[str, List[str]]` – Dictionary with keys `"train"`, `"val"`, `"test"` mapping to lists of filenames.

## Output Files

### JSON Manifest
Always creates `split.json`:
```json
{
  "train": ["case_001.nii.gz", "case_003.nii.gz", ...],
  "val": ["case_007.nii.gz", ...],
  "test": ["case_012.nii.gz", ...]
}
```

### File Copies (Optional)
When `copy_files=True`, creates:
```
output_path/
├── split.json
├── train/
│   ├── case_001.nii.gz
│   └── ...
├── val/
│   ├── case_007.nii.gz
│   └── ...
└── test/
    ├── case_012.nii.gz
    └── ...
```

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |
| `ValueError`        | Ratios do not sum to approximately 1.0                 |
| `ValueError`        | Ratios tuple does not contain exactly 3 values         |

## Usage Notes

- **Reproducibility**: Using the same `seed` always produces the same split
- **Manifest Only**: By default, only the JSON manifest is created (no file copying)
- **Rounding**: Due to integer rounding, the test set may receive slightly more or fewer files

## Examples

### Basic Usage
Create a 70/15/15 split:

```python
from nidataset.analysis import split_dataset

splits = split_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/splits/"
)
print(f"Train: {len(splits['train'])} files")
print(f"Val:   {len(splits['val'])} files")
print(f"Test:  {len(splits['test'])} files")
```

### Copy Files into Subdirectories
Physically organize files by split:

```python
splits = split_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/organized/",
    ratios=(0.8, 0.1, 0.1),
    copy_files=True,
    seed=123
)
```

### Custom Ratios
Use an 80/10/10 split:

```python
splits = split_dataset(
    nii_folder="data/all_scans/",
    output_path="data/",
    ratios=(0.8, 0.1, 0.1),
    seed=42,
    debug=True
)
```

### Load Existing Split
Reuse a previously created split:

```python
import json

with open("dataset/splits/split.json") as f:
    splits = json.load(f)

print(f"Training files: {len(splits['train'])}")
for fname in splits['train'][:5]:
    print(f"  {fname}")
```

## Typical Workflow

```python
from nidataset.analysis import split_dataset
import json

# 1. Create reproducible split
splits = split_dataset(
    nii_folder="data/preprocessed/",
    output_path="data/experiment_01/",
    ratios=(0.7, 0.15, 0.15),
    seed=42,
    copy_files=True,
    debug=True
)

# 2. Verify distribution
total = sum(len(v) for v in splits.values())
for subset, files in splits.items():
    print(f"{subset}: {len(files)} files ({len(files)/total:.0%})")

# 3. Share the manifest with collaborators
# The split.json file ensures everyone uses the same split
```

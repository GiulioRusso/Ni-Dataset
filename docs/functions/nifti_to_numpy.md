---
title: nifti_to_numpy
parent: Package Functions
nav_order: 45
---

# `nifti_to_numpy`

Convert a NIfTI volume to a NumPy `.npy` or compressed `.npz` file.

```python
nifti_to_numpy(
    nii_path: str,
    output_path: str,
    compressed: bool = True,
    debug: bool = False
) -> str
```

## Overview

This function extracts the voxel data from a NIfTI file and saves it as a NumPy array file. This is useful for:
- Loading data faster in Python-based training pipelines
- Reducing storage with compressed `.npz` format
- Interfacing with frameworks that expect NumPy arrays
- Creating lightweight data copies without NIfTI headers

## Parameters

| Name         | Type   | Default    | Description                                                                     |
|--------------|--------|------------|---------------------------------------------------------------------------------|
| `nii_path`   | `str`  | *required* | Path to the input `.nii.gz` file.                                              |
| `output_path`| `str`  | *required* | Directory where the NumPy file will be saved.                                  |
| `compressed` | `bool` | `True`     | If `True`, saves as compressed `.npz`. If `False`, saves as `.npy`.            |
| `debug`      | `bool` | `False`    | If `True`, logs the output path.                                               |

## Returns

`str` – Path to the saved NumPy file.

## Output File

```
<PREFIX>.npz   (when compressed=True)
<PREFIX>.npy   (when compressed=False)
```

**Example**: Input `scan_001.nii.gz` → Output `scan_001.npz`

## Exceptions

| Exception           | Condition                              |
|---------------------|----------------------------------------|
| `FileNotFoundError` | Input file does not exist              |
| `ValueError`        | File is not a valid `.nii.gz` format   |

## Usage Notes

- **Compressed Format**: `.npz` files are smaller but slightly slower to load
- **Data Key**: In `.npz` files, the array is stored under the key `"data"`
- **No Header**: Affine and header information is not preserved; use `numpy_to_nifti` with a reference to restore it
- **Data Type**: The original NIfTI data type is preserved

## Examples

### Basic Usage
Convert to compressed NumPy:

```python
from nidataset.transforms import nifti_to_numpy

nifti_to_numpy("scan.nii.gz", "numpy_data/")
# Output: numpy_data/scan.npz
```

### Uncompressed Format

```python
nifti_to_numpy("scan.nii.gz", "numpy_data/", compressed=False)
# Output: numpy_data/scan.npy
```

### Load the Converted Data

```python
import numpy as np

# From .npz
data = np.load("numpy_data/scan.npz")["data"]

# From .npy
data = np.load("numpy_data/scan.npy")

print(f"Shape: {data.shape}, Dtype: {data.dtype}")
```

### Batch Conversion

```python
import os
from nidataset.transforms import nifti_to_numpy

input_folder = "nifti_scans/"
for fname in os.listdir(input_folder):
    if fname.endswith(".nii.gz"):
        nifti_to_numpy(os.path.join(input_folder, fname), "numpy_scans/")
```

## Typical Workflow

```python
from nidataset.transforms import nifti_to_numpy

# 1. Convert NIfTI to NumPy for faster data loading
out_path = nifti_to_numpy(
    nii_path="data/preprocessed/scan.nii.gz",
    output_path="data/numpy/",
    compressed=True,
    debug=True
)

# 2. Use in training pipeline
import numpy as np
data = np.load(out_path)["data"]
print(f"Ready for training: shape={data.shape}")
```

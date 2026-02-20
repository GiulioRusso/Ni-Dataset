---
title: numpy_to_nifti
parent: Package Functions
nav_order: 46
---

# `numpy_to_nifti`

Convert a NumPy `.npy` or `.npz` file to NIfTI format with optional affine matrix and header from a reference.

```python
numpy_to_nifti(
    npy_path: str,
    output_path: str,
    affine: Optional[np.ndarray] = None,
    reference_nifti: Optional[str] = None,
    debug: bool = False
) -> str
```

## Overview

This function converts a NumPy array file back to NIfTI format. It is the inverse operation of `nifti_to_numpy`. You can provide spatial metadata either via a 4x4 affine matrix or by copying it from a reference NIfTI file.

This is useful for:
- Saving model predictions back to NIfTI format
- Converting processed NumPy arrays to medical imaging format
- Restoring spatial metadata from original files

## Parameters

| Name              | Type          | Default    | Description                                                                           |
|-------------------|---------------|------------|---------------------------------------------------------------------------------------|
| `npy_path`        | `str`         | *required* | Path to the input `.npy` or `.npz` file.                                             |
| `output_path`     | `str`         | *required* | Directory where the NIfTI file will be saved.                                        |
| `affine`          | `np.ndarray`  | `None`     | 4x4 affine transformation matrix. If `None`, uses identity or reference.             |
| `reference_nifti` | `str`         | `None`     | Optional NIfTI file to copy affine and header from. Overrides `affine` parameter.    |
| `debug`           | `bool`        | `False`    | If `True`, logs the output path.                                                     |

## Returns

`str` – Path to the saved NIfTI file.

## Output File

```
<PREFIX>.nii.gz
```

**Example**: Input `scan_001.npz` → Output `scan_001.nii.gz`

## Exceptions

| Exception           | Condition                              |
|---------------------|----------------------------------------|
| `FileNotFoundError` | Input file does not exist              |

## Usage Notes

- **Reference Priority**: If `reference_nifti` is provided, it overrides the `affine` parameter
- **Identity Fallback**: If neither `affine` nor `reference_nifti` is given, an identity matrix is used
- **NPZ Key**: For `.npz` files, the first stored array is used
- **Header Preservation**: When using `reference_nifti`, both affine and header are copied

## Examples

### With Reference NIfTI
Restore spatial metadata from the original file:

```python
from nidataset.transforms import numpy_to_nifti

numpy_to_nifti(
    npy_path="predictions/case001.npz",
    output_path="results/",
    reference_nifti="original/case001.nii.gz"
)
# Output: results/case001.nii.gz (with correct spatial metadata)
```

### With Custom Affine

```python
import numpy as np
from nidataset.transforms import numpy_to_nifti

affine = np.diag([1.0, 1.0, 2.0, 1.0])  # 1mm x 1mm x 2mm spacing
numpy_to_nifti(
    npy_path="data/volume.npy",
    output_path="nifti/",
    affine=affine
)
```

### Identity Affine (Default)

```python
numpy_to_nifti("data/array.npy", "output/")
# Uses identity affine (1mm isotropic spacing)
```

### Round-Trip Conversion

```python
from nidataset.transforms import nifti_to_numpy, numpy_to_nifti

# Convert to NumPy
nifti_to_numpy("original.nii.gz", "temp/")

# ... process the array ...

# Convert back with original metadata
numpy_to_nifti("temp/original.npz", "output/", reference_nifti="original.nii.gz")
```

## Typical Workflow

```python
from nidataset.transforms import numpy_to_nifti

# 1. Save model prediction back to NIfTI
out_path = numpy_to_nifti(
    npy_path="model_output/prediction.npz",
    output_path="results/",
    reference_nifti="data/original_scan.nii.gz",
    debug=True
)

# 2. The output can be viewed in any NIfTI viewer
print(f"Prediction saved: {out_path}")
```

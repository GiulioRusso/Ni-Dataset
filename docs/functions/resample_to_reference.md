---
title: resample_to_reference
parent: Package Functions
nav_order: 42
---

# `resample_to_reference`

Resample a NIfTI volume to match the spatial grid (origin, spacing, direction, size) of a reference volume.

```python
resample_to_reference(
    nii_path: str,
    reference_path: str,
    output_path: str,
    interpolation: str = "linear",
    debug: bool = False
) -> str
```

## Overview

This function resamples a moving NIfTI volume so that it occupies the exact same physical space and voxel grid as a reference volume. The output will have the same origin, spacing, direction, and size as the reference.

This is useful for:
- Bringing volumes from different sources into the same physical space
- Ensuring consistent dimensions before voxel-wise operations
- Aligning multi-modal images (CT, MRI, masks) to a common grid
- Preparing data for atlas-based analysis

The output is saved as `<PREFIX>_resampled_to_ref.nii.gz`.

## Parameters

| Name             | Type   | Default      | Description                                                                          |
|------------------|--------|--------------|--------------------------------------------------------------------------------------|
| `nii_path`       | `str`  | *required*   | Path to the input (moving) `.nii.gz` file.                                          |
| `reference_path` | `str`  | *required*   | Path to the reference `.nii.gz` file defining the target space.                     |
| `output_path`    | `str`  | *required*   | Directory where the resampled volume will be saved.                                 |
| `interpolation`  | `str`  | `"linear"`   | Interpolation method: `"linear"`, `"nearest"`, or `"bspline"`.                      |
| `debug`          | `bool` | `False`      | If `True`, logs the output path.                                                    |

## Returns

`str` – Path to the saved resampled file.

## Output File

The resampled volume is saved as:
```
<PREFIX>_resampled_to_ref.nii.gz
```

## Interpolation Methods

| Method      | Description                              | Best For                                      |
|-------------|------------------------------------------|-----------------------------------------------|
| `linear`    | Bilinear interpolation (default)         | General-purpose resampling                     |
| `nearest`   | Nearest-neighbor interpolation           | Binary masks and label maps                    |
| `bspline`   | B-spline interpolation                   | Smooth, high-quality resampling of intensities |

## Exceptions

| Exception           | Condition                              |
|---------------------|----------------------------------------|
| `FileNotFoundError` | Input or reference file does not exist |
| `ValueError`        | Unknown interpolation method           |

## Usage Notes

- **Identity Transform**: Uses an identity transform — the volumes must already be roughly in the same physical space
- **Nearest for Masks**: Always use `interpolation="nearest"` for binary masks or label maps to avoid interpolation artifacts
- **Metadata Inheritance**: The output inherits all spatial properties from the reference volume

## Examples

### Basic Usage
Resample a scan to match a template:

```python
from nidataset.transforms import resample_to_reference

resample_to_reference(
    nii_path="patient_scan.nii.gz",
    reference_path="template.nii.gz",
    output_path="resampled/"
)
# Output: resampled/patient_scan_resampled_to_ref.nii.gz
```

### Resample a Mask
Use nearest-neighbor interpolation for a binary mask:

```python
resample_to_reference(
    nii_path="brain_mask.nii.gz",
    reference_path="template.nii.gz",
    output_path="resampled/",
    interpolation="nearest"
)
```

### Multi-Modal Alignment
Align different modalities to the same grid:

```python
reference = "ct_scan.nii.gz"

resample_to_reference("mri_t1.nii.gz", reference, "aligned/", interpolation="bspline")
resample_to_reference("pet_scan.nii.gz", reference, "aligned/", interpolation="linear")
resample_to_reference("seg_mask.nii.gz", reference, "aligned/", interpolation="nearest")
```

## Typical Workflow

```python
from nidataset.transforms import resample_to_reference

# 1. Resample moving image to reference space
out_path = resample_to_reference(
    nii_path="data/moving.nii.gz",
    reference_path="data/fixed.nii.gz",
    output_path="data/aligned/",
    interpolation="linear",
    debug=True
)

# 2. Now both volumes share the same voxel grid
print(f"Resampled volume: {out_path}")
```

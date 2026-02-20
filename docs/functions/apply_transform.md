---
title: apply_transform
parent: Package Functions
nav_order: 1
---

# `apply_transform`

Apply a saved spatial transformation to any NIfTI volume without filename suffix restrictions.

```python
apply_transform(
    nii_path: str,
    transform_path: str,
    reference_path: str,
    output_path: str,
    interpolation: str = "linear",
    debug: bool = False
) -> str
```

## Overview

This is a generic function for applying a previously computed spatial transformation (stored as a `.tfm` file) to any NIfTI volume. Unlike `register_mask` or `register_annotation`, it does not impose filename suffix requirements (e.g., `_mask.nii.gz` or `_bbox.nii.gz`), making it suitable for transforming any volume type.

The output is saved as `<PREFIX>_transformed.nii.gz`.

## Parameters

| Name             | Type   | Default    | Description                                                                          |
|------------------|--------|------------|--------------------------------------------------------------------------------------|
| `nii_path`       | `str`  | *required* | Path to the input `.nii.gz` file to transform.                                      |
| `transform_path` | `str`  | *required* | Path to the `.tfm` transformation file.                                             |
| `reference_path` | `str`  | *required* | Path to the reference `.nii.gz` file defining the target space.                     |
| `output_path`    | `str`  | *required* | Directory where the transformed volume will be saved.                               |
| `interpolation`  | `str`  | `"linear"` | Interpolation method: `"linear"`, `"nearest"`, or `"bspline"`.                      |
| `debug`          | `bool` | `False`    | If `True`, logs the output path.                                                    |

## Returns

`str` – Path to the saved transformed file.

## Output File

The transformed volume is saved as:
```
<PREFIX>_transformed.nii.gz
```

## Exceptions

| Exception           | Condition                              |
|---------------------|----------------------------------------|
| `FileNotFoundError` | Input, transform, or reference file does not exist |
| `ValueError`        | Unknown interpolation method           |

## Usage Notes

- **Generic**: Works with any NIfTI file regardless of naming convention
- **Interpolation**: Use `"nearest"` for label maps/masks, `"linear"` or `"bspline"` for intensity images
- **Transform Files**: Compatible with `.tfm` files produced by `register_CTA` or SimpleITK

## Examples

### Basic Usage
Apply a registration transform to a volume:

```python
from nidataset.transforms import apply_transform

apply_transform(
    nii_path="any_volume.nii.gz",
    transform_path="registration/case001_transformation.tfm",
    reference_path="registration/case001_registered.nii.gz",
    output_path="transformed/"
)
# Output: transformed/any_volume_transformed.nii.gz
```

### Transform a Label Map
Use nearest-neighbor for discrete labels:

```python
apply_transform(
    nii_path="atlas_labels.nii.gz",
    transform_path="case001_transformation.tfm",
    reference_path="case001_registered.nii.gz",
    output_path="transformed/",
    interpolation="nearest"
)
```

### Propagate Multiple Volumes Through Same Transform

```python
from nidataset.transforms import apply_transform

transform = "registration/case001_transformation.tfm"
reference = "registration/case001_registered.nii.gz"

volumes = ["ct_scan.nii.gz", "pet_overlay.nii.gz", "probability_map.nii.gz"]
for vol in volumes:
    apply_transform(vol, transform, reference, "registered/", debug=True)
```

## Typical Workflow

```python
from nidataset.transforms import apply_transform

# 1. After registration, apply the transform to additional volumes
out_path = apply_transform(
    nii_path="data/additional_modality.nii.gz",
    transform_path="registration/patient_transformation.tfm",
    reference_path="registration/patient_registered.nii.gz",
    output_path="data/registered/",
    interpolation="linear",
    debug=True
)

# 2. The output is now in the same space as the registered image
print(f"Transformed volume: {out_path}")
```

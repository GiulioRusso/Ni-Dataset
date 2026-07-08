---
title: rescale_intensity
parent: Quick Operations (nii)
nav_order: 8
---

# `rescale_intensity`

Linearly rescale a NIfTI volume's intensities to a target range.

```python
rescale_intensity(
    nii_path: str,
    output_path: str,
    out_min: float = 0.0,
    out_max: float = 1.0,
    in_min: Optional[float] = None,
    in_max: Optional[float] = None,
    debug: bool = False
) -> str
```

## Overview

Maps intensities linearly to `[out_min, out_max]`. Values are first clipped to the
input range `[in_min, in_max]` (defaulting to the volume's own min/max), then
mapped. This is the plain linear counterpart to
[`intensity_normalization`](../functions/transforms/intensity_normalization.md), which offers
z-score / percentile / histogram modes — use `rescale_intensity` when you just
want a fixed output range (e.g. `0–255` for 8-bit export).

The output is saved as `<PREFIX>_rescaled.nii.gz` in `float32`.

## Parameters

| Name          | Type    | Default    | Description                                                |
|---------------|---------|------------|------------------------------------------------------------|
| `nii_path`    | `str`   | *required* | Path to the input `.nii.gz` file.                         |
| `output_path` | `str`   | *required* | Directory where the rescaled volume is saved.             |
| `out_min`     | `float` | `0.0`      | Lower bound of the output range.                          |
| `out_max`     | `float` | `1.0`      | Upper bound of the output range.                          |
| `in_min`      | `float` | `None`     | Lower bound of the input range (default: data min).       |
| `in_max`      | `float` | `None`     | Upper bound of the input range (default: data max).       |
| `debug`       | `bool`  | `False`    | If `True`, logs the output path and ranges.               |

## Returns

`str` – Path to the saved rescaled file.

## Output File

```
<PREFIX>_rescaled.nii.gz
```

## Exceptions

| Exception           | Condition                                    |
|---------------------|----------------------------------------------|
| `FileNotFoundError` | Input file does not exist                    |
| `ValueError`        | Input file is not a valid `.nii.gz` / `.nii` |
| `ValueError`        | `out_max <= out_min`, or the input range is empty |

## Usage Notes

- **Clipping**: values outside `[in_min, in_max]` are clamped before mapping.
- **Fixed windows**: set `in_min`/`in_max` explicitly to rescale a known intensity
  window consistently across volumes (values outside clamp to the range ends).
- Output is always `float32`.

## Examples

### Rescale to 0–255

```python
from nidataset.transforms import rescale_intensity

rescale_intensity("scan.nii.gz", "out/", out_min=0, out_max=255)
# Output: out/scan_rescaled.nii.gz
```

### Rescale a fixed input window to [0, 1]

```python
rescale_intensity(
    "scan.nii.gz", "out/",
    out_min=0, out_max=1,
    in_min=25, in_max=75,
)
```

## Typical Workflow

```python
from nidataset.transforms import rescale_intensity

# Normalize to a fixed 8-bit range for export / visualization
out = rescale_intensity(
    nii_path="data/scan.nii.gz",
    output_path="data/rescaled/",
    out_min=0,
    out_max=255,
    debug=True,
)
print(f"Rescaled volume: {out}")
```

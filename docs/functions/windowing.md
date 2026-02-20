---
title: windowing
parent: Package Functions
nav_order: 45
---

# `windowing`

Apply CT windowing (window center + window width) to a NIfTI volume for optimized tissue visualization.

```python
windowing(
    nii_path: str,
    output_path: str,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    preset: Optional[str] = None,
    normalize: bool = True,
    debug: bool = False
) -> str
```

## Overview

CT windowing adjusts the displayed intensity range to emphasize specific tissues or pathologies. This function clips the volume intensities to the window range defined by a center and width, and optionally normalizes the result to [0, 1].

You can specify the window parameters directly or use a named preset. The output is saved as `<PREFIX>_windowed.nii.gz` (or `<PREFIX>_windowed_<PRESET>.nii.gz` when using a preset).

## Available Presets

| Preset         | Center | Width | Use Case                             |
|----------------|--------|-------|--------------------------------------|
| `brain`        | 40     | 80    | Brain parenchyma                     |
| `subdural`     | 75     | 215   | Subdural hematoma                    |
| `stroke`       | 40     | 40    | Acute stroke                         |
| `bone`         | 480    | 2500  | Bone structures                      |
| `soft_tissue`  | 50     | 350   | Soft tissue                          |
| `lung`         | -600   | 1500  | Lung parenchyma                      |
| `liver`        | 60     | 160   | Liver                                |
| `mediastinum`  | 50     | 350   | Mediastinal structures               |

## Parameters

| Name            | Type    | Default    | Description                                                                            |
|-----------------|---------|------------|----------------------------------------------------------------------------------------|
| `nii_path`      | `str`   | *required* | Path to the input `.nii.gz` file.                                                     |
| `output_path`   | `str`   | *required* | Directory where the windowed volume will be saved.                                    |
| `window_center` | `float` | `None`     | Center of the window in Hounsfield units.                                             |
| `window_width`  | `float` | `None`     | Width of the window.                                                                  |
| `preset`        | `str`   | `None`     | Named window preset (overrides `window_center` and `window_width`).                   |
| `normalize`     | `bool`  | `True`     | If `True`, scales windowed values to [0, 1].                                          |
| `debug`         | `bool`  | `False`    | If `True`, logs the output path and window parameters.                                |

## Returns

`str` – Path to the saved windowed file.

## Output File

The windowed volume is saved as:
```
<PREFIX>_windowed_<PRESET>.nii.gz   (when using a preset)
<PREFIX>_windowed.nii.gz            (when using custom center/width)
```

## Exceptions

| Exception    | Condition                                                          |
|--------------|--------------------------------------------------------------------|
| `ValueError` | Unknown preset name                                                |
| `ValueError` | Neither preset nor both `window_center`/`window_width` provided    |
| `FileNotFoundError` | Input file does not exist                                 |

## Usage Notes

- **Hounsfield Units**: Input CT volumes should be in Hounsfield units for presets to work correctly
- **Normalization**: When `normalize=True`, the output range is [0, 1]; otherwise, the clipped HU values are preserved
- **Preset Priority**: If `preset` is specified, it overrides any manual `window_center`/`window_width` values

## Examples

### Using a Preset
Apply brain window:

```python
from nidataset.transforms import windowing

windowing(
    nii_path="ct_scan.nii.gz",
    output_path="windowed/",
    preset="brain"
)
# Output: windowed/ct_scan_windowed_brain.nii.gz
```

### Custom Window Parameters
Define your own window:

```python
windowing(
    nii_path="ct_scan.nii.gz",
    output_path="windowed/",
    window_center=50,
    window_width=100,
    normalize=True
)
# Output: windowed/ct_scan_windowed.nii.gz
```

### Multiple Windows for Same Scan
Generate different tissue views:

```python
from nidataset.transforms import windowing

scan = "ct_scan.nii.gz"
output = "views/"

for preset in ["brain", "bone", "soft_tissue"]:
    windowing(scan, output, preset=preset)
# Creates: ct_scan_windowed_brain.nii.gz
#          ct_scan_windowed_bone.nii.gz
#          ct_scan_windowed_soft_tissue.nii.gz
```

### Without Normalization
Keep original Hounsfield unit scale:

```python
windowing(
    nii_path="ct_scan.nii.gz",
    output_path="windowed/",
    preset="brain",
    normalize=False
)
# Output intensity range: [0, 80] HU (for brain window)
```

## Typical Workflow

```python
from nidataset.transforms import windowing

# 1. Apply brain window for stroke analysis
out_path = windowing(
    nii_path="data/ct_head.nii.gz",
    output_path="data/processed/",
    preset="stroke",
    normalize=True,
    debug=True
)

# 2. Use the windowed volume for training or visualization
print(f"Windowed volume saved: {out_path}")
```

---
title: overlay_mask_on_volume
parent: Package Functions
nav_order: 47
---

# `overlay_mask_on_volume`

Create colored overlay images of a segmentation mask blended on top of a grayscale NIfTI volume for each slice along a chosen anatomical axis.

```python
overlay_mask_on_volume(
    nii_path: str,
    mask_path: str,
    output_path: str,
    view: str = "axial",
    alpha: float = 0.4,
    colormap: str = "jet",
    output_format: str = "png",
    debug: bool = False
) -> List[str]
```

## Overview

This function generates RGB overlay images showing how a segmentation mask maps onto the underlying grayscale volume. For each slice along the specified anatomical axis, the grayscale volume is rendered as a base image and the non-zero mask regions are blended on top using a configurable colormap and opacity.

This is useful for:
- Visual quality control of segmentations
- Generating figures for publications and presentations
- Reviewing registration or annotation results
- Creating training data visualizations

Each slice is saved as an individual image file.

## Parameters

| Name            | Type    | Default    | Description                                                                          |
|-----------------|---------|------------|--------------------------------------------------------------------------------------|
| `nii_path`      | `str`   | *required* | Path to the grayscale `.nii.gz` volume.                                             |
| `mask_path`     | `str`   | *required* | Path to the segmentation mask `.nii.gz` volume.                                     |
| `output_path`   | `str`   | *required* | Directory where overlay images will be saved.                                       |
| `view`          | `str`   | `"axial"`  | Anatomical view: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `alpha`         | `float` | `0.4`      | Mask overlay opacity (0.0 = transparent, 1.0 = opaque).                             |
| `colormap`      | `str`   | `"jet"`    | Matplotlib colormap name for the mask overlay.                                      |
| `output_format` | `str`   | `"png"`    | Image format: `"png"`, `"tif"`, or `"jpg"`.                                        |
| `debug`         | `bool`  | `False`    | If `True`, logs the output directory and slice count.                               |

## Returns

`List[str]` – List of saved overlay image paths.

## Output Files

Each slice is saved as:
```
<PREFIX>_overlay_<VIEW>_<NNN>.<FORMAT>
```

**Example**: `brain_overlay_axial_042.png`

## Exceptions

| Exception           | Condition                                |
|---------------------|------------------------------------------|
| `FileNotFoundError` | Volume or mask file does not exist       |
| `ValueError`        | Volume and mask have different shapes    |
| `ValueError`        | Invalid view parameter                   |

## Usage Notes

- **Shape Matching**: Volume and mask must have identical dimensions
- **Colormap**: Any matplotlib colormap name is accepted (e.g., `"jet"`, `"hot"`, `"viridis"`, `"Reds"`)
- **Alpha Blending**: Only non-zero mask voxels are blended; background remains grayscale
- **Progress Display**: Shows a tqdm progress bar during processing
- **Matplotlib Optional**: Falls back to a simple red colormap if matplotlib is not installed

## Examples

### Basic Usage
Generate axial overlay images:

```python
from nidataset.visualization import overlay_mask_on_volume

paths = overlay_mask_on_volume(
    nii_path="brain_scan.nii.gz",
    mask_path="brain_mask.nii.gz",
    output_path="overlays/"
)
print(f"Generated {len(paths)} overlay images")
```

### Custom Colormap and Opacity

```python
paths = overlay_mask_on_volume(
    nii_path="ct_scan.nii.gz",
    mask_path="lesion_mask.nii.gz",
    output_path="overlays/",
    view="coronal",
    alpha=0.6,
    colormap="hot"
)
```

### Multiple Views

```python
from nidataset.visualization import overlay_mask_on_volume

for view in ["axial", "coronal", "sagittal"]:
    overlay_mask_on_volume(
        nii_path="scan.nii.gz",
        mask_path="mask.nii.gz",
        output_path=f"overlays/{view}/",
        view=view
    )
```

## Typical Workflow

```python
from nidataset.visualization import overlay_mask_on_volume

# 1. Generate overlay images for quality control
paths = overlay_mask_on_volume(
    nii_path="data/patient_scan.nii.gz",
    mask_path="data/patient_segmentation.nii.gz",
    output_path="qa/patient_overlays/",
    view="axial",
    alpha=0.4,
    colormap="jet",
    debug=True
)

# 2. Review the generated images
print(f"Review {len(paths)} slices in qa/patient_overlays/")
```

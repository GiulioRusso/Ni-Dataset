---
title: overlay_mask_on_volume_dataset
parent: Package Functions
nav_order: 48
---

# `overlay_mask_on_volume_dataset`

Create mask overlay images for all matching NIfTI volume-mask pairs in two folders.

```python
overlay_mask_on_volume_dataset(
    nii_folder: str,
    mask_folder: str,
    output_path: str,
    view: str = "axial",
    alpha: float = 0.4,
    colormap: str = "jet",
    output_format: str = "png",
    debug: bool = False
) -> int
```

## Overview

This function batch-generates overlay images for an entire dataset. It matches NIfTI files by filename between the volume and mask folders, then generates per-slice overlay images for each matched pair using `overlay_mask_on_volume`. Each case's overlays are organized into a dedicated subdirectory.

## Parameters

| Name            | Type    | Default    | Description                                                                          |
|-----------------|---------|------------|--------------------------------------------------------------------------------------|
| `nii_folder`    | `str`   | *required* | Folder containing grayscale `.nii.gz` volumes.                                      |
| `mask_folder`   | `str`   | *required* | Folder containing mask `.nii.gz` volumes (matched by filename).                     |
| `output_path`   | `str`   | *required* | Output directory. Subdirectories are created per case.                              |
| `view`          | `str`   | `"axial"`  | Anatomical view: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `alpha`         | `float` | `0.4`      | Mask overlay opacity (0.0–1.0).                                                     |
| `colormap`      | `str`   | `"jet"`    | Matplotlib colormap name.                                                           |
| `output_format` | `str`   | `"png"`    | Image format: `"png"`, `"tif"`, or `"jpg"`.                                        |
| `debug`         | `bool`  | `False`    | If `True`, logs total overlay count.                                                |

## Returns

`int` – Total number of overlay images generated across all cases.

## Output Structure

```
output_path/
├── case_001/
│   ├── case_001_overlay_axial_000.png
│   ├── case_001_overlay_axial_001.png
│   └── ...
├── case_002/
│   └── ...
└── ...
```

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | Folder does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Filename Matching**: Volumes and masks are matched by identical filename
- **Unmatched Files**: Volumes without a matching mask are skipped with a warning
- **Error Handling**: Cases that fail to process are skipped
- **Progress Display**: Shows a tqdm progress bar at the case level

## Examples

### Basic Usage
Generate overlays for all cases:

```python
from nidataset.visualization import overlay_mask_on_volume_dataset

total = overlay_mask_on_volume_dataset(
    nii_folder="dataset/scans/",
    mask_folder="dataset/masks/",
    output_path="dataset/overlays/"
)
print(f"Generated {total} overlay images")
```

### Custom Settings

```python
total = overlay_mask_on_volume_dataset(
    nii_folder="scans/",
    mask_folder="segmentations/",
    output_path="qa_overlays/",
    view="coronal",
    alpha=0.5,
    colormap="Reds",
    output_format="jpg",
    debug=True
)
```

## Typical Workflow

```python
from nidataset.visualization import overlay_mask_on_volume_dataset

# 1. Generate overlays for quality control
total = overlay_mask_on_volume_dataset(
    nii_folder="data/processed_scans/",
    mask_folder="data/segmentations/",
    output_path="data/qa_review/",
    view="axial",
    alpha=0.4,
    debug=True
)

# 2. Review the outputs
print(f"Generated {total} images for QA review")
```
